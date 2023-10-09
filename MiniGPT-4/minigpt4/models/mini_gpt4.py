import logging
import random
import torch.nn.functional as F


import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

import re
import copy, gc, itertools
import numpy as np
import time
from minigpt4.common.registry import registry
from minigpt4.models.blip2 import Blip2Base, disabled_train
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import LlamaTokenizer
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from rake_nltk import Rake
import nltk
rake = Rake()
from nltk.tokenize import sent_tokenize



from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

@registry.register_model("mini_gpt4")
class MiniGPT4(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna0": "configs/models/minigpt4_vicuna0.yaml",
        "pretrain_llama2": "configs/models/minigpt4_llama2.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model=".cache/hub/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        has_qformer=True,
        freeze_qformer=True,
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        lora_r=0,
        lora_target_modules=["q_proj", "v_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        apply_lemmatizer=False,
        process_answer=False,
        conversation=False,
        prefix="",
        multiple_choice=False,
        answer_refinement_prompt="",
        answer_processor="vqa",
        paraphrase_prompt="Paraphrase: {}",
        context_paraphrase_prompt="\nQuestion: {}\nBased on the context, rephrase the question.",
        par_model_name='tuner007/pegasus_paraphrase',
        orig_seed = 42,
        par_num_beams = 10,
        num_add_candidates = 10,
        paraphrase = False,
        keyword_pipeline = False,
        reason = False,
        reason_image = False,
        reason_ask = False,
        ext_paraphrase = True,
        verbose = False,
        alt_device= -1,
        constrained=True,
        prefix_answer=False,
        perform_selection=False,
        perform_ensembling=False,
        selection_criterion=''
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        self.has_qformer = has_qformer
        if self.has_qformer:
            print('Loading Q-Former')
            self.Qformer, self.query_tokens = self.init_Qformer(
                num_query_token, self.visual_encoder.num_features
            )
            self.Qformer.cls = None
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.load_from_pretrained(url_or_filename=q_former_model)

            if freeze_qformer:
                for name, param in self.Qformer.named_parameters():
                    param.requires_grad = False
                self.Qformer = self.Qformer.eval()
                self.Qformer.train = disabled_train
                self.query_tokens.requires_grad = False
                logging.info("freeze Qformer")

            img_f_dim = self.Qformer.config.hidden_size
            print('Loading Q-Former Done')
        else:
            img_f_dim = self.visual_encoder.num_features * 4
            print('Do not use Q-Former here.')
        
        self.lm_name = llama_model.lower()
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = "$$"
        

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )

        if lora_r > 0:
            self.llama_model = prepare_model_for_int8_training(self.llama_model)
            loraconfig = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.llama_model = get_peft_model(self.llama_model, loraconfig)
            self.llama_model.print_trainable_parameters()

        else:
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
        print('Loading LLAMA Done')

        self.llama_proj = nn.Linear(
            img_f_dim, self.llama_model.config.hidden_size
        )
        self._apply_lemmatizer = apply_lemmatizer
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        self.process_answer = process_answer
        self.conversation = conversation
        self.prefix = prefix
        self.multiple_choice = multiple_choice
        self.answer_refinement_prompt = answer_refinement_prompt
        self.answer_processor = answer_processor
        self.prefix_answer = prefix_answer

        self.prompt_template = prompt_template.replace('\\n', '\n')
        self.paraphrase = paraphrase
        self.keyword_pipeline = keyword_pipeline
        self.reason = reason
        self.reason_image = reason_image
        self.reason_ask = reason_ask
        self.ext_paraphrase = ext_paraphrase
        self.verbose = verbose
        self.alt_device = alt_device
        self.constrained = constrained

        self.perform_selection = perform_selection
        self.perform_ensembling = perform_ensembling
        self.selection_criterion = selection_criterion

        self.par_model_name = par_model_name
        self.par_num_beams = par_num_beams
        self.num_add_candidates = num_add_candidates
        self.set_alt_device()
        if self.ext_paraphrase: self.init_ext_paraphrase_model()
        else: 
            self.paraphrase_prompt = paraphrase_prompt
            self.context_paraphrase_prompt = context_paraphrase_prompt

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

    def set_alt_device(self):
        torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch_device == 'cuda' and self.alt_device > 0:
            a = torch.full([2], True).cuda(self.alt_device)
            torch_device = a.device
        self.alt_device = torch_device

    def init_ext_paraphrase_model(self):
        torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.par_tokenizer = PegasusTokenizer.from_pretrained(self.par_model_name)
        self.par_model = PegasusForConditionalGeneration.from_pretrained(self.par_model_name).to(torch_device)
        print('Par model device: ' , self.par_model.device)
        
    def init_caption_model(self):
        self.cap_model = BatchPromptCap("vqascore/promptcap-coco-vqa", self.alt_device)

    def ext_paraphrase_response(self, input_text, device):
        self.par_model.to(device)
        if self.num_add_candidates > 0:
            constraint = ['?']
            constraint_ids = self.par_tokenizer(constraint, return_tensors='pt', add_special_tokens=False).input_ids.detach().cpu()
            constraint_ids = constraint_ids[:,1:].tolist()
            batch = self.par_tokenizer(input_text,truncation=True,padding='longest',max_length=60, return_tensors="pt").to(device)
            if self.constrained:
                translated = self.par_model.generate(**batch, force_words_ids=constraint_ids, max_length=60,num_beams=self.par_num_beams, num_return_sequences=self.num_add_candidates, temperature=0.7).detach().cpu()
            else: 
                translated = self.par_model.generate(**batch, max_length=60,num_beams=self.par_num_beams, num_return_sequences=self.num_add_candidates, temperature=0.7).detach().cpu()
            tgt_text = self.par_tokenizer.batch_decode(translated, skip_special_tokens=True)
            p = self.num_add_candidates
            tgt_chunks = [[input_text[i]] +  tgt_text[i*p:i*p + p] for i in range(len(input_text))]
            del batch
        else: tgt_chunks = [[input_text[i]] for i in range(len(input_text))]
        return tgt_chunks
    
    def identify_keywords(self, sent):
        kw = rake.extract_keywords_from_text(sent)
        ranked_phrases = rake.get_ranked_phrases()
        keywords = []
        is_noun = lambda pos: pos[:2] == 'NN' or 'JJ' in pos
        for phrase in ranked_phrases:
            tokenized = nltk.word_tokenize(sent)
            nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos) and word in phrase] 
            keywords.extend(nouns)
        return keywords
    def verbalize_desc(self, desc_dict):
        out = ''
        for key, values in desc_dict.items():
            out += '{}:\n'.format(key)
            for value in values:
                out += '\t- {}\n'.format(value)
        return out
    
    def process_keyword_pipeline(self, model_out, model_inp):
        assert len(model_out) == len(model_inp)
        remaining = [out.replace(inp, '') for out, inp in zip(model_out, model_inp)]
        out = [remain.split('?')[0].split(':')[-1].replace('\n', '').strip(' ') for remain in remaining]
        out = [re.sub(r'[^a-zA-Z\' ]', '', o).strip(' ').capitalize() + '?' for o in out]
        return out
        
    def process_reason(self, model_out, model_inp):
        assert len(model_out) == len(model_inp)
        remaining = [out.replace(inp, '').split(':')[0].split(self.end_sym)[0] for out, inp in zip(model_out, model_inp)]
        out = [[re.sub(r'\d+', '',i).split('\n')[0].strip(' ').lower() for i in remain.split('.')[1:]] for remain in remaining]
        return out

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_img(self, image):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            if self.has_qformer:
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

                inputs_llama = self.llama_proj(query_output.last_hidden_state)
            else:
                image_embeds = image_embeds[:, 1:, :]
                bs, pn, hs = image_embeds.shape
                image_embeds = image_embeds.view(bs, int(pn / 4), int(hs * 4))

                inputs_llama = self.llama_proj(image_embeds)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama

    def get_context_emb(self, prompt, img_list):
        device = img_list[0].device
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(device).input_ids
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.embed_tokens(seg_t) for seg_t in seg_tokens]

        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs

    def prompt_wrap(self, img_embeds, atts_img, prompts):
        if prompts:
            emb_lists = []
            if isinstance(prompts, str):
                prompts = [prompts] * len(img_embeds)
            assert len(prompts) == img_embeds.shape[0]        
            
            
            for each_img_embed, each_prompt in zip(img_embeds, prompts):
                p_before, p_after = each_prompt.split('<ImageHere>')
                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_before_embed = self.embed_tokens(p_before_tokens.input_ids)
                p_after_embed = self.embed_tokens(p_after_tokens.input_ids)
                wrapped_emb = torch.cat([p_before_embed, each_img_embed[None], p_after_embed], dim=1)
                emb_lists.append(wrapped_emb)
                del p_before_tokens, p_after_tokens, p_before_embed, p_after_embed
            emb_lens = [emb.shape[1] for emb in emb_lists]
            pad_emb = self.embed_tokens(torch.tensor(self.llama_tokenizer.pad_token_id, device=img_embeds.device))
            wrapped_embs = pad_emb.expand(len(emb_lens), max(emb_lens), -1).clone()
            wrapped_atts = torch.zeros([len(emb_lens), max(emb_lens)], dtype=torch.int, device=img_embeds.device)
            for i, emb in enumerate(emb_lists):
                wrapped_embs[i, :emb_lens[i]] = emb
                wrapped_atts[i, :emb_lens[i]] = 1
            return wrapped_embs, wrapped_atts
        else:
            return img_embeds, atts_img

    def concat_emb_input_output(self, input_embs, input_atts, output_embs, output_atts):
        input_lens = []
        cat_embs = []
        cat_atts = []
        for i in range(input_embs.size(0)):
            input_len = input_atts[i].sum()
            input_lens.append(input_len)
            cat_embs.append(
                torch.cat([
                    input_embs[i][:input_len],
                    output_embs[i],
                    input_embs[i][input_len:]
                ])
            )
            cat_atts.append(
                torch.cat([
                    input_atts[i][:input_len],
                    output_atts[i],
                    input_atts[i][input_len:]
                ])
            )
        cat_embs = torch.stack(cat_embs)
        cat_atts = torch.stack(cat_atts)
        return cat_embs, cat_atts, input_lens

    def forward(self, samples):
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)

        if self.prompt_list:
            instruction = random.choice(self.prompt_list)
        else:
            instruction = samples["instruction_input"] if "instruction_input" in samples else None

        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, instruction)

        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["answer"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds, attention_mask, input_lens = \
            self.concat_emb_input_output(img_embeds, atts_img, to_regress_embeds, to_regress_tokens.attention_mask)
        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, attention_mask], dim=1)

        part_targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        targets = (
            torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
                       dtype=torch.long).to(image.device).fill_(-100)
        )

        for i, target in enumerate(part_targets):
            targets[i, input_lens[i] + 1:input_lens[i] + len(target) + 1] = target 

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}

    def embed_tokens(self, token_ids):
        if hasattr(self.llama_model.base_model, 'model'): ## lora wrapped model
            embeds = self.llama_model.base_model.model.model.embed_tokens(token_ids)
        else:
            embeds = self.llama_model.base_model.embed_tokens(token_ids)
        return embeds



    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        use_multi_choice_processor=False,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        print('Entered generate function, please check.')
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)   
        self.llama_tokenizer.padding_side = "right" 
        if 'prompt' in samples.keys():
            text = [t + self.end_sym for t in samples["prompt"]]
          
        inputs_embeds, attention_mask = self.prompt_wrap(img_embeds, atts_img, text)

        
    

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=torch.long,
                         device=samples['image'].device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]
    
        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,

                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.llama_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
        print(output_text)
        if self.process_answer:
            if 'vicuna' in self.lm_name:
                if self.answer_processor == 'multiple-choice' or (use_multi_choice_processor): output_text = self.mc_vicuna_answer_processor(output_text) #
                elif self.answer_processor == 'vqa': output_text = self.vqa_vicuna_answer_processor(output_text)
                elif self.answer_processor == 'aok-vqa': output_text = self.vqa_vicuna_answer_processor(output_text)
                else: assert False
            elif 'llama' in self.lm_name:
                output_text = self.llama_answer_processor(output_text)
                print(output_text)
                import pdb; pdb.set_trace()

        return output_text
    
    def run_entity_keyword_pipeline(self, keywords, samples, num_beams, inference_method, max_len, min_len, num_ans_candidates, answer_list, length_penalty, **kwargs):
        start = time.time()
        ## Generate descriptors
        description_lookup = {}

        counter = 0
        
        longest = max([len(k) for k in keywords])
        description_lookup = {k:{} for k in range(len(keywords))}
        for l in range(longest):
            entities = []
            image_stack = []
            lookup = []
            for k in range(len(keywords)):
                if l < len(keywords[k]):
                    lookup.append(k)
                    entities.append(keywords[k][l])
                    image_stack.append(samples['image'][k])
            messages = ['### Human: <Img><ImageHere></Img>### Human: What can you tell me about the {} in this image?'.format(entity) for entity in entities]
            descs, _, _ = self.converse({'image': torch.stack(image_stack, dim=0)}, num_beams,inference_method,max_len,min_len,num_ans_candidates,answer_list, length_penalty, reply=True, messages=messages, use_multi_choice_processor=False, **kwargs)
            for k in lookup:
                entity = entities.pop(0)
                desc = descs.pop(0)
                if not "I'm sorry" in desc: description_lookup[k][entity] = sent_tokenize(desc)

        ## Modify the question using descriptions:

        mod_questions = []
        mod_ques_prompt = '### Human: You are given a question about an image. Modify the question by adding descreptive phrases to entities using different details provided below. Both original and modified questions MUST have similar meaning.'
        mod_ques_prompt += '\n### Human: Question: {}\nDetails:\n{}### Assistant: Modified Question: {}'.format('What is the man wearing?', self.verbalize_desc({'man': ['He is standing on the sidewalk']}), 'What is the man who is standing on the sidewalk wearing?')
        mod_ques_prompt += '\n### Human: Question: {}\nDetails:\n{}### Assistant: Modified Question: {}'.format('Are there any flowers?', self.verbalize_desc({'flowers': ['There is flowers are in a vase', 'The vase is blue in color and sitting on a table.']}), 'Are there any flowers in the vase on the table?')
        mod_ques_prompt += '\n### Human: Question: {}\nDetails:\n{}### Assistant: Modified Question: '
        mod_inp = [mod_ques_prompt.format(samples['text_input'][i], self.verbalize_desc(description_lookup[i])) for i in range(samples['image'].shape[0])]
        mod_tokens = self.llama_tokenizer(mod_inp, padding="longest", return_tensors="pt").to(samples['image'].device)

        
        if self.num_add_candidates > 0:
            alt_questions = self.llama_model.generate(**mod_tokens, max_new_tokens=40, num_return_sequences=self.num_add_candidates, do_sample=True, top_p=0.95)
            alt_questions  = self.llama_tokenizer.batch_decode(alt_questions, skip_special_tokens=True) 
            del mod_tokens
            rephrased_text_input = self.process_keyword_pipeline(alt_questions, list(np.repeat(mod_inp, self.num_add_candidates)))
            tgt_text = rephrased_text_input 
            tgt_text = [t.lower().replace('yes', '').replace('no', '').strip(' ').capitalize() for t in tgt_text]
            p = self.num_add_candidates 
            output_text = [[samples['text_input'][i]] +  tgt_text[i*p:i*p + p] for i in range(len(samples['text_input']))]
        else:
            output_text = [[text] for text in samples['text_input']]
            rephrased_text_input = []

        return output_text, rephrased_text_input, description_lookup



    def converse(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        reply = False,
        messages = [],
        use_multi_choice_processor = False,
        **kwargs     
    ):
        image = samples["image"]
        if self.conversation: 
            img_embeds, atts_img = self.encode_img(image)
            if not reply:
                if self.prompt_template:
                    if not self.multiple_choice:
                        text = [self.prompt_template.format(question) for question in samples["text_input"]]
                    else:
                        if len(samples['choices']) == len(samples['text_input'])//(self.num_add_candidates+1):
                            samples['choices'] = [[choices]*(self.num_add_candidates + 1) for choices in samples['choices']]
                            samples['choices'] =  list(itertools.chain.from_iterable(samples["choices"]))
                        text = [self.prompt_template.format(question, op_1, op_2, op_3, op_4) for question, [op_1, op_2, op_3, op_4] in zip(samples["text_input"], samples["choices"])]            
                else:
                    text = samples['text_input']   
            elif reply and len(messages):
                text = messages 
            else: assert False
            text = [t + self.end_sym for t in text]
            if img_embeds.shape[0] != len(text):
                img_embeds = torch.repeat_interleave(img_embeds, len(text)//img_embeds.shape[0], dim=0)
                atts_img = torch.repeat_interleave(atts_img, len(text)//img_embeds.shape[0], dim=0)
            inputs_embeds, attention_mask = self.prompt_wrap(img_embeds, atts_img, text)

        batch_size = len(samples['image'])

        bos = torch.ones([batch_size, 1],
                        dtype=torch.long,
                        device=samples['image'].device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)

        atts_bos = attention_mask[:, :1] 
        bos_embeds = torch.repeat_interleave(bos_embeds, inputs_embeds.shape[0]//batch_size, dim=0)
        
        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, attention_mask], dim=1)

        del bos_embeds, atts_bos, bos
        del img_embeds, atts_img

        with self.maybe_autocast():
            outputs = self.llama_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                length_penalty=length_penalty,
            )
            output_text = self.llama_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
        
        del outputs, inputs_embeds, attention_mask
        gc.collect()

        if self.process_answer:
            if 'vicuna' in self.lm_name:
                if self.answer_processor == 'multiple-choice' or (use_multi_choice_processor): 
                    output_text = self.mc_vicuna_answer_processor(output_text) #
                elif self.answer_processor == 'vqa': output_text = self.vqa_vicuna_answer_processor(output_text)
                elif self.answer_processor == 'aok-vqa': output_text = self.vqa_vicuna_answer_processor(output_text)
                else: assert False
            elif 'llama' in self.lm_name:
                output_text = self.llama_answer_processor(output_text)
                import pdb; pdb.set_trace()

        
        if self.verbose and self.num_add_candidates: return output_text, [], text
        else: return output_text, [], text
        

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        **kwargs
    ):
        image = samples["image"]
        batch_size = len(samples['image'])
        self.llama_tokenizer.padding_side = "right"
        samples['text_input'] = [t.capitalize() for t in samples['text_input']]
        keywords = list(map(self.identify_keywords, samples['text_input']))

        paraphrase_text_input, keyword_pipeline_text_input, reason_text_input = [],[],[]
        par_rephrased_inputs, keyword_pipeline_rephrased_inputs, reason_rephrased_inputs = [], [], []

        if self.paraphrase and self.num_add_candidates > 0:
            par_device = image.device if True else self.alt_device #if self.use_caption else
            if self.ext_paraphrase: paraphrase_text_input = self.ext_paraphrase_response(samples["text_input"], par_device)
            else: 
                paraphrase_text_input = self.int_paraphrase_response(samples["text_input"], self.context_paraphrase_prompt, image.device, img_embeds, atts_img)
        
        if self.keyword_pipeline:
            keyword_pipeline_text_input, keyword_pipeline_rephrased_inputs, desc_lookup = self.run_entity_keyword_pipeline(keywords, samples, num_beams, inference_method, max_len, min_len, num_ans_candidates, answer_list, length_penalty, **kwargs)
        
        if self.reason:
            ## Generate descriptors
            r_messages = ['### Human: <Img><ImageHere></Img>### Human: Describe the image in a couple of sentences.']*len(samples['image'])
            captions, _, _ = self.converse(samples,num_beams,inference_method,25,min_len,num_ans_candidates,answer_list,length_penalty, reply=True, messages=r_messages, use_multi_choice_processor=False, **kwargs)
            int_text, _, int_last_messages = self.converse(samples,num_beams,inference_method,max_len,min_len,num_ans_candidates,answer_list,length_penalty, False, [], **kwargs)
            int_messages = [last + "\n" + " Assistant: {}\n### Human: Give a brief explanation for your answer.".format(ans) for ans,last in zip(int_text, int_last_messages)]
            exp_text, _, exp_last_messages = self.converse(samples,num_beams,inference_method,60,min_len,num_ans_candidates,answer_list,length_penalty, reply=True, messages=int_messages, use_multi_choice_processor=False, **kwargs)
            r_prompt = '### Human: You are given a description of an image, a question and its response below.\nImage Content: {}\nQuestion: {}\nResponse: {}. List up to 3 objects or from the image were relevant to answering the question? Describe each object ONLY 2-3 words.### Assistant: Enumerated list of top-3 relevant objects used: '
            r_inp = [r_prompt.format(captions[i], samples['text_input'][i], exp_text[i]) for i in range(samples['image'].shape[0])]
            r_tokens = self.llama_tokenizer(r_inp, padding="longest", return_tensors="pt").to(samples['image'].device)
            reasons = self.llama_model.generate(**r_tokens, max_new_tokens=30, num_return_sequences=1, num_beams=num_beams)
            del r_tokens
            reasons  = self.llama_tokenizer.batch_decode(reasons, skip_special_tokens=True) 
            other_entities = self.process_reason(reasons, r_inp)
            reason_text_input, reason_rephrased_inputs, desc_lookup = self.run_entity_keyword_pipeline(keywords, samples, num_beams, inference_method, max_len, min_len, num_ans_candidates, answer_list, length_penalty, **kwargs)
        
        ## Combining
        all_text_input = []
        all_rephrases = []
        for i in range(len(samples['text_input'])):
            current = []
            rephrasing = []
            if self.num_add_candidates > 0:
                if self.paraphrase: current.extend([q for q in paraphrase_text_input[i][1:] if '?' in q])
                if self.keyword_pipeline: current.extend([q for q in keyword_pipeline_text_input[i][1:] if '?' in q])
                if self.reason: current.extend([q for q in reason_text_input[i][1:] if '?' in q])
                current = list(set(current))
                if not len(current): 
                    rephrasing = [samples['text_input'][i]] * self.num_add_candidates
                try:
                    rephrasing = random.sample(current, k=self.num_add_candidates)
                except: 
                    rephrasing = random.choices([samples['text_input'][i]] + current, k=self.num_add_candidates)
            all_text_input.append([samples['text_input'][i]] + rephrasing)
            all_rephrases.append(rephrasing)
            
        samples['text_input'] = list(itertools.chain.from_iterable(all_text_input))
        samples['rephrased_text_input'] = all_rephrases
        start = time.time()
        
        output_text, _, last_messages = self.converse(samples,num_beams,inference_method,max_len,min_len,num_ans_candidates,answer_list,length_penalty, False, [], False, **kwargs)
        if self.answer_refinement_prompt:
            if not self.multiple_choice:
                new_messages = [self.answer_refinement_prompt.format(ans) for ans in output_text]
            else:
                new_messages = [self.answer_refinement_prompt.format(ans, op1, op2, op3, op4) for ans, [op1, op2, op3, op4] in zip(output_text, samples['choices'])]

            messages = [last + "\n" + new for last, new in zip(last_messages, new_messages) ]
            small_max_len = 12 if not self.multiple_choice else 24
            use_multi_choice_processor = False
            output_text, _, last_messages = self.converse(samples,num_beams,inference_method,small_max_len,min_len,num_ans_candidates,answer_list,length_penalty, reply=True, messages=messages, use_multi_choice_processor=use_multi_choice_processor, **kwargs)
            output_text = self.mc_vicuna_answer_processor(output_text)
        output_text = [out.rstrip('.') for out in output_text]
        if self._apply_lemmatizer:
            output_text = self._lemmatize(output_text)

        ## Perform selection
        if self.selection_criterion == 'Qconf':
            template = '### Human: <Img><ImageHere></Img>### Human: Ask me a question based on the image.### Assistant: {}'
            prompt = [template.format(ques) for ques in samples['text_input']]
            output_scores = self.compute_score(torch.repeat_interleave(samples['image'], self.num_add_candidates + 1, dim=0), prompt, samples['text_input'])
        elif self.selection_criterion == 'Aconf':
            if not self.multiple_choice:
                template = '### Human: <Img><ImageHere></Img>### Human: Based on the image, provide a very brief answer to the question below.\nQuestion: {}### Assistant: {}'
                prompt = [template.format(ques, ans) for ques, ans in zip(samples['text_input'], output_text)]
            else:
                template = '### Human: <Img><ImageHere></Img>### Human: Based on the image, select the correct answer to the question from the options.\nQuestion: {}\nOptions: A. {}, B. {}, C. {}, D. {}.### Assistant: Option {}'
                prompt = [template.format(ques, op1, op2, op3, op4, ans) for ques,[op1, op2, op3, op4], ans in zip(samples['text_input'], samples['choices'], output_text)]
            output_scores = self.compute_score(torch.repeat_interleave(samples['image'], self.num_add_candidates + 1, dim=0), prompt, output_text)
        elif self.selection_criterion == 'PlausibleAconf':
            if not self.multiple_choice:
                p = self.num_add_candidates + 1
                int_text = [output_text[i*p: i*p + p] for i in range(batch_size)]
                refined_text = [[random.shuffle(list(set(int)), random.random)] * (self.num_add_candidates + 1) for int in int_text]
                refined_text = sum(refined_text, [])
                template = "template = '### Human: <Img><ImageHere></Img>### Human: Based on the image, provide a very brief answer to the question below.\nQuestion: {}\n### Human: Here are some plausible answers: {}.### Assistant: {}'"
                prompt = [template.format(ques, str(plaus_ans).lstrip('[').rstrip(']'), ans) for ques, plaus_ans, ans in zip(samples['text_input'], refined_text, output_text)]
            else:
                assert False
            output_scores = self.compute_score(torch.repeat_interleave(samples['image'], self.num_add_candidates + 1, dim=0), prompt, output_text)

        p = self.num_add_candidates + 1
        output_text = [output_text[i*p: i*p + p] for i in range(batch_size)]
        output_scores = np.array([output_scores[i*p: i*p + p] for i in range(batch_size)])


        if self.perform_selection and len(self.selection_criterion):
            samples['text_input'] = [samples['text_input'][i*p:i*p + p] for i in range(batch_size)]
            if self.prefix_answer: 
                output_scores = output_scores[:, 1:]
            select_idx = np.argmax(output_scores, axis=1).tolist()
            if self.prefix_answer: 
                output_text = [[output_text[s][0], output_text[s][idx+1]] for s,idx in enumerate(select_idx)]
                samples['text_input'] = [[samples['text_input'][s][0], samples['text_input'][s][idx+1]] for s,idx in enumerate(select_idx)]
            else:
                output_text = [output_text[s][idx] for s,idx in enumerate(select_idx)]
                samples['text_input'] = [samples['text_input'][s][idx] for s,idx in enumerate(select_idx)]


            output_scores = np.max(output_scores, axis=1)
            output_scores = output_scores.tolist()
        elif self.perform_ensembling:
            assert len(output_text) == batch_size
            output_scores = []
        else: output_scores = []

        if self.verbose and self.num_add_candidates: return output_text, [], [] #samples['rephrased_text_input'].#output_scores.tolist()
        else: return output_text, [], []

    def compute_score(self, image, text, label):
        assert image.shape[0] == len(text)
        labels = self.llama_tokenizer(label, add_special_tokens=False).input_ids
        img_embeds, atts_img = self.encode_img(image)
        inputs_embeds, attention_mask = self.prompt_wrap(img_embeds, atts_img, text)
        batch_size = len(text)
        bos = torch.ones([batch_size, 1],
                        dtype=torch.long,
                        device=image.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)

        atts_bos = attention_mask[:, :1] 
        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, attention_mask], dim=1)

        logits = self.llama_model.forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits.detach().cpu()
        all_logprobs = torch.log(torch.softmax(logits.float(), dim=-1))
        filter_sums = []
        for row, label in zip(all_logprobs[:,:-1,:], labels):
            row = row[-len(label):, :].float().numpy() 
            vocab_size = row.shape[-1]
            if len(label):
                loc = F.one_hot(torch.tensor(label), num_classes=vocab_size).numpy().astype(bool)
                summed_logprob = np.sum(row, where = loc)
            else: summed_logprob = -100 
            filter_sums.append(summed_logprob / max(1, len(label)))
        return np.array(filter_sums)
    
    def compute_binary_score(self):
        return

    
    def vqa_vicuna_answer_processor(self, answers):
        answers = [a.strip(self.end_sym) for a in answers]
        answers = [a.split(':')[1].strip(' ') if ':' in a else a.strip(' ') for a in answers]
        answers = [a.split(self.end_sym)[0].strip('\n') for a in answers]
        return answers

    def aok_vqa_vicuna_answer_processor(self, answers):
        answers = [a.strip('###') for a in answers]
        answers = [a.split(':')[1].strip(' ') if ':' in a else a.strip(' ') for a in answers]
        answers = [a.split('###')[0].strip('\n') for a in answers]
        return answers

    def mc_vicuna_answer_processor(self, answers):
        answers = [re.findall(r"[A-D]\.", a)[0].rstrip('.') if ('A.' in a or 'B.' in a or 'C.' in a or 'D.' in a) else random.choice(['A', 'B', 'C', 'D']) for a in answers]
        return answers

    
    def llama_answer_processor(self, answers):
        answers = [a.strip(self.end_sym).strip('\n') for a in answers]
        answers = [a.split(':')[-1].strip(' ') for a in answers]
        answers = [a.split()[0].rstrip(',').rstrip('.') if len(a) else a for a in answers]
        return answers

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", ".cache/hub/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        has_qformer = cfg.get("has_qformer", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)
        process_answer = cfg.get("process_answer", False)
        multiple_choice = cfg.get("multiple_choice", False)
        answer_refinement_prompt = cfg.get("answer_refinement_prompt", "")
        answer_processor = cfg.get("answer_processor", "vqa")
        conversation = cfg.get("conversation", False)
        prefix = cfg.get('prefix', '')

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        lora_r = cfg.get("lora_r", 0)
        lora_alpha = cfg.get("lora_alpha", 32)

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            has_qformer=has_qformer,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            conversation=conversation,
            prefix=prefix,
            process_answer=process_answer,
            multiple_choice=multiple_choice,
            answer_refinement_prompt=answer_refinement_prompt,
            answer_processor=answer_processor,
            par_num_beams = cfg.get('par_num_beams', 5),
            num_add_candidates = cfg.get('num_add_candidates', 0),
            paraphrase = cfg.get('paraphrase', False),
            keyword_pipeline = cfg.get('keyword_pipeline', False),
            reason = cfg.get('reason', False),
            reason_image = cfg.get('reason_image', True),
            reason_ask = cfg.get('reason_ask', False),
            ext_paraphrase = cfg.get('ext_paraphrase', True),
            verbose = cfg.get('verbose', True),
            alt_device = cfg.get('alt_device', -1),
            prefix_answer = cfg.get('prefix_answer', True),
            perform_selection = cfg.get('perform_selection', False),
            perform_ensembling = cfg.get('perform_ensembling', False),
            selection_criterion = cfg.get('selection_criterion', 'Qconf')
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
