"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast
import itertools
import numpy as np
import gc, copy
from collections import Counter
import time
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from transformers import BertTokenizer
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from PIL import Image
import spacy
from rake_nltk import Rake
import nltk
rake = Rake()
import random 

@registry.register_model("blip2_t5_par")
class Blip2T5Par(Blip2Base):
    """
    BLIP2 T5 model.
    Supported model types:
        - pretrain_flant5xl: pretrained model with FlanT5-XL
        - pretrain_flant5xl_vitL: pretrained model with FlanT5-XL
        - pretrain_flant5xxl: pretrained model with FlanT5-XXL
        - caption_coco_flant5xl: fintuned image captioning model with FlanT5-XL
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_t5", "pretrain_flant5xl")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",
        "pretrain_flant5xl_vitL": "configs/models/blip2/blip2_pretrain_flant5xl_vitL.yaml",
        "pretrain_flant5xxl": "configs/models/blip2/blip2_pretrain_flant5xxl.yaml",
        "caption_coco_flant5xl": "configs/models/blip2/blip2_caption_flant5xl.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        t5_model="google/flan-t5-xl",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
        multiple_choice=False,
        paraphrase_prompt="Paraphrase: {}",
          context_paraphrase_prompt = "\nSentence: {}.\nWhat entity is the subject of the sentence?",
        # context_paraphrase_prompt = "\nQuestion: {}. Your task is to modify the question by adding some information from the context. Perform this task by following a sequence of sub-tasks.\n1. What entity is the subject of the question?\n2. Describe information present about this entity in context via a short phrase.\n3. Fuse the question by adding this detail about the entity (as a phrase) to the original question structure.\n\nResponse:",
        # context_paraphrase_prompt="\nQuestion: {}\nModify the question by adding some information from the context.",
        par_model_name='tuner007/pegasus_paraphrase',
        orig_seed = 42,
        par_num_beams = 10,
        num_add_candidates = 10,
        paraphrase = False,
        ext_paraphrase = True,
        perform_selection = False,
        calibrate = False,
        dropout_aggregate=False,
        selection_criterion = 'Aconf',
        perform_ensembling = True,
        verbose=True,
        constrained=True,
        use_caption=False,
        use_promptcap=False,
        alt_device = 0,
        caption_prompt='Please describe this image according to the given question: {}',
        keyword_pipeline = False,
        reason = False,
        prefix_answer = False,
        
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config
        )

        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            param.data = param.data.bfloat16()

        self.t5_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.t5_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        self.orig_seed = orig_seed

        self._apply_lemmatizer = apply_lemmatizer
        self.multiple_choice = multiple_choice
        self._lemmatizer = None
        self.paraphrase = paraphrase
        self.ext_paraphrase = ext_paraphrase
        self.perform_selection = perform_selection
        self.calibrate = calibrate
        self.dropout_aggregate = dropout_aggregate
        self.selection_criterion = selection_criterion
        self.perform_ensembling = perform_ensembling
        self.verbose = verbose
        self.constrained = constrained
        self.use_caption = use_caption
        self.caption_prompt = caption_prompt
        self.alt_device = alt_device
        self.keyword_pipeline = keyword_pipeline
        self.reason = reason
        self.prefix_answer = prefix_answer

        # External paraphrasing part
        self.par_model_name = par_model_name
        self.par_num_beams = par_num_beams
        self.num_add_candidates = num_add_candidates
        self.set_alt_device()
        if self.ext_paraphrase: self.init_ext_paraphrase_model()
        else: 
            self.paraphrase_prompt = paraphrase_prompt
            self.context_paraphrase_prompt = context_paraphrase_prompt
        print('Config: Selection: {}, Criterion: {}, Calibrate {}'.format(self.perform_selection, self.selection_criterion, self.calibrate))

    def set_alt_device(self):
        torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch_device == 'cuda' and self.alt_device > 0:
            a = torch.full([2], True).cuda(self.alt_device)
            torch_device = a.device
        self.alt_device = torch_device

    def init_ext_paraphrase_model(self):
        torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not self.use_caption: torch_device = self.alt_device
        self.par_tokenizer = PegasusTokenizer.from_pretrained(self.par_model_name)
        self.par_model = PegasusForConditionalGeneration.from_pretrained(self.par_model_name).to(torch_device)
        print('Par model device: ' , self.par_model.device)
        

    def ext_paraphrase_response(self, input_text, device, keywords=[]):
        self.par_model.to(device)
        if self.num_add_candidates > 0:
            if len(keywords):
                constraint = [key for key in keywords]
                
                constraint_ids = []
                for const in constraint:
                    const_ids = self.t5_tokenizer(const, add_special_tokens=False).input_ids
                    const_ids = const_ids + [[58]]
                    constraint_ids.append(const_ids) 
            else:
                constraint_ids = [[58]*len(input_text)] # token for '?
            batch = self.par_tokenizer(input_text,truncation=True,padding='longest',max_length=60, return_tensors="pt").to(device)
            if self.constrained:
                tgt_text = []
                for i in range(len(input_text)):
                    small_batch = {}
                    small_batch['input_ids'] = batch.input_ids[i].unsqueeze(0)
                    small_batch['attention_mask'] = batch.attention_mask[i].unsqueeze(0)
                    translated = self.par_model.generate(**small_batch, force_words_ids=constraint_ids[i], max_length=60,num_beams=self.par_num_beams, num_return_sequences=self.num_add_candidates, temperature=0.7, repetition_penalty=1.5).detach().cpu()
                    int_text = self.par_tokenizer.batch_decode(translated, skip_special_tokens=True)
                    tgt_text.append(int_text[0])
            else: 
                translated = self.par_model.generate(**batch, max_length=60,num_beams=self.par_num_beams, num_return_sequences=self.num_add_candidates, temperature=0.7).detach().cpu()
                tgt_text = self.par_tokenizer.batch_decode(translated, skip_special_tokens=True)
            p = self.num_add_candidates
            tgt_chunks = [[input_text[i]] +  tgt_text[i*p:i*p + p] for i in range(len(input_text))]
            del batch
            gc.collect()
        else: tgt_chunks = [[input_text[i]] for i in range(len(input_text))]
        return tgt_chunks
    
    def int_paraphrase_response(self, input_text, prompt, device, context=None, context_atts=None, keywords=[]):
        prompted_input_text = [prompt.format(text) for text in input_text]
        if len(keywords):
            constraint = [key for key in keywords]
            
            constraint_ids = []
            for const in constraint:
                const_ids = self.t5_tokenizer(const, add_special_tokens=False).input_ids
                const_ids = const_ids + [[58]]
                constraint_ids.append(const_ids)            
        else:
            constraint = ['?']
            constraint_ids = self.t5_tokenizer(constraint, return_tensors='pt', add_special_tokens=False).input_ids.detach().cpu()
            constraint_ids = constraint_ids[:,1:].tolist()
        if context is not None:
            prefix = 'Context: '
            prefix_tokens = self.t5_tokenizer([prefix], return_tensors='pt').to(device)
            prefix_ids = torch.repeat_interleave(prefix_tokens.input_ids[:,:-1], len(context), dim=0)
            prefix_embeds = self.t5_model.encoder.embed_tokens(prefix_ids) 
            prefix_atts = torch.repeat_interleave(prefix_tokens.attention_mask[:,:-1], len(context), dim=0)

        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(prompted_input_text, padding="longest", truncation=True, max_length=self.max_txt_len, return_tensors="pt").to(device)
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            encoder_atts = input_tokens.attention_mask
            if context is not None:
                inputs_embeds = torch.cat([prefix_embeds, context, inputs_embeds], dim=1)
                encoder_atts = torch.cat([prefix_atts, context_atts, encoder_atts], dim=1)
            tgt_text = []
            for i in range(inputs_embeds.shape[0]):
                translated = self.t5_model.generate(
                    inputs_embeds=inputs_embeds[i,:,:].unsqueeze(0),
                    attention_mask=encoder_atts[i,:].unsqueeze(0),
                    force_words_ids=constraint_ids[i],
                    temperature=0.9,
                    num_beams=self.par_num_beams,
                    max_new_tokens=50,
                    num_return_sequences=self.num_add_candidates,
                )
                int_text = self.t5_tokenizer.batch_decode(translated, skip_special_tokens=True)
                tgt_text.append(int_text[0])

            p = self.num_add_candidates
            tgt_chunks = [[input_text[i]] +  tgt_text[i*p:i*p + p] for i in range(len(input_text))]
            return tgt_chunks
        
    def ensemble(self, answers):
        adict = Counter(answers)
        return [adict.most_common(1)[0][0]]
    
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
        for key, value in desc_dict.items():
            out += '- {}: {}\n'.format(key, value)
        return out.rstrip()

    def forward(self, samples):
        image = samples["image"]

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                samples["text_input"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            output_tokens = self.t5_tokenizer(
                samples["text_output"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

            return {"loss": loss}

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
        constrained=False,
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
        image = samples["image"]

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)
        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        if isinstance(prompt, str):
            prompt = [prompt] * samples['image'].size(0)
        else:
            assert len(prompt) == samples['image'].size(
                0
            ), "The number of prompts must be equal to the batch size."

        input_tokens = self.t5_tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).to(image.device)
        del image
        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        with self.maybe_autocast(dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
            if not constrained:
                outputs = self.t5_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=encoder_atts,
                    do_sample=use_nucleus_sampling,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_new_tokens=max_length,
                    min_length=min_length,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                )
                output_text = self.t5_tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
            else:
                outputs = self.t5_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=encoder_atts,
                    do_sample=False,
                    force_words_ids=[[58]],
                    # top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_new_tokens=max_length,
                    min_length=min_length,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                )
                output_text = self.t5_tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )

        del outputs, inputs_embeds, encoder_atts
        gc.collect()
        return output_text
    
    def run_entity_keyword_pipeline(self, keywords, samples, num_beams):
        ## Generate descriptors
        description_lookup = {}

        flattened_keywords = list(itertools.chain.from_iterable(keywords))
        counter = 0
        image = []
        flattened_keywords = []
        for k, keyword in enumerate(keywords):
            for key in keyword:
                flattened_keywords.append(key)
                image.append(samples['image'][k])
        image = torch.stack(image, dim=0)
        messages = ['What can you tell me about the {} in this image?'.format(entity) for entity in flattened_keywords]
        descs = self.generate(samples={'image': image, 'prompt': messages})
        for k, keyword in enumerate(keywords):
            ent_desc = {}
            for entity in keyword:
                desc = descs[counter]
                ent_desc[entity] = desc
                counter += 1
            description_lookup[k] = ent_desc

        mod_questions = []
        mod_ques_prompt = 'You are given a question about an image. Modify the question by adding descreptive phrases to entities based on the provided details. Both original and modified questions MUST have similar meaning.'
        mod_ques_prompt += '\nQuestion: {}\nDetails:\n{}Modified Question: {}'.format('What is the man wearing?', self.verbalize_desc({'man': 'he is standing on the sidewalk'}), 'What is the man who is standing on the sidewalk wearing?')
        mod_ques_prompt += '\nQuestion: {}\nDetails:\n{}Modified Question: {}'.format('How many flowers are there?', self.verbalize_desc({'flowers': 'there is vase containing flowers on the table'}), 'How many flowers are in the vase on the table?')
        mod_ques_prompt += '\nQuestion: {}\nDetails:\n{}Modified Question:'
        mod_inp = [mod_ques_prompt.format(samples['text_input'][i], self.verbalize_desc(description_lookup[i])) for i in range(samples['image'].shape[0])]
        
        mod_tokens = self.t5_tokenizer(mod_inp, padding="longest", return_tensors="pt").to(samples['image'].device)

        if self.num_add_candidates > 0:
            alt_questions = self.t5_model.generate(**mod_tokens, do_sample=True, top_p=0.95, max_new_tokens=20, num_return_sequences=self.num_add_candidates)
            rephrased_text_input = self.t5_tokenizer.batch_decode(alt_questions, skip_special_tokens=True)
            tgt_text = rephrased_text_input 
            p = self.num_add_candidates 
            output_text = [[samples['text_input'][i]] +  tgt_text[i*p:i*p + p] for i in range(len(samples['text_input']))]
        else:
            output_text = [[text] for text in samples['text_input']]

        return output_text, rephrased_text_input, description_lookup

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
        samples['text_input'] = [t.capitalize() for t in samples['text_input']]
        
        if self.multiple_choice: 
            assert 'choices' in samples.keys(), "Wrong dataset format for multiple choice."
            samples['orig_choices'] =  copy.deepcopy(samples['choices'])
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        # Text-grounding in Q-former
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        bs = len(samples['image'])
        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)
        if self.perform_ensembling:
            orig_inputs_t5 = inputs_t5
            orig_atts_t5 = atts_t5
        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]
        keywords = list(map(self.identify_keywords, samples['text_input']))

        paraphrase_text_input, keyword_pipeline_text_input, reason_text_input = [],[],[]
        par_rephrased_inputs, keyword_pipeline_rephrased_inputs, reason_rephrased_inputs = [], [], []

        if self.prompt:
            if self.multiple_choice:
                samples['choices'] = [[choices]*(self.num_add_candidates + 1) for choices in samples['choices']]
                samples['choices'] =  list(itertools.chain.from_iterable(samples["choices"]))
            if self.paraphrase:
                par_device = image.device if self.use_caption else self.alt_device
                if self.ext_paraphrase: par_rephrased_inputs = self.ext_paraphrase_response(samples["text_input"], par_device)
                else: 
                    par_rephrased_inputs = self.int_paraphrase_response(samples["text_input"], self.context_paraphrase_prompt, image.device, inputs_t5, atts_t5, keywords=keywords)
                paraphrase_text_input = par_rephrased_inputs #list(itertools.chain.from_iterable(par_rephrased_inputs))
                start = time.time()

            if self.keyword_pipeline:
                keyword_pipeline_text_input, keyword_pipeline_rephrased_inputs, pipe_desc_lookup = self.run_entity_keyword_pipeline(keywords, samples, num_beams)

            
            if self.reason:
                ## Generate descriptors
                other_entities = []
                for i in range(samples['image'].shape[0]):
                    temp = self.generate(samples={'image': samples['image'][i].unsqueeze(0), 'prompt': 'Question:{}\nWhich all entities or objects from this image would I need to observe to answer this question?'.format(samples['text_input'][i])})
                    other_entities.extend([t.split(', ') for t in temp])
                    # ent_desc = {}
                reason_text_input, reason_rephrased_inputs, reason_desc_lookup = self.run_entity_keyword_pipeline(other_entities, samples, num_beams)

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
                    if not len(current): rephrasing = [samples['text_input'][i]] * self.num_add_candidates
                    try:
                        rephrasing = random.sample(current, k=self.num_add_candidates)
                    except: 
                        rephrasing = random.choices([samples['text_input'][i]] + current, k=self.num_add_candidates)
                all_text_input.append([samples['text_input'][i]] + rephrasing)
                all_rephrases.append(rephrasing)
                
            samples['text_input'] = list(itertools.chain.from_iterable(all_text_input))
            samples['rephrased_text_input'] = all_rephrases

            if self.use_caption: 
                
                samples['captions'] = self.generate(samples={'image': samples['image']})
                samples['captions'] = np.repeat(samples['captions'], self.num_add_candidates + 1).tolist()
                samples['captions'] = [cap.capitalize() + '.' if cap[-1] != '.' else cap.capitalize() for cap in samples['captions']]
            if not self.multiple_choice:
                text_input = [self.prompt.format(question) for question in samples["text_input"]]
            else:
                text_input = [self.prompt.format(question, op_1, op_2, op_3, op_4) for question, [op_1, op_2, op_3, op_4] in zip(samples["text_input"], samples["choices"])]
            if self.use_caption:
                text_input = [cap + ' ' + txt for cap, txt in zip(samples['captions'], text_input)]
        else:
            text_input = samples["text_input"]
        input_tokens = self.t5_tokenizer(
            text_input, padding="longest", return_tensors="pt"
        ).to(image.device)
        if self.perform_selection and self.selection_criterion == 'Qconf':
            ques_input = [txt.split("Question: ")[0] + "Question: " for txt in text_input]
            ques_tokens = self.t5_tokenizer(
                ques_input, padding="longest", return_tensors="pt"
            ).to(image.device)
 

        if not self.use_caption: 
            atts_t5 = torch.repeat_interleave(atts_t5, self.num_add_candidates + 1, dim=0)
            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
            
            if self.perform_selection and self.selection_criterion == 'Qconf':
                ques_atts = torch.cat([atts_t5, ques_tokens.attention_mask], dim=1)
        else:
            encoder_atts = input_tokens.attention_mask
            if self.perform_selection and self.selection_criterion == 'Qconf': ques_atts = ques_tokens.attention_mask

        samples['image'].detach().cpu()

        if self.perform_ensembling:
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
                image_embeds = image_embeds.float()
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        with self.maybe_autocast(dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            if self.perform_selection and self.selection_criterion == 'Qconf':
                ques_embeds = self.t5_model.encoder.embed_tokens(ques_tokens.input_ids)
            
            if not self.use_caption:
                inputs_t5 = torch.repeat_interleave(inputs_t5, self.num_add_candidates + 1, dim=0)
                inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
                if self.perform_selection and self.selection_criterion == 'Qconf':
                    ques_embeds = torch.cat([inputs_t5, ques_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                length_penalty=length_penalty,
            ) 
            
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            output_scores = []
            
            
            if self.selection_criterion == 'Qconf':
                if self.dropout_aggregate:
                    output_scores = self.compute_score_with_dropout(ques_embeds, ques_atts, samples['text_input'])
                else: output_scores = self.compute_score(ques_embeds, ques_atts, samples['text_input'])
            elif self.selection_criterion == 'Aconf':
                if self.dropout_aggregate:
                    output_scores = self.compute_score_with_dropout(inputs_embeds, encoder_atts, output_text)    
                else: 
                    output_scores = self.compute_score(inputs_embeds, encoder_atts, output_text) 
                    if self.calibrate:
                        na_output_scores = self.compute_score(self.t5_model.encoder.embed_tokens(input_tokens.input_ids), input_tokens.attention_mask, output_text)
                        output_scores = output_scores - na_output_scores
            elif 'Plausible' in self.selection_criterion:
                if not self.multiple_choice:
                    p = self.num_add_candidates + 1
                    int_text = [output_text[i*p: i*p + p] for i in range(bs)]
                    refined_text = [[list(set(int))] * (self.num_add_candidates + 1) for int in int_text]
                    refined_text = sum(refined_text, [])
                    if not 'binary' in self.selection_criterion.lower():
                        plaus_text = ['Question: {} Plausible Answers: {}\nFinal Answer: '.format(samples['text_input'][i], str(refined_text[i]).strip('[').rstrip(']')) for i in range(len(output_text))]
                    else: 
                        plaus_text = ['Question: {} Plausible Answers: {}\nFinal Answer: {}.\nSelect the best option ("A" or "B"): Is the final answer correct?\nA: Yes, B: No.'.format(samples['text_input'][i], str(refined_text[i]).strip('[').rstrip(']'), output_text[i]) for i in range(len(output_text))]
                    # import pdb; pdb.set_trace()
                    plaus_tokens = self.t5_tokenizer(
                        plaus_text, padding="longest", return_tensors="pt", max_length=self.max_txt_len,
                    ).to(image.device)
                    plaus_embeds = self.t5_model.encoder.embed_tokens(plaus_tokens.input_ids)
                    plaus_embeds = torch.cat([inputs_t5, plaus_embeds], dim=1)
                    plaus_atts = torch.cat([atts_t5, plaus_tokens.attention_mask], dim=1)
                    if not 'binary' in self.selection_criterion.lower():
                        output_scores = self.compute_score(plaus_embeds, plaus_atts, output_text)
                    else:
                        output_scores = self.compute_binary_score(plaus_embeds, plaus_atts, labels=['A', 'B'])
                else: assert False
            elif self.selection_criterion == 'EnsembleQA':
                # check for self.max_txt_len
                p = self.num_add_candidates + 1
                agg_questions = [samples['text_input'][i*p: i*p + p] for i in range(bs)]
                agg_questions = [list(set(ques)) for ques in agg_questions]
                if not self.multiple_choice:
                    qensb_text = ['You are given a list of possible questions based on the image. Generate a short answer that answers most or all of the questions below.\nPossible Questions:\n{}\nShort Answer: '.format(self.combine_ques_text(ques)) for ques in agg_questions]
                else:
                    qensb_text = ['You are given a list of possible questions based on the image. Select an option that answers most or all of the questions below.\nPossible Questions:\n{}Options: A. {}, B. {}, C. {}, D. {}\nAnswer: Option '.format(self.combine_ques_text(ques), op1, op2, op3, op4) for ques, [op1, op2, op3, op4] in zip(agg_questions, samples['orig_choices'])]
                qensb_tokens = self.t5_tokenizer(
                        qensb_text, padding="longest", return_tensors="pt", max_length=self.max_txt_len
                    ).to(image.device)
                qensb_embeds = self.t5_model.encoder.embed_tokens(qensb_tokens.input_ids)
                qensb_embeds = torch.cat([orig_inputs_t5, qensb_embeds], dim=1)
                qensb_atts = torch.cat([orig_atts_t5, qensb_tokens.attention_mask], dim=1)
                with self.maybe_autocast(dtype=torch.bfloat16):
                    ensb_outputs = self.t5_model.generate(
                    inputs_embeds=qensb_embeds,
                    attention_mask=qensb_atts,
                    do_sample=False,
                    num_beams=num_beams,
                    max_new_tokens=max_len,
                    min_length=min_len,
                    length_penalty=length_penalty) 
                output_text = self.t5_tokenizer.batch_decode(
                    # outputs.sequences.detach().cpu(), skip_special_tokens=True
                    ensb_outputs, skip_special_tokens=True
                )
                del ensb_outputs

            elif self.selection_criterion == 'EnsembleDescQA':
                assert self.keyword_pipeline or self.reason
                # check for self.max_txt_len
                p = self.num_add_candidates + 1
                agg_questions = [samples['text_input'][i*p: i*p + p] for i in range(bs)]
                agg_questions = [list(set(ques)) for ques in agg_questions]
                descs = []
                if self.keyword_pipeline:
                    for j in range(bs):
                        if self.reason:
                            pipe_desc_lookup[j].update(reason_desc_lookup[j])
                        descs.append(pipe_desc_lookup[j])
                else:
                    for j in range(bs):
                        descs.append(reason_desc_lookup[j])
                if not self.multiple_choice:
                    qensb_text = ['You are given a list of possible questions based on the image. Here are some details about the image.\n{}\nGenerate a short answer that answers most or all of the questions below.\nPossible Questions:\n{}\nShort Answer: '.format(self.verbalize_desc(desc), self.combine_ques_text(ques)) for desc, ques in zip(descs, agg_questions)]
                else:
                    qensb_text = ['You are given a list of possible questions based on the image. Here are some details about the image.\n{}\nSelect an option that answers most or all of the questions below.\nPossible Questions:\n{}Options: A. {}, B. {}, C. {}, D. {}.\nAnswer: Option '.format(self.verbalize_desc(desc), self.combine_ques_text(ques), op1, op2, op3, op4) for desc, ques, [op1, op2, op3, op4] in zip(descs, agg_questions, samples['orig_choices'])]
                qensb_tokens = self.t5_tokenizer(
                        qensb_text, padding="longest", return_tensors="pt", max_length=self.max_txt_len
                    ).to(image.device)
                qensb_embeds = self.t5_model.encoder.embed_tokens(qensb_tokens.input_ids)
                qensb_embeds = torch.cat([orig_inputs_t5, qensb_embeds], dim=1)
                qensb_atts = torch.cat([orig_atts_t5 , qensb_tokens.attention_mask], dim=1)
                with self.maybe_autocast(dtype=torch.bfloat16):
                    ensb_outputs = self.t5_model.generate(
                    inputs_embeds=qensb_embeds,
                    attention_mask=qensb_atts,
                    do_sample=False,
                    num_beams=num_beams,
                    max_new_tokens=max_len,
                    min_length=min_len,
                    length_penalty=length_penalty,
                ) 
                output_text = self.t5_tokenizer.batch_decode(
                    # outputs.sequences.detach().cpu(), skip_special_tokens=True
                    ensb_outputs, skip_special_tokens=True
                )
                # import pdb; pdb.set_trace()
                del ensb_outputs


            elif self.selection_criterion == 'AQconf':
                aq_text_input = ['Answer: {} Proposed Question:'.format(ans) for ans in zip (output_text)]
                aq_tokens = self.t5_tokenizer(aq_text_input, padding='longest', return_tensors="pt").to(image.device)
                aq_atts = torch.cat([atts_t5, aq_tokens.attention_mask], dim=1)
                aq_embeds = self.t5_model.encoder.embed_tokens(aq_tokens.input_ids)
                aq_embeds = torch.cat([inputs_t5, aq_embeds], dim=1)
                output_scores = self.compute_score(aq_embeds, aq_atts, samples['text_input'])
                if self.calibrate: 
                    na_output_scores = self.compute_score(self.t5_model.encoder.embed_tokens(aq_tokens.input_ids), aq_tokens.attention_mask, samples['text_input'])

            elif self.selection_criterion == 'Binary':
                if self.multiple_choice: assert False, 'Not valid/supported'
                combined_text_input = [txt.replace('Short answer:', 'Proposed Answer:') + ' ' + ans + '.\nSelect the best option ("A" or "B"): Is the proposed answer correct?\nA: Yes, B: No.' for txt, ans in zip(text_input, output_text)]
                combined_tokens = self.t5_tokenizer(combined_text_input, padding="longest", return_tensors="pt").to(image.device)
                combined_atts = torch.cat([atts_t5, combined_tokens.attention_mask], dim=1)
                combined_embeds = self.t5_model.encoder.embed_tokens(combined_tokens.input_ids)
                combined_embeds = torch.cat([inputs_t5, combined_embeds], dim=1)
                output_scores = self.compute_binary_score(combined_embeds, combined_atts, labels=['A', 'B'])
                if self.calibrate:
                    na_output_scores = self.compute_binary_score(self.t5_model.encoder.embed_tokens(combined_tokens.input_ids), combined_tokens.attention_mask, labels=['A', 'B'])
                    output_scores = output_scores - na_output_scores
                
        if self._apply_lemmatizer:
            output_text = self._lemmatize(output_text)

        
        if not self.perform_ensembling:
            p = self.num_add_candidates + 1
            output_text = [output_text[i*p: i*p + p] for i in range(bs)]
            output_scores = np.array([output_scores[i*p: i*p + p] for i in range(bs)])

        
        if self.perform_selection and len(self.selection_criterion):
            samples['text_input'] = [samples['text_input'][i*p:i*p + p] for i in range(bs)]
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
            assert len(output_text) == bs
            output_scores = []
        else: output_scores = []
        if self.verbose and self.num_add_candidates > 0 and not self.perform_ensembling: return output_text, samples['text_input'], output_scores
        else: return output_text, [], []

    

    def combine_ques_text(self, questions):
        out_str = "\n * ".join(questions)
        return ' * ' + out_str.lstrip()
        

    def compute_score(self, inputs_embeds, encoder_atts, output_text, return_logits = False):
        decoder_input = self.t5_tokenizer(output_text, return_tensors='pt', padding=True)
        decoder_input_ids = decoder_input['input_ids']
        decoder_attention_mask = decoder_input['attention_mask']
        logits = self.t5_model.forward(inputs_embeds=inputs_embeds, attention_mask=encoder_atts, decoder_input_ids=decoder_input_ids.cuda(), decoder_attention_mask=decoder_attention_mask.cuda()).logits.detach().cpu()
        if return_logits: return logits
        all_logprobs = torch.log(torch.softmax(logits, dim=-1))
        labels = self.t5_tokenizer(output_text)['input_ids']
        labels = [l[:-1] for l in labels]
        filter_sums = []
        for row, label in zip(all_logprobs, labels):
            row = row[:len(label), :].float().numpy()
            vocab_size = row.shape[-1]
            if len(label):
                loc = F.one_hot(torch.tensor(label), num_classes=vocab_size).numpy().astype(bool)
                summed_logprob = np.sum(row, where = loc)
            else: summed_logprob = -100 # Degenerate generation
            filter_sums.append(summed_logprob / max(1, len(label)))
        return np.array(filter_sums)
    
    def compute_binary_score(self, inputs_embeds, encoder_atts, labels=['yes', 'no']):
        label_ids = self.t5_tokenizer(labels).input_ids
        label_ids = [l[0] for l in label_ids]
        bs = inputs_embeds.shape[0]
        logits = self.compute_score(inputs_embeds, encoder_atts, [labels[0]] * bs, return_logits=True).float().numpy()[:,0,:]
        yn_logits = logits[:, label_ids]
        log_probs = torch.log(torch.softmax(torch.tensor(yn_logits), dim = -1)).float().numpy()[:,0]
        return log_probs
    
    def compute_score_with_dropout(self, inputs_embeds, encoder_atts, output_text, K=20):
        probs = []
        with torch.no_grad():
            self.t5_model.train()
            for k in range(K):
                torch.manual_seed(k)
                probs.append(np.exp(self.compute_score(inputs_embeds, encoder_atts, output_text)))
        self.t5_model.eval()
        torch.manual_seed(self.orig_seed)
        logprobs = np.log(np.mean(probs, axis=0))
        return logprobs

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
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        print('Prompt: ', prompt)
        max_txt_len = cfg.get("max_txt_len", 32)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)
        multiple_choice = cfg.get('multiple_choice', False)
        keyword_pipeline = cfg.get('keyword_pipeline', True)
        reason = cfg.get('reason', False)
        paraphrase = cfg.get('paraphrase', False) and not keyword_pipeline
        prefix_answer = cfg.get('prefix_answer', True)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            multiple_choice=multiple_choice,
            orig_seed = cfg.get('orig_seed', 42),
            par_num_beams = cfg.get('par_num_beams', 5),
            num_add_candidates = cfg.get('num_add_candidates', 5),
            paraphrase = paraphrase,
            ext_paraphrase = cfg.get('ext_paraphrase', True),
            perform_selection = cfg.get('perform_selection', False),
            calibrate = cfg.get('calibrate', False),
            dropout_aggregate = cfg.get('dropout_aggregate', False),
            selection_criterion = cfg.get('selection_criterion', ''),
            perform_ensembling = cfg.get('perform_ensembling', False),
            verbose = cfg.get('verbose', True),
            constrained = cfg.get('constrained', True),
            use_caption = cfg.get('use_caption', False),
            use_promptcap = cfg.get('use_promptcap', False),
            alt_device = cfg.get('alt_device', -1),
            keyword_pipeline = keyword_pipeline,
            reason = reason,
            prefix_answer = prefix_answer,
        )
        model.load_checkpoint_from_config(cfg)

        return model
