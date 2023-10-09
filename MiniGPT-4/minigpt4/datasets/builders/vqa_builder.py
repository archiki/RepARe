"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from minigpt4.common.registry import registry
from minigpt4.datasets.datasets.aok_vqa_datasets import AOKVQADataset, AOKVQAEvalDataset
from minigpt4.datasets.datasets.coco_vqa_datasets import COCOVQADataset, COCOVQAEvalDataset


@registry.register_builder("coco_vqa")
class COCOVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOVQADataset
    eval_dataset_cls = COCOVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_vqa.yaml",
        "eval": "configs/datasets/coco/eval_vqa.yaml",
    }


@registry.register_builder("ok_vqa")
class OKVQABuilder(COCOVQABuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/okvqa/defaults.yaml",
    }


@registry.register_builder("aok_vqa")
class AOKVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = AOKVQADataset
    eval_dataset_cls = AOKVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/aokvqa/defaults.yaml",
        "eval": "configs/datasets/aokvqa/eval_aokvqa.yaml",
    }
   


# @registry.register_builder("gqa")
# class GQABuilder(BaseDatasetBuilder):
#     train_dataset_cls = GQADataset
#     eval_dataset_cls = GQAEvalDataset

#     DATASET_CONFIG_DICT = {
#         "default": "configs/datasets/gqa/defaults.yaml",
#         "balanced_val": "configs/datasets/gqa/balanced_val.yaml",
#         "balanced_testdev": "configs/datasets/gqa/balanced_testdev.yaml",
#     }