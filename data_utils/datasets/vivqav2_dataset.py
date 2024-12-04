import torch

from data_utils.utils import preprocess_sentence
from data_utils.datasets.feature_dataset import FeatureDataset
from data_utils.datasets.dictionary_dataset import DictionaryDataset
from utils.instance import Instance
from builders.dataset_builder import META_DATASET
from data_utils.datasets.base_dataset import BaseDataset

import numpy as np
from typing import Dict, List

@META_DATASET.register()
class Vivqav2FeatureDataset(FeatureDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__(json_path, vocab, config)

    @property
    def questions(self):
        return [ann["question"] for ann in self.annotations]

    @property
    def answers(self):
        return [ann["answer"] for ann in self.annotations]

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    for answer in ann["answers"]:
                        question = ann["question"]
                        answer = preprocess_sentence(answer, self.vocab.tokenizer)
                        annotation = {
                            "question_id": ann["id"],
                            "question": question,
                            "answer": answer,
                            "image_id": ann["image_id"]
                        }
                        annotations.append(annotation)
                    break

        return annotations

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        question = item["question"]
        answer = self.vocab.encode_answer(item["answer"])

        shifted_right_answer = torch.zeros_like(answer).fill_(self.vocab.padding_idx)
        shifted_right_answer[:-1] = answer[1:]
        answer = torch.where(answer == self.vocab.eos_idx, self.vocab.padding_idx, answer) # remove eos_token in answer
        
        features = self.load_features(self.annotations[idx]["image_id"])

        return Instance(
            question=question,
            answer_tokens=answer,
            shifted_right_answer_tokens=shifted_right_answer,
            **features,
        )

    def __len__(self) -> int:
        return len(self.annotations)



@META_DATASET.register()
class Vivqav2DictinaryDataset(BaseDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__(json_path, vocab, config)

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    answers = [preprocess_sentence(answer, self.vocab.tokenizer) for answer in ann["answer"]]
                    answers = [" ".join(answer) for answer in answers]
                    annotation = {
                        "question_id": ann["id"],
                        "question": ann["question"],
                        "answer": answers,
                        "image_id": ann["image_id"],
                    }
                    break

            annotations.append(annotation)

        return annotations

    def __len__(self) -> int:
        return super().__len__()
    
    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        image_id = item["image_id"]
        features = self.load_features(image_id)
        question = item["question"]
        answers = item["answer"]

        return Instance(
            question_id=item["question_id"],
            image_id=image_id,
            features=features,
            question=question,
            answers=answers,
            **features
        )

