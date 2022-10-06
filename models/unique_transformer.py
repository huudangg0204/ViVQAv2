import torch
from torch import nn
from torch.nn import functional as F

from .base_unique_transformer import BaseUniqueTransformer
from utils.instances import Instances
from builders.encoder_builder import build_encoder
from builders.text_embedding_builder import build_text_embedding
from builders.vision_embedding_builder import build_vision_embedding
from builders.model_builder import META_ARCHITECTURE

@META_ARCHITECTURE.register()
class UniqueTransformer(BaseUniqueTransformer):
    def __init__(self, config, vocab):
        super().__init__(config, vocab)

        self.device = torch.device(config.DEVICE)
        self.vocab = vocab

        self.region_embedding = build_vision_embedding(config.REGION_EMBEDDING)
        self.grid_embedding = build_vision_embedding(config.GRID_EMBEDDING)
        self.box_embedding = build_vision_embedding(config.BOX_EMBEDDING)
        self.text_embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)

        self.encoder = build_encoder(config.ENCODER)
        self.fc = nn.Linear(config.D_MODEL, len(vocab), bias=False)

    def embed_features(self, input_features: Instances):
        region_features = input_features.region_features
        region_features, region_padding_mask = self.region_embedding(region_features)
        region_feat_tokens = torch.ones((region_features.shape[0], region_features.shape[1])).long().to(region_features.device) * self.vocab.feat_idx
        region_feat_embedded, _ = self.text_embedding(region_feat_tokens)
        region_features += region_feat_embedded

        region_boxes = input_features.region_boxes
        region_boxes, region_boxes_padding_mask = self.box_embedding(region_boxes)
        region_box_tokens = torch.ones((region_boxes.shape[0], region_boxes.shape[1])).long().to(region_boxes.device) * self.vocab.box_idx
        region_box_embedded, _ = self.text_embedding(region_box_tokens)
        region_boxes += region_box_embedded

        grid_features = input_features.grid_features
        grid_features, grid_padding_mask = self.grid_embedding(grid_features)
        grid_feat_tokens = torch.ones((grid_features.shape[0], grid_features.shape[1])).long().to(grid_features.device) * self.vocab.feat_idx
        grid_feat_embedded, _ = self.text_embedding(grid_feat_tokens)
        grid_features += grid_feat_embedded
        
        grid_boxes = input_features.grid_boxes
        grid_boxes, grid_boxes_padding_mask = self.box_embedding(grid_boxes)
        grid_box_tokens = torch.ones((grid_boxes.shape[0], grid_boxes.shape[1])).long().to(grid_boxes.device) * self.vocab.box_idx
        grid_box_embedded, _ = self.text_embedding(grid_box_tokens)
        grid_boxes += grid_box_embedded

        vision_features = torch.cat([region_features, region_boxes, grid_features, grid_boxes], dim=1)
        vision_padding_mask = torch.cat([region_padding_mask, region_boxes_padding_mask, grid_padding_mask, grid_boxes_padding_mask], dim=-1)

        question_tokens = input_features.question_tokens
        q_tokens = torch.ones((question_tokens.shape[0], question_tokens.shape[1])).long().to(question_tokens.device) * self.vocab.question_idx
        question_features, (question_padding_mask, _) = self.text_embedding(question_tokens)
        q_embeded, _ = self.text_embedding(q_tokens)
        question_features += q_embeded

        joint_features = torch.cat([vision_features, question_features], dim=1)
        joint_attention_mask = torch.cat([vision_padding_mask, question_padding_mask], dim=-1)

        return joint_features, joint_attention_mask

    def append_answer(self, joint_features, joint_attention_mask, answer_tokens):
        answer_features, (answer_padding_mask, answer_sequential_mask) = self.text_embedding(answer_tokens)
        answer_self_attention_mask = torch.logical_or(answer_padding_mask, answer_sequential_mask) # (bs, 1, answer_len, answer_len)
        a_tokens = torch.ones((answer_tokens.shape[0], answer_tokens.shape[1])).long().to(answer_tokens.device) * self.vocab.answer_idx
        a_embedded, _ = self.text_embedding(a_tokens)
        answer_features += a_embedded
        
        joint_features = torch.cat([joint_features, answer_features], dim=1)
        answer_len = answer_features.shape[1]
        # joint features cannot see the answer features
        batch_size = joint_features.shape[0]
        joint_features_len = joint_features.shape[1]
        joint_self_attention_mask = joint_attention_mask.expand((-1, -1, joint_features_len, -1)) # (bs, 1, joint_features_len, joint_features_len)
        joint_features_mask_answer = torch.ones((batch_size, joint_features_len, answer_len)).bool().to(joint_features.device) # (bs, 1, joint_features_len, answer_len)
        joint_features_mask_answer = torch.cat([joint_self_attention_mask, joint_features_mask_answer], dim=-1) # (bs, 1, joint_features_len, joint_features_len + answer_len)
        # answer tokens can attend to all joint features
        answer_attend_joint_features = torch.zeros((batch_size, answer_len, joint_features_len)).bool().to(answer_features.device) # (bs, 1, answer_len, joint_features_len)
        answer_attend_joint_features = torch.cat([answer_attend_joint_features, answer_self_attention_mask], dim=-1) # (bs, 1, answer_len, joint_features_len + answer_len)
        
        joint_attention_mask = torch.cat([joint_features_mask_answer, answer_attend_joint_features], dim=-2) # (bs, 1 , joint_features_len + answer_len, joint_features_len + answer_len)

        return joint_features, joint_attention_mask

    def forward(self, input_features: Instances):
        joint_features, joint_attention_mask = self.embed_features(input_features)
        joint_features_len = joint_features.shape[1]
        answer_tokens = input_features.answer_tokens
        joint_features, joint_attention_mask = self.append_answer(joint_features, joint_attention_mask, answer_tokens)

        out = self.encoder(Instances(
            features=joint_features,
            features_padding_mask=joint_attention_mask
        ))

        return F.log_softmax(out[:, joint_features_len:], dim=-1)
