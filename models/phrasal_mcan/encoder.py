"""
Phrasal-aware Encoders for Vietnamese VQA.

Implements:
- PhrasalEncoder: Self-attention encoder with phrasal score propagation
- PhrasalGuidedAttentionEncoder: Cross-modal encoder with phrasal-aware attention
"""

import torch
from torch import nn

from models.modules.pos_embeddings import SinusoidPositionalEmbedding
from models.phrasal_mcan.layers import PhrasalEncoderLayer, PhrasalGuidedEncoderLayer
from builders.encoder_builder import META_ENCODER


@META_ENCODER.register()
class PhrasalEncoder(nn.Module):
    """
    Self-attention encoder with phrasal score propagation across layers.
    
    Implements Co-Text Module from ViWordFormer:
    - Each layer receives phrasal features from the previous layer
    - Gated fusion prevents phrasal information loss in deep networks
    - Phrasal scores flow through the encoder via shortcut connections
    """

    def __init__(self, config):
        super(PhrasalEncoder, self).__init__()
        
        self.pos_embedding = SinusoidPositionalEmbedding(config.D_MODEL)
        self.layer_norm = nn.LayerNorm(config.D_MODEL)

        self.d_model = config.D_MODEL
        self.layers = nn.ModuleList([
            PhrasalEncoderLayer(config.SELF_ATTENTION) 
            for _ in range(config.LAYERS)
        ])

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor):
        """
        Forward pass with phrasal feature propagation.
        
        Args:
            features: Input tensor (batch_size, seq_len, d_model)
            padding_mask: Padding mask for attention
        
        Returns:
            out: Encoded features with phrasal-aware attention
        """
        # Initial embedding with positional encoding
        out = self.layer_norm(features) + self.pos_embedding(features)
        
        # Phrasal features for Co-Text propagation (shortcut connection)
        phrasal_features = None
        
        for layer in self.layers:
            # Each layer receives phrasal features from previous layer
            out, phrasal_scores = layer(
                queries=out, 
                keys=out, 
                values=out, 
                attention_mask=padding_mask,
                phrasal_features=phrasal_features
            )
            
            # Update phrasal features for next layer (Co-Text shortcut)
            # Use current layer output as phrasal context for next layer
            phrasal_features = out

        return out


@META_ENCODER.register()
class PhrasalGuidedAttentionEncoder(nn.Module):
    """
    Guided attention encoder with phrasal-aware cross-modal attention.
    
    Based on Deep Modular Co-Attention Network (MCAN) architecture,
    enhanced with phrasal score propagation for Vietnamese text.
    """

    def __init__(self, config):
        super(PhrasalGuidedAttentionEncoder, self).__init__()

        self.pos_embedding = SinusoidPositionalEmbedding(config.D_MODEL)
        self.layer_norm = nn.LayerNorm(config.D_MODEL)

        self.d_model = config.D_MODEL

        self.guided_attn_layers = nn.ModuleList([
            PhrasalGuidedEncoderLayer(config.GUIDED_ATTENTION) 
            for _ in range(config.LAYERS)
        ])

    def forward(self, vision_features: torch.Tensor, vision_padding_mask: torch.Tensor, 
                language_features: torch.Tensor, language_padding_mask: torch.Tensor):
        """
        Forward pass with guided attention and phrasal propagation.
        
        Args:
            vision_features: Visual features (batch_size, n_regions, d_model)
            vision_padding_mask: Mask for visual features
            language_features: Text features (batch_size, seq_len, d_model)
            language_padding_mask: Mask for text features
        
        Returns:
            out: Visual features enhanced by language guidance
        """
        # Initial embedding with positional encoding
        out = self.layer_norm(vision_features) + self.pos_embedding(vision_features)
        
        # Phrasal features for Co-Text propagation
        phrasal_features = None
        
        for guided_attn_layer in self.guided_attn_layers:
            out, phrasal_scores = guided_attn_layer(
                queries=out,
                keys=language_features,
                values=language_features,
                self_attention_mask=vision_padding_mask,
                guided_attention_mask=language_padding_mask,
                phrasal_features=phrasal_features
            )
            
            # Update phrasal features for next layer
            phrasal_features = out

        return out
