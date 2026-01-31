"""
Phrasal-aware Layers for Vietnamese VQA.

Implements:
- PhrasalMultiHeadAttention: Multi-head wrapper with phrasal score propagation
- PhrasalEncoderLayer: Encoder layer with Co-Text gated fusion
"""

import torch
from torch import nn

from models.modules.containers import Module
from models.modules.positionwise_feed_forward import PositionWiseFeedForward
from builders.attention_builder import build_attention


class PhrasalMultiHeadAttention(Module):
    """
    Multi-head attention with phrasal score propagation.
    
    Wraps PhrasalScaledDotProductAttention and handles:
    - Dropout and Layer Normalization
    - Phrasal score pass-through for Co-Text module
    - Optional Attention-on-Attention (AoA) mechanism
    """

    def __init__(self, config):
        super(PhrasalMultiHeadAttention, self).__init__()
        
        d_model = config.D_MODEL

        self.use_aoa = getattr(config, 'USE_AOA', False)
        
        if self.use_aoa:
            self.informative_attention = nn.Linear(2 * d_model, d_model)
            self.gated_attention = nn.Linear(2 * d_model, d_model)

        self.attention = build_attention(config)

        self.dropout = nn.Dropout(p=config.DROPOUT)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = getattr(config, 'CAN_BE_STATEFUL', False)
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask, **kwargs):
        """
        Forward pass with phrasal score propagation.
        
        Returns:
            out: Attention output
            phrasal_scores: Phrasal score matrix for Co-Text propagation
        """
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        # PhrasalScaledDotProductAttention returns (out, att_weights, phrasal_scores)
        attention_output = self.attention(queries, keys, values, attention_mask, **kwargs)
        
        # Handle both standard attention (2 returns) and phrasal attention (3 returns)
        if len(attention_output) == 3:
            out, _, phrasal_scores = attention_output
        else:
            out, _ = attention_output
            phrasal_scores = None

        # Residual connection and layer normalization
        out = self.dropout(out)
        out = self.layer_norm(queries + out)

        if self.use_aoa:
            aoa_input = torch.cat([queries, out], dim=-1)
            i = self.informative_attention(aoa_input)
            g = torch.sigmoid(self.gated_attention(aoa_input))
            out = i * g
            
        return out, phrasal_scores


class PhrasalEncoderLayer(nn.Module):
    """
    Encoder layer with Co-Text gated fusion for phrasal feature propagation.
    
    This layer preserves phrasal structure information across deep transformer
    layers by using a gated residual connection that fuses:
    - Current layer attention output
    - Phrasal features from previous layers
    
    The gating mechanism learns to balance between standard attention and
    phrasal-aware features, preventing information loss in deep networks.
    """

    def __init__(self, config):
        super(PhrasalEncoderLayer, self).__init__()
        
        d_model = config.D_MODEL
        
        self.mhatt = PhrasalMultiHeadAttention(config)
        self.pwff = PositionWiseFeedForward(config)
        
        # Co-Text Gated Fusion: combines current output with phrasal features
        self.use_cotext_gate = getattr(config, 'USE_COTEXT_GATE', True)
        if self.use_cotext_gate:
            self.gate_proj = nn.Linear(d_model * 2, d_model)
            self.gate_sigmoid = nn.Linear(d_model * 2, d_model)
            self.cotext_layer_norm = nn.LayerNorm(d_model)

    def forward(self, queries, keys, values, attention_mask, phrasal_features=None, **kwargs):
        """
        Forward pass with Co-Text residual connection.
        
        Args:
            queries: Query tensor (batch_size, seq_len, d_model)
            keys: Key tensor
            values: Value tensor
            attention_mask: Attention mask
            phrasal_features: Features from previous layer for Co-Text fusion
        
        Returns:
            out: Layer output
            phrasal_scores: Phrasal scores for next layer's Co-Text fusion
        """
        # Self-attention with phrasal scoring
        att_out, phrasal_scores = self.mhatt(
            queries=queries, 
            keys=keys, 
            values=values, 
            attention_mask=attention_mask, 
            **kwargs
        )
        
        # Co-Text Gated Fusion: inject phrasal features from previous layer
        if self.use_cotext_gate and phrasal_features is not None:
            # Concatenate current output with previous phrasal features
            combined = torch.cat([att_out, phrasal_features], dim=-1)
            
            # Compute gating weights
            gate = torch.sigmoid(self.gate_sigmoid(combined))
            
            # Gated fusion: interpolate between current and combined features
            fused = self.gate_proj(combined)
            att_out = gate * fused + (1 - gate) * att_out
            att_out = self.cotext_layer_norm(att_out)
        
        # Feed-forward network
        ff_out = self.pwff(att_out)
        
        return ff_out, phrasal_scores


class PhrasalGuidedEncoderLayer(nn.Module):
    """
    Guided encoder layer with phrasal-aware attention.
    
    Used for cross-modal attention (vision guided by language or vice versa),
    with phrasal score computation on the guiding features.
    """

    def __init__(self, config):
        super(PhrasalGuidedEncoderLayer, self).__init__()
        
        d_model = config.D_MODEL
        
        # Self-attention on queries
        self.self_mhatt = PhrasalMultiHeadAttention(config)
        # Guided attention from keys/values
        self.guided_mhatt = PhrasalMultiHeadAttention(config)
        # Feed-forward
        self.pwff = PositionWiseFeedForward(config)
        
        # Co-Text gating for self-attention path
        self.use_cotext_gate = getattr(config, 'USE_COTEXT_GATE', True)
        if self.use_cotext_gate:
            self.gate_proj = nn.Linear(d_model * 2, d_model)
            self.gate_sigmoid = nn.Linear(d_model * 2, d_model)
            self.cotext_layer_norm = nn.LayerNorm(d_model)

    def forward(self, queries, keys, values, self_attention_mask, guided_attention_mask, 
                phrasal_features=None, **kwargs):
        """
        Forward pass with guided attention and Co-Text fusion.
        
        Returns:
            out: Layer output
            phrasal_scores: Phrasal scores from self-attention
        """
        # Self-attention with phrasal scoring
        self_att, phrasal_scores = self.self_mhatt(
            queries=queries,
            keys=queries, 
            values=queries,
            attention_mask=self_attention_mask,
            **kwargs
        )
        
        # Co-Text fusion
        if self.use_cotext_gate and phrasal_features is not None:
            combined = torch.cat([self_att, phrasal_features], dim=-1)
            gate = torch.sigmoid(self.gate_sigmoid(combined))
            fused = self.gate_proj(combined)
            self_att = gate * fused + (1 - gate) * self_att
            self_att = self.cotext_layer_norm(self_att)
        
        # Guided attention (cross-modal)
        guided_att, _ = self.guided_mhatt(
            queries=self_att, 
            keys=keys, 
            values=values,
            attention_mask=guided_attention_mask,
            **kwargs
        )

        # Feed-forward
        ff_out = self.pwff(guided_att)

        return ff_out, phrasal_scores
