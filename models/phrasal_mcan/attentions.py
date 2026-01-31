"""
Phrasal-aware Attention Mechanisms for Vietnamese VQA.

Implements Phrasal Lexeme module from ViWordFormer paper:
- PhrasalScaledDotProductAttention: Computes phrasal scores via Bilinear layer
  and integrates them into standard scaled dot-product attention.
"""

import torch
from torch import nn
import numpy as np
from builders.attention_builder import META_ATTENTION


@META_ATTENTION.register()
class PhrasalScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention enhanced with Phrasal Score computation.
    
    The Phrasal Score P(i,j) captures the likelihood that tokens i and j 
    belong to the same phrase (cụm từ) in Vietnamese syllable-level input.
    
    Attention formula:
        A = (Q @ K^T) / sqrt(d_k)
        P = sigmoid(Bilinear(x_i, x_j))  for all token pairs (i,j)
        A_final = softmax(A + lambda * log(P + eps))
    
    GPU-optimized: Uses tensor broadcasting instead of for-loops.
    """

    def __init__(self, config):
        super(PhrasalScaledDotProductAttention, self).__init__()

        d_model = config.D_MODEL
        h = config.HEAD
        d_k = config.D_KEY
        d_v = config.D_VALUE

        # Standard Q, K, V projections
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        # Phrasal Score computation: Bilinear(hidden_dim, hidden_dim, 1)
        self.phrasal_bilinear = nn.Bilinear(d_model, d_model, 1, bias=True)
        
        # Learnable lambda parameter to control phrasal influence
        lambda_init = getattr(config, 'LAMBDA_INIT', 1.0)
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)
        # Initialize bilinear layer
        nn.init.xavier_uniform_(self.phrasal_bilinear.weight)
        nn.init.constant_(self.phrasal_bilinear.bias, 0)

    def compute_phrasal_scores(self, x):
        """
        Compute Phrasal Score matrix P for all token pairs (i, j).
        
        Memory-efficient implementation using matrix multiplication:
        P(i,j) = x_i^T * W * x_j + b
        Implemented as P = (x @ W) @ x.T + b
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            P: Phrasal score matrix of shape (batch_size, seq_len, seq_len)
               Values are in range (0, 1) after sigmoid activation.
        """
        b_s, n, d = x.shape
        
        # self.phrasal_bilinear.weight has shape (1, d, d)
        # self.phrasal_bilinear.bias has shape (1,)
        W = self.phrasal_bilinear.weight[0]  # (d, d)
        
        # Step 1: x_W = x @ W -> (b_s, n, d)
        x_W = torch.matmul(x, W)
        
        # Step 2: P = x_W @ x^T -> (b_s, n, n)
        # Using transpose on the last two dimensions
        P = torch.matmul(x_W, x.transpose(-1, -2))
        
        # Step 3: Add bias and apply sigmoid
        P = P + self.phrasal_bilinear.bias
        P = torch.sigmoid(P)
        
        return P

    def forward(self, queries, keys, values, attention_mask=None, **kwargs):
        """
        Forward pass with phrasal-enhanced attention.
        
        Args:
            queries: (batch_size, n_queries, d_model)
            keys: (batch_size, n_keys, d_model)
            values: (batch_size, n_keys, d_model)
            attention_mask: Optional mask for attention
        
        Returns:
            out: Attention output (batch_size, n_queries, d_model)
            att: Attention weights (batch_size, heads, n_queries, n_keys)
            phrasal_scores: Phrasal score matrix (batch_size, n_queries, n_queries)
        """
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        # Step 1: Compute standard Q, K, V projections
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)      # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)    # (b_s, h, nk, d_v)

        # Step 2: Compute standard attention scores A = (Q @ K^T) / sqrt(d_k)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # Step 3: Compute Phrasal Scores P for query tokens
        # Only compute for self-attention case (queries == keys in terms of representation)
        phrasal_scores = self.compute_phrasal_scores(queries)  # (b_s, nq, nq)
        
        # Step 4: Integrate phrasal scores into attention
        # For self-attention: nq == nk, directly add phrasal influence
        # For cross-attention: nq != nk, we still use phrasal scores from queries
        if nq == nk:
            # Self-attention case: A_final = A + lambda * log(P + eps)
            # Expand phrasal scores for multi-head: (b_s, nq, nk) -> (b_s, 1, nq, nk)
            phrasal_log = self.lambda_param * torch.log(phrasal_scores.unsqueeze(1) + 1e-9)
            att = att + phrasal_log
        else:
            # Cross-attention case: phrasal scores capture query structure
            # Apply phrasal bias only for query side coherence (optional enhancement)
            pass

        # Step 5: Apply attention mask if provided
        if attention_mask is not None:
            att = att + attention_mask

        # Step 6: Softmax normalization
        att = torch.softmax(att, dim=-1)

        # Step 7: Compute output
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)

        return out, att, phrasal_scores
