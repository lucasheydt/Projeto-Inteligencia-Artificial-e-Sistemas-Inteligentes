import torch
import torch.nn as nn


# =====================================================================
# 1. SELF-ATTENTION SIMPLIFICADO
# =====================================================================


def simple_self_attention(inputs):
    """
    Self-Attention simplificado.

    Entrada:
        inputs: tensor [num_tokens, d_in]

    Saídas:
        context_vecs: tensor [num_tokens, d_in]
        attn_weights: matriz de atenção [num_tokens, num_tokens]
    """

    # 1. Calcula os attention scores
    attn_scores = inputs @ inputs.T

    # 2. Normaliza os scores utilizando Softmax
    attn_weights = torch.softmax(attn_scores, dim=-1)

    # 3. Calcula os vetores de contexto
    context_vecs = attn_weights @ inputs

    return context_vecs, attn_weights


# =====================================================================
# 2. SCALED DOT-PRODUCT ATTENTION
# =====================================================================


class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):

        super().__init__()

        self.W_query = nn.Linear(
            d_in,
            d_out,
            bias=qkv_bias
        )

        self.W_key = nn.Linear(
            d_in,
            d_out,
            bias=qkv_bias
        )

        self.W_value = nn.Linear(
            d_in,
            d_out,
            bias=qkv_bias
        )

    def forward(self, x):

        # Projeções Q, K e V
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Attention scores
        attn_scores = queries @ keys.transpose(-2, -1)

        # Dimensão das keys
        d_k = keys.shape[-1]

        # Scaled Dot-Product + Softmax
        attn_weights = torch.softmax(
            attn_scores / (d_k ** 0.5),
            dim=-1
        )

        # Vetores de contexto
        context_vecs = attn_weights @ values

        return context_vecs, attn_weights


# =====================================================================
# 3. CAUSAL ATTENTION
# =====================================================================


class CausalAttention(nn.Module):

    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        dropout,
        qkv_bias=False
    ):

        super().__init__()

        self.d_out = d_out

        # Projeções Q, K e V
        self.W_query = nn.Linear(
            d_in,
            d_out,
            bias=qkv_bias
        )

        self.W_key = nn.Linear(
            d_in,
            d_out,
            bias=qkv_bias
        )

        self.W_value = nn.Linear(
            d_in,
            d_out,
            bias=qkv_bias
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Máscara triangular superior.
        # Os valores acima da diagonal representam tokens futuros.
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(
                    context_length,
                    context_length
                ),
                diagonal=1
            )
        )

    def forward(self, x):

        # x = [batch, num_tokens, d_in]
        b, num_tokens, d_in = x.shape

        # Projeções
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Attention scores
        attn_scores = queries @ keys.transpose(1, 2)

        # Seleciona apenas a parte necessária da máscara
        mask_bool = self.mask.bool()[
            :num_tokens,
            :num_tokens
        ]

        # Impede acesso aos tokens futuros
        attn_scores.masked_fill_(
            mask_bool,
            -torch.inf
        )

        # Scaled Dot-Product + Softmax
        attn_weights = torch.softmax(
            attn_scores / (keys.shape[-1] ** 0.5),
            dim=-1
        )

        # Dropout
        attn_weights = self.dropout(attn_weights)

        # Vetores de contexto
        context_vecs = attn_weights @ values

        return context_vecs


# =====================================================================
# 4. MULTI-HEAD ATTENTION
# =====================================================================


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        dropout,
        num_heads,
        qkv_bias=False
    ):

        super().__init__()

        # d_out precisa ser divisível pelo número de heads
        assert d_out % num_heads == 0, \
            "d_out deve ser divisível por num_heads"

        self.d_out = d_out
        self.num_heads = num_heads

        # Dimensão de cada head
        self.head_dim = d_out // num_heads

        # Projeções Q, K e V
        self.W_query = nn.Linear(
            d_in,
            d_out,
            bias=qkv_bias
        )

        self.W_key = nn.Linear(
            d_in,
            d_out,
            bias=qkv_bias
        )

        self.W_value = nn.Linear(
            d_in,
            d_out,
            bias=qkv_bias
        )

        # Projeção final
        self.out_proj = nn.Linear(
            d_out,
            d_out
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Máscara causal
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(
                    context_length,
                    context_length
                ),
                diagonal=1
            )
        )

    def forward(self, x):

        # x:
        # [batch, num_tokens, d_in]

        b, num_tokens, d_in = x.shape

        # =========================================================
        # 1. Calcula Q, K e V
        # =========================================================

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # =========================================================
        # 2. Divide em múltiplas heads
        # =========================================================

        queries = queries.view(
            b,
            num_tokens,
            self.num_heads,
            self.head_dim
        )

        keys = keys.view(
            b,
            num_tokens,
            self.num_heads,
            self.head_dim
        )

        values = values.view(
            b,
            num_tokens,
            self.num_heads,
            self.head_dim
        )

        # =========================================================
        # 3. Coloca num_heads antes de num_tokens
        # =========================================================

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # =========================================================
        # 4. Calcula Attention Scores
        # =========================================================

        attn_scores = queries @ keys.transpose(2, 3)

        # =========================================================
        # 5. Máscara causal
        # =========================================================

        mask_bool = self.mask.bool()[
            :num_tokens,
            :num_tokens
        ]

        attn_scores.masked_fill_(
            mask_bool,
            -torch.inf
        )

        # =========================================================
        # 6. Scaled Dot-Product + Softmax
        # =========================================================

        attn_weights = torch.softmax(
            attn_scores / (keys.shape[-1] ** 0.5),
            dim=-1
        )

        # =========================================================
        # 7. Dropout
        # =========================================================

        attn_weights = self.dropout(attn_weights)

        # =========================================================
        # 8. Calcula contexto
        # =========================================================

        context_vecs = attn_weights @ values


        context_vecs = context_vecs.transpose(1, 2)

        # =========================================================
        # 9. Junta todas as heads
        # =========================================================

        context_vecs = context_vecs.contiguous().view(
            b,
            num_tokens,
            self.d_out
        )

        # =========================================================
        # 10. Projeção final
        # =========================================================

        context_vecs = self.out_proj(context_vecs)

        return context_vecs