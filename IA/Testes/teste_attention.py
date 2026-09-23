import sys
import torch
from pathlib import Path

# Garante que o Python encontre a pasta 'Codigo' estando dentro da pasta 'Testes'
caminho_raiz = Path(__file__).parent.parent
sys.path.append(str(caminho_raiz))

from Codigo.attention import (
    simple_self_attention,
    ScaledDotProductAttention,
    CausalAttention,
    MultiHeadAttention
)

# ==========================================
# 1. PARÂMETROS E TENSORES DE TESTE
# ==========================================
torch.manual_seed(42) # Semente fixa para previsibilidade

batch_size = 2
num_tokens = 4
d_in = 6
d_out = 8
num_heads = 2
context_length = num_tokens

# A primeira função espera um tensor 2D [Tokens, Dimensão]
inputs_2d = torch.randn(num_tokens, d_in)

# As outras classes lidam nativamente com lotes (Batch), então usamos um tensor 3D
inputs_3d = torch.randn(batch_size, num_tokens, d_in)

print("Iniciando testes de sanidade (Sanity Checks) do attention.py...\n")

# ==========================================
# 2. EXECUÇÃO DOS TESTES
# ==========================================

# TESTE 1: Self-Attention Simplificado
print("--- Teste 1: Self-Attention Simplificado ---")
ctx_simple, w_simple = simple_self_attention(inputs_2d)
print(f"Entrada 2D: {inputs_2d.shape}")
print(f"Saída Contexto: {ctx_simple.shape} -> (Esperado: [4, 6])")
print(f"Saída Pesos: {w_simple.shape} -> (Esperado: [4, 4])\n")

# TESTE 2: Scaled Dot-Product Attention
print("--- Teste 2: Scaled Dot-Product Attention ---")
scaled_att = ScaledDotProductAttention(d_in=d_in, d_out=d_out)
ctx_scaled, w_scaled = scaled_att(inputs_3d) 
print(f"Entrada 3D: {inputs_3d.shape}")
print(f"Saída Contexto: {ctx_scaled.shape} -> (Esperado: [2, 4, 8])")
print(f"Saída Pesos: {w_scaled.shape} -> (Esperado: [2, 4, 4])\n")

# TESTE 3: Causal Attention
print("--- Teste 3: Causal Attention ---")
causal_att = CausalAttention(d_in=d_in, d_out=d_out, context_length=context_length, dropout=0.0)
ctx_causal = causal_att(inputs_3d)
print(f"Entrada 3D: {inputs_3d.shape}")
print(f"Saída Contexto: {ctx_causal.shape} -> (Esperado: [2, 4, 8])\n")

# TESTE 4: Multi-Head Attention
print("--- Teste 4: Multi-Head Attention ---")
mha = MultiHeadAttention(d_in=d_in, d_out=d_out, context_length=context_length, dropout=0.0, num_heads=num_heads)
ctx_mha = mha(inputs_3d)
print(f"Entrada 3D: {inputs_3d.shape}")
print(f"Saída Contexto: {ctx_mha.shape} -> (Esperado: [2, 4, 8])\n")

print("✅ Todos os 4 mecanismos processaram os tensores básicos sem quebrar!")