import torch

# Criando um tensor 1D com 6 elementos: [0, 1, 2, 3, 4, 5]
original = torch.arange(6)

# Mudar o formato para 2 linhas e 3 colunas usando .view()
reformatado = original.view(2, 3)

# Transpor (trocar linhas por colunas) usando .T
transposto = reformatado.T

print("Original (1D):", original)
print("Reformatado (2x3):\n", reformatado)
print("Transposto (3x2):\n", transposto)