import torch

# Criar a partir de uma lista comum do Python
tensor_lista = torch.tensor([1, 2, 3])

# Criar um tensor preenchido com zeros (2 linhas x 3 colunas)
tensor_zeros = torch.zeros((2, 3))

# Criar um tensor com números aleatórios
tensor_rand = torch.rand((2, 3))

print("Tensor Lista:", tensor_lista)
print("Tensor Zeros:\n", tensor_zeros)