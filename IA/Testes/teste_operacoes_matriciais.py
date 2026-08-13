import torch

# Matriz A (2x2) e Matriz B (2x2)
matriz_a = torch.tensor([[1.0, 2.0], 
                         [3.0, 4.0]])

matriz_b = torch.tensor([[5.0, 6.0], 
                         [7.0, 8.0]])

# Multiplicação matricial (não é elemento por elemento!)
resultado = matriz_a @ matriz_b

print("Resultado da Multiplicação Matricial:\n", resultado)