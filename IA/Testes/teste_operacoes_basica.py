import torch

a = torch.tensor([10.0, 20.0, 30.0])
b = torch.tensor([2.0, 3.0, 4.0])

soma = a + b              # [12.0, 23.0, 34.0]
multiplicacao = a * 2     # Multiplica todos por 2 -> [20.0, 40.0, 60.0]

print("Soma:", soma)
print("Multiplicação por escalar:", multiplicacao)