import torch

# Define 'x' com suporte para rastrear o gradiente (requires_grad=True)
x = torch.tensor(3.0, requires_grad=True)

# Define uma função: y = x² + 5
y = x**2 + 5

# Pede para o PyTorch calcular a derivada de 'y' em relação a 'x'
y.backward()

# A derivada de (x² + 5) é 2x. Como x = 3, o gradiente esperado é 2*(3) = 6.
print("Valor de y:", y.item())
print("Gradiente dy/dx em x=3:", x.grad.item())