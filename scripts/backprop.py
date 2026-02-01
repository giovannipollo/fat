import torch

# 1. Define inputs with gradient tracking enabled
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# 2. Perform the operations (The Forward Pass)
a = x + y
out = a * y

target = torch.tensor(17.0)
loss = (out - target) ** 2

# 3. Trigger the backpropagation
loss.backward()

# 4. Check the results
print(f"Output: {out.item()}") # 15.0
print(f"Gradient of x (d_out/dx): {x.grad}") # 3.0
print(f"Gradient of y (d_out/dy): {y.grad}") # 8.0