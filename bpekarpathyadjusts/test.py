import torch
b = 2
s = 1
n = 4
c = 5
x = torch.tensor(torch.randn(b,n,s,c))
res = torch.tensor( torch.eye(n) + torch.randn(n,n)*0.01)

print(x)
