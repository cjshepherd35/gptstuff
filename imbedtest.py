import torch
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print('device is: ', device)
chars = sorted(list(set(text)))
vocab_size = len(chars)
max_iters = 5000
eval_iters = 200
eval_interval = 500
n_embed = int(384/2)
block_size = int(256/2)
batch_size = 12
learning_rate = 3e-4
n_head = 6
n_layer = 6
dropout = 0.2

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]

torch.manual_seed(1337)


def get_batch(split):
    #generate a small batch of data of inputs x and y
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for  i in ix])
    x,y = x.to(device), y.to(device)
    return x,y
xb, yb = get_batch('train')

class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.toke_embed = nn.Embedding(3,5)
    def forward(self, x):
        return self.toke_embed(x)
    
print(xb.shape)
m = net().to(device)
out = m(xb)
print(out.shape)