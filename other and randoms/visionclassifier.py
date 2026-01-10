import os
import numpy as np
import cv2 as cv
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as plt

VIDEO_DIR = 'C:/Users/cshep/Documents/datasets/videotraffic/video' #os.path.join('Users/cshep/Documents/datasets', 'video' )
freqs = 2_000
allmovies = []
j = 0
for filename in os.listdir(VIDEO_DIR):
    vpath= os.path.join( VIDEO_DIR, filename )
    cap = cv.VideoCapture(vpath)
    frames = []
    # print(cap.isOpened())
    i = 0
    
    while cap.isOpened()  and i < 40:  
        ret, frame = cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.resize(frame, (240,240))

        frames.append(frame)
        i += 1
    frames = np.array(frames)
    j+=1
    allmovies.append(frames)
allmovies = np.array(allmovies)

#normalize images to 0-1 and reshape to put channels in a usable format
allmovies = allmovies / 255.0
allmovies = allmovies.reshape([254,40,240*240])

fftmovs = np.fft.fft(allmovies)
sfftmovs = fftmovs[:,:,:freqs]
#shape is 254,40,40_000
sfftmovs = np.concatenate((sfftmovs.real,  sfftmovs.imag), axis=-1)
train_movs = torch.FloatTensor(sfftmovs[:200])
test_movs = torch.FloatTensor(sfftmovs[200:])


#gpt stuff starts here
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device is: ', device)
vocab_size = freqs*2
max_iters = 2_000
# eval_iters = 200
eval_interval = 50
n_embed = freqs*2
block_size = 10
batch_size = 8
learning_rate = 3e-4
n_head = 4
n_layer = 4
dropout = 0.2

# def  get_batch(split):
#     data = train_movs if split == 'train' else test_movs

#     ix = torch.randint(len(data), (batch_size,))
#     x = torch.stack([data[i,:block_size] for i in ix])
#     y = torch.stack([data[i,1:block_size+1] for  i in ix])
#     x,y = x.to(device), y.to(device)
#     return x,y

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b,t,c = x.shape
        k = self.key(x)
        q = self.query(x)
        #compute attention scores
        wei = q @ k.transpose(-2,-1) * c**-0.5
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf')) #(b,t,t)
        # wei = F.softmax(wei, dim=-1)
        wei = F.relu(wei)
        wei = self.dropout(wei)
        #perform weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out

class MultiheadAttention(nn.Module):
    def __init__(self,num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(), 
            nn.Linear(4*n_embed, n_embed), 
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiheadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed) #final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size) 

    def forward(self, idx, targets=None):
        x = self.blocks(idx)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            b,t,c = logits.shape
            logits = logits.view(b*t,c)
            targets = targets.view(b*t, c)
            loss = F.mse_loss(logits, targets)
        return logits, loss
    
    
model = BigramLanguageModel()
m = model.to(device)
# xb, yb = get_batch('train')
# logits, loss = m(xb,yb)
# # print('x ', xb.shape)
# logits= logits.view(batch_size, block_size,  freqs*2)
# # print(logits.shape)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):

    #sample batch of data
    xb, yb = get_batch('train')
    
    #evaluate loss
    logits, loss = m(xb, yb)
    if not iter % eval_interval:
        print('step ', iter, ' loss ', loss.item())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
