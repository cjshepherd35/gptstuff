import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
import sys
sys.stdout.reconfigure(encoding="utf-8")

textraw = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
sample = textraw['train'].select(range(10000))

rows = sample["text"]

# Join into one long string (with spaces or newlines)
text = " ".join(rows)  

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# with open('input.txt', 'r', encoding='utf-8') as f:
#     text = f.read()
print('device is: ', device)
chars = sorted(list(set(text)))
vocab_size = len(chars)
#parameters to tweak
max_iters = 15001
eval_iters = 100
eval_interval =  1000  #500
n_embed = 88   #64
block_size = 64
batch_size = 16
learning_rate = 3e-4
n_head = 4  #4
n_layer = 4
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
# xb, yb = get_batch('train')

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y = get_batch(split)
            logits, loss = model(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class MiniHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed//2, head_size, bias=False)
        self.query = nn.Linear(n_embed//2, head_size, bias=False)
        self.value = nn.Linear(n_embed//2, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.final = nn.Linear(head_size, head_size)

    def forward(self, x):
        b,t,c = x.shape
        k = self.key(x)
        q = self.query(x)
        #compute attention scores
        wei = q @ k.transpose(-2,-1) * c**-0.5
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf')) #(b,t,t)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        #perform weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        out = self.final(out)
        return out
    

class MinisecondHead(nn.Module):
    def __init__(self, head_size, prevkey, prevquery, prevvalue):
        super().__init__()
        self.key = prevkey
        self.query = prevquery
        self.value = prevvalue
        # self.key = nn.Linear(n_embed//2, head_size, bias=False)
        # self.query = nn.Linear(n_embed//2, head_size, bias=False)
        # self.value = nn.Linear(n_embed//2, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.final = nn.Linear(head_size, head_size)

    def forward(self, x):
        b,t,c = x.shape
        k = self.key(x)
        q = self.query(x)
        #compute attention scores
        wei = q @ k.transpose(-2,-1) * c**-0.5
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf')) #(b,t,t)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        #perform weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        out = self.final(out)
        return out

class MiniMultiheadAttention(nn.Module):
    def __init__(self,num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([MiniHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed//2, n_embed//2)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class MinisecMultiheadAttention(nn.Module):
    def __init__(self,num_heads, head_size,prevkey, prevquery, prevvalue):
        super().__init__()
        self.heads = nn.ModuleList([MinisecondHead(head_size,prevkey[i], prevquery[i], prevvalue[i] ) for i in range(num_heads)])
        self.proj = nn.Linear(n_embed//2, n_embed//2)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class MiniFeedForward(nn.Module):
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
    
class MiniBlock(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MiniMultiheadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MinisecBlock(nn.Module):
    def __init__(self, n_embed, n_head, prevkey, prevquery, prevvalue):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MinisecMultiheadAttention(n_head, head_size, prevkey, prevquery, prevvalue)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



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
        wei = F.softmax(wei, dim=-1)
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
        self.final = nn.Linear(n_embed, n_embed//2)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        x = self.final(x)
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        #each token reads off the logits for the next tokenfrom a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) 
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.fblock = Block(n_embed=n_embed, n_head=n_head)
        #should have made this with the blocks layers but too busy to figure that out now..
        self.fmblock = MiniBlock(n_embed=n_embed//2, n_head=n_head)
        prevkeys = []
        prevqueries = []
        prevvalues = []
        for head in self.fmblock.sa.heads:
            prevkeys.append(head.key)
            prevqueries.append(head.query)
            prevvalues.append(head.value)
        self.blocks = nn.Sequential(*[MinisecBlock(n_embed=n_embed//2, n_head=n_head, prevkey=prevkeys, prevquery=prevqueries, prevvalue=prevvalues) for  _ in range(n_layer-2)])
        # self.blocks = nn.Sequential(*[MiniBlock(n_embed=n_embed//2, n_head=n_head) for  _ in range(n_layer-1)])
        # self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed//2) #final layer norm
        self.lm_head = nn.Linear(n_embed//2, vocab_size) 

    def forward(self, idx, targets=None):
        b,t = idx.shape
        #idx and targets are both (b,t) tensor of integers
        token_embed = self.token_embedding_table(idx) #(b,t,c)
        pos_embed = self.position_embedding_table(torch.arange(t, device=device)) #also (b,t,c)
        x = pos_embed + token_embed
        x = self.fblock(x)
        x = self.fmblock(x)
        x = self.blocks(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            b,t,c = logits.shape
            logits = logits.view(b*t,c)
            targets = targets.view(b*t)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the  last block_size tokens
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel()
total_params = sum(p.numel() for p in model.parameters())
print('size of model',total_params)
m = model.to(device)

# logits, loss = m(xb,yb)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):

    #every once in awhile evaluate the loss on traon and val sets
    if not iter % eval_interval:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    #sample batch of data
    xb, yb = get_batch('train')
    
    #evaluate loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = idx=torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))