#this is amazing!
import torch
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print('device is: ', device)
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(vocab_size)
#parameters to tweak
max_iters = 1001
eval_iters = 100
eval_interval =  200  #500
n_embed = 64   #64
block_size = 64
batch_size = 12
learning_rate = 3e-4
n_head = 4  #4
n_layer = 6  #6
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




def get_retnet_gammas(n_heads):
    # The heads have different decay rates based on the formula:
    # gamma = 1 - 2^(-5 - i)  where i is the head index
    
    # Create index vector [0, 1, ..., n_heads-1]
    indices = torch.arange(0, n_heads).float()
    
    # Apply the formula
    gammas = 1 - 2 ** (-5 - indices)
    
    return gammas.tolist()

def create_retention_mask(seq_len, gamma, device='cpu'):
    
    # 1. Create indices for rows (n) and columns (m)
    # n: shape [seq_len, 1] -> [[0], [1], [2], ...]
    n = torch.arange(seq_len, device=device).unsqueeze(1)
    # m: shape [1, seq_len] -> [[0, 1, 2, ...]]
    m = torch.arange(seq_len, device=device).unsqueeze(0)
    
    # 2. Compute the relative distance (n - m)
    # This broadcasts to shape [seq_len, seq_len]
    distance = n - m
    
    # 3. Create the causal mask (lower triangular)
    # We only care about positions where n >= m (past and present)
    causal_mask = distance >= 0
    
    # 4. Calculate the decay values: gamma^(n-m)
    # We initialize a matrix of zeros
    D = torch.zeros(seq_len, seq_len, device=device)
    
    # Fill the valid lower-triangular part with calculated decay values
    # Note: We only compute power for valid positions to ensure numerical stability/correctness,
    # though gamma^(negative) would just be handled by the mask zeroing it out anyway.
    D[causal_mask] = gamma ** distance[causal_mask].float()
    
    return D


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
class RetentionHead(nn.Module):
    def __init__(self, head_size, n_embed, gamma):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.gamma = gamma # Store the specific gamma for this head
        
        # RetNet typically uses GroupNorm, but we will keep Dropout here 
        # to match your original structure if needed, though RetNet 
        # relies less on Dropout due to the decay mechanism.
        self.dropout = nn.Dropout(0.1) 

    def forward(self, x):
        b, t, c = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # 1. Compute Attention Scores (Q @ K^T)
        # Note: RetNet allows for an optional scaling factor, often kept for stability.
        # Some implementations remove the scale, but we keep it for convergence similar to Transformers.
        wei = q @ k.transpose(-2, -1) * (c**-0.5) 

        # 2. Generate the RetNet Decay Mask (D)
        # We generate this on the fly for the current sequence length 't'
        D = create_retention_mask(t, self.gamma, device=x.device)
        
        # 3. Apply the Mask (Hadamard Product)
        # CRITICAL CHANGE: We multiply element-wise, we do NOT mask_fill with -inf
        wei = wei * D 
        
        # 4. No Softmax!
        # In RetNet, the decay matrix D acts as the normalization mechanism.
        
        # 5. Aggregate Values
        out = wei @ v
        
        return out

class MultiheadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        
        # 1. Get the distinct gamma value for each head
        gammas = get_retnet_gammas(num_heads)
        
        # 2. Create heads, passing a unique gamma to each
        self.heads = nn.ModuleList([
            RetentionHead(head_size, n_embed, gamma=g) for g in gammas
        ])
        
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        
        # RetNet often uses GroupNorm here instead of just LayerNorm/Projection
        # But sticking to your interface:
        self.group_norm = nn.GroupNorm(num_heads, n_embed) 

    def forward(self, x):
        # Run each head (Parallel Representation)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        
        # The RetNet paper suggests GroupNorm on the concatenated output
        # We verify shape is (B, T, C) -> GroupNorm expects (B, C, T) usually, 
        # so we might need to permute or just rely on the projection for this snippet.
        # For simplicity/compatibility with your code, we proceed with projection:
        
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

class Transformer(nn.Module):

    def __init__(self):
        super().__init__()
        #each token reads off the logits for the next tokenfrom a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) 
        # self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed) #final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size) 

    def forward(self, idx, targets=None):
        b,t = idx.shape
        #idx and targets are both (b,t) tensor of integers
        token_embed = self.token_embedding_table(idx) #(b,t,c)
        # pos_embed = self.position_embedding_table(torch.arange(t, device=device)) #also (b,t,c)
        x = token_embed #pos_embed + token_embed
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

model = Transformer()
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



