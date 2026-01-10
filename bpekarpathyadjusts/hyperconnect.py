#doesn't have a causal mask so it is not finished. will look like a perfect model if trained. 
#also does not have the decoding of the tokens yet.
import torch
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print('device is: ', device)

#parameters to tweak
max_iters = 1_001
eval_iters = 100
eval_interval =  500
n_embed = 64   #64
block_size = 64
batch_size = 12
learning_rate = 3e-4
n_head = 4  #4
n_layer = 4  #6
dropout = 0.2 
expans_rate = 4

vocab_size = 1100 #my own preset number, may need changing
num_merges = vocab_size - 256 #256 is how many distinct utf-8 tokens there are.

tokens = text.encode("utf-8")
tokens = list(map(int, tokens))

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i<len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i+= 2
        else:
            newids.append(ids[i])
            i+=1
    return newids

# ids = list(tokens)

merges = {}

vocab = {idx: bytes([idx]) for  idx in range(256)}
for (p0,p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]
def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors='replace')
    return text

def encode(text):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break #nothing else can be merged.
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]

torch.manual_seed(1337)
# print(train_data[:50])


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

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. The Hyper-Connection Wrapper (Same as before) ---
class HyperConnectionWrapper(nn.Module):
    def __init__(self, base_width, expansion_rate, sub_layer):
        """
        Wraps a standard sub-layer (Attn or MLP) with Hyper-Connections.
        """
        super().__init__()
        self.C = base_width
        self.n = expansion_rate
        self.sub_layer = sub_layer # The function F(.)

        # Learnable Mappings
        # H_res: Mixes the n streams directly (n x n)
        self.H_res = nn.Parameter(torch.eye(self.n) + torch.randn(self.n, self.n) * 0.01)
        
        # H_pre: Compresses n streams -> 1 input (1 x n)
        self.H_pre = nn.Parameter(torch.randn(self.n) * 0.02)
        
        # H_post: Expands 1 output -> n streams (1 x n)
        self.H_post = nn.Parameter(torch.randn(self.n) * 0.02)

    def forward(self, x):
        # x shape: (Batch, Seq, n*C)
        B, S, _ = x.shape
        
        # View as (Batch, Seq, n, C)
        x_reshaped = x.view(B, S, self.n, self.C)

        # 1. Residual Path: Mix streams
        # (B, S, n, C) @ (n, n) -> (B, S, n, C)
        res_branch = torch.einsum('bsnc, nm -> bsnc', x_reshaped, self.H_res)

        # 2. Compute Path: Compress -> Compute -> Expand
        # Compress: (B, S, n, C) @ (n) -> (B, S, C)
        layer_input = torch.einsum('bsnc, n -> bsc', x_reshaped, self.H_pre)
        
        # Compute: F(.)
        layer_output = self.sub_layer(layer_input)
        
        # Expand: (B, S, C) @ (n) -> (B, S, n, C)
        post_branch = torch.einsum('bsc, n -> bsnc', layer_output, self.H_post)

        # Combine and flatten
        return (res_branch + post_branch).view(B, S, -1)


# --- 2. Standard Components (Attention & MLP) ---
# These operate strictly on the narrower dimension 'C'

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.ln = nn.LayerNorm(d_model) # Pre-Norm
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        
    def forward(self, x):
        # x: (Batch, Seq, C)
        x_norm = self.ln(x)
        # Standard PyTorch Attention usually requires mask for causal, omitted here for brevity
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        return attn_out

class MLP(nn.Module):
    def __init__(self, d_model, expansion=4):
        super().__init__()
        self.ln = nn.LayerNorm(d_model) # Pre-Norm
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Linear(d_model * expansion, d_model)
        )

    def forward(self, x):
        return self.net(self.ln(x))


# --- 3. The Hyper-Connected Transformer Block ---
class HCTransformerBlock(nn.Module):
    def __init__(self, base_width, expansion_rate, num_heads=4):
        super().__init__()
        
        # The internal computation width (C)
        self.C = base_width
        # The expansion rate (n)
        self.n = expansion_rate
        
        # -- Sub-Layer 1: Attention --
        # We create a standard attention block of width C
        attn_core = CausalSelfAttention(d_model=base_width, num_heads=num_heads)
        # We wrap it in HC to handle the n*C stream
        self.hc_attn = HyperConnectionWrapper(base_width, expansion_rate, sub_layer=attn_core)
        
        # -- Sub-Layer 2: MLP --
        # We create a standard MLP of width C
        mlp_core = MLP(d_model=base_width)
        # We wrap it in HC
        self.hc_mlp = HyperConnectionWrapper(base_width, expansion_rate, sub_layer=mlp_core)

    def forward(self, x):
        """
        Input x: (Batch, Seq, n * C) - The wide residual stream
        """
        # Pass through HC-Attention
        # x = H_res * x + H_post * Attn(H_pre * x)
        x = self.hc_attn(x)
        
        # Pass through HC-MLP
        # x = H_res * x + H_post * MLP(H_pre * x)
        x = self.hc_mlp(x)
        
        return x


# Verify parameter count logic
# A standard transformer with width 256 is huge.
# This one has the FLOPs of width 64, but the memory width of 256.
class Transformer(nn.Module):

    def __init__(self):
        super().__init__()
        #each token reads off the logits for the next tokenfrom a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) 
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.C = n_embed
        self.n = expans_rate
        self.stream_dim = self.n * self.C
        self.entry_expansion = nn.Linear(n_embed, self.stream_dim)
        self.blocks = nn.Sequential(*[HCTransformerBlock(n_embed, expans_rate, n_head) for _ in range(n_layer)])
        self.exit_compression = nn.Linear(self.stream_dim, n_embed)
        self.ln_f = nn.LayerNorm(n_embed) #final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size) 

    def forward(self, idx, targets=None):
        b,t = idx.shape
        #idx and targets are both (b,t) tensor of integers
        token_embed = self.token_embedding_table(idx) #(b,t,c)
        pos_embed = self.position_embedding_table(torch.arange(t, device=device)) #also (b,t,c)
        x = pos_embed + token_embed
        x = self.entry_expansion(x)
        x = self.blocks(x)
        x = self.exit_compression(x)
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