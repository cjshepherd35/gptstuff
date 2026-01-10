import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
import sys
sys.stdout.reconfigure(encoding="utf-8")

textraw = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
sample = textraw['train'].select(range(100_000))
rows = sample["text"]
# Join into one long string (with spaces or newlines)
text = " ".join(rows)  

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('device is: ', device)
chars = sorted(list(set(text)))
vocab_size = len(chars)
#parameters to tweak
max_iters = 30_001
eval_iters = 100
eval_interval =  2000  #500
n_embed = 32   #64
block_size = 64
batch_size = 16
learning_rate = 3e-4
n_head = 4  #4
n_layer = 6  #6
dropout = 0.2
# MoR-specific hyperparameters
max_recursion_depth = 4 # Maximum number of recursions
dim_growth_factor = 1.5 # Factor by which dimension increases at each step
max_recursion_depth = 3

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

# --- Dimension-Aware Transformer Components ---
class Head(nn.Module):
    # This module remains mostly the same, as its internal head_size
    # is derived from the input dimension of its parent MultiheadAttention module.
    def __init__(self, n_embed_in, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed_in, head_size, bias=False)
        self.query = nn.Linear(n_embed_in, head_size, bias=False)
        self.value = nn.Linear(n_embed_in, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, t, c = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiheadAttention(nn.Module):
    """ Multi-head attention that can project from n_embed_in to n_embed_out. """
    def __init__(self, n_embed_in, n_embed_out, num_heads):
        super().__init__()
        assert n_embed_in % num_heads == 0
        head_size = n_embed_in // num_heads
        self.heads = nn.ModuleList([Head(n_embed_in, head_size) for _ in range(num_heads)])
        # The final projection layer is what changes the dimension
        self.proj = nn.Linear(n_embed_in, n_embed_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ A simple feed-forward layer that can project from n_embed_in to n_embed_out. """
    def __init__(self, n_embed_in, n_embed_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed_in, 4 * n_embed_in),
            nn.ReLU(),
            nn.Linear(4 * n_embed_in, n_embed_out), # This layer handles the projection
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block with dimension-changing capability. """
    def __init__(self, n_embed_in, n_embed_out, n_head):
        super().__init__()
        self.sa = MultiheadAttention(n_embed_in, n_embed_out, n_head)
        self.ffwd = FeedForward(n_embed_out, n_embed_out) # FFWD operates on the new dimension
        self.ln1 = nn.LayerNorm(n_embed_in)
        self.ln2 = nn.LayerNorm(n_embed_out)
        
        # Projection for the residual connection if dimensions don't match
        if n_embed_in == n_embed_out:
            self.residual_proj = nn.Identity()
        else:
            self.residual_proj = nn.Linear(n_embed_in, n_embed_out)

    def forward(self, x):
        # Project residual path first, then add attention output
        x = self.residual_proj(x) + self.sa(self.ln1(x))
        # Second residual connection doesn't need projection as ffwd maintains dimension
        x = x + self.ffwd(self.ln2(x))
        return x

# --- Router and Main Model ---

class Router(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.router_layer = nn.Sequential(nn.Linear(n_embed, 1), nn.Sigmoid())

    def forward(self, x):
        return self.router_layer(x)

class MoRTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- Define dimensions for each recursion depth ---
        dims = [n_embed]
        current_dim = n_embed
        for _ in range(max_recursion_depth):
            # Calculate the next dimension and make it divisible by n_head
            next_dim = int(current_dim * dim_growth_factor)
            next_dim = (next_dim + n_head - 1) // n_head * n_head # round up to nearest multiple of n_head
            dims.append(next_dim)
            current_dim = next_dim
        print(f"Dimension progression: {dims}")
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        
        # Create lists of blocks, routers, and projectors for each depth
        self.blocks = nn.ModuleList([Block(dims[i], dims[i+1], n_head) for i in range(max_recursion_depth)])
        self.routers = nn.ModuleList([Router(dims[i+1]) for i in range(max_recursion_depth)])
        self.output_projections = nn.ModuleList([nn.Linear(dims[i+1], n_embed) for i in range(max_recursion_depth)])

        self.ln_f = nn.LayerNorm(n_embed) # Final layer norm operates on the original dimension
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        token_embed = self.token_embedding_table(idx)
        pos_embed = self.position_embedding_table(torch.arange(t, device=device))
        x = token_embed + pos_embed

        # --- Recursive Forward Pass ---
        final_output = torch.zeros_like(x)
        remaining_prob = torch.ones(b, t, 1, device=device)
        
        current_x = x
        for depth in range(max_recursion_depth):
            # Get the specific block, router, and projector for this depth
            block = self.blocks[depth]
            router = self.routers[depth]
            projection = self.output_projections[depth]
            
            processed_x = block(current_x)
            halt_prob = router(processed_x)
            
            # prob_halting_now = remaining_prob * halt_prob
            
            # # Project the high-dimensional output back to n_embed before accumulating
            # final_output += prob_halting_now * projection(processed_x)
            
            # remaining_prob = remaining_prob * (1.0 - halt_prob)
            current_x = processed_x

        # Add the final state, projected back to n_embed
        final_state_projection = self.output_projections[-1](processed_x)
        final_output += remaining_prob * final_state_projection
        
        x = self.ln_f(final_output)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            b, t, c = logits.shape
            loss = F.cross_entropy(logits.view(b*t, c), targets.view(b*t))

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

model = MoRTransformer()
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