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
n_embed = 64   #64
block_size = 64
batch_size = 16
learning_rate = 3e-4
n_head = 4  #4
n_layer = 4  #6
dropout = 0.2
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


class RePU(nn.Module):
   
    def __init__(self, initial_a=0.0, initial_b=1.0, initial_c=0.0, initial_d=1.0):
       
        super(RePU, self).__init__()
        # nn.Parameter wraps a tensor to make it a learnable parameter of the module
        self.a = nn.Parameter(torch.tensor(initial_a))
        self.b = nn.Parameter(torch.tensor(initial_b))
        self.c = nn.Parameter(torch.tensor(initial_c))
        self.d = nn.Parameter(torch.tensor(initial_d))

    def forward(self, x):
        # 1. Calculate the polynomial P(x) = a*x^2 + b*x + c
        polynomial = self.d * (x**3) + self.a * (x**2) + self.b * x + self.c
        
        # 2. Apply the rectification (i.e., max(0, P(x)))
        return F.relu(polynomial)


class Head(nn.Module):
    def __init__(self, head_size, query, key, value):
        super().__init__()
        self.query = query  # Shared linear layer for query
        self.key = key      # Shared linear layer for key
        self.value = value  # Shared linear layer for value
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x):
        b, t, c = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiheadAttention(nn.Module):
    def __init__(self, num_heads, head_size, shared_query_layers=None, shared_key_layers=None, shared_value_layers=None):
        super().__init__()
        # Use shared layers if provided (for blocks 2-4), otherwise create new ones (for block 1)
        if shared_query_layers is not None and shared_key_layers is not None and shared_value_layers is not None:
            self.head_query_layers = shared_query_layers
            self.head_key_layers = shared_key_layers
            self.head_value_layers = shared_value_layers
        else:
            self.head_query_layers = nn.ModuleList([nn.Linear(n_embed, head_size, bias=False) for _ in range(num_heads)])
            self.head_key_layers = nn.ModuleList([nn.Linear(n_embed, head_size, bias=False) for _ in range(num_heads)])
            self.head_value_layers = nn.ModuleList([nn.Linear(n_embed, head_size, bias=False) for _ in range(num_heads)])
        
        self.heads = nn.ModuleList([
            Head(head_size, self.head_query_layers[i], self.head_key_layers[i], self.head_value_layers[i])
            for i in range(num_heads)
        ])
        self.proj = nn.Linear(num_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 2 * n_embed),
            RePU(),
            nn.Linear(2 * n_embed, n_embed),
            RePU(),
            nn.Linear(n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ The standard Transformer block """
    def __init__(self, n_embed, n_head, shared_query_layers=None, shared_key_layers=None, shared_value_layers=None):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiheadAttention(n_head, head_size, shared_query_layers, shared_key_layers, shared_value_layers)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Router(nn.Module):
    """Calculates a halt probability for each token."""
    def __init__(self, n_embed):
        super().__init__()
        self.router_layer = nn.Sequential(
            nn.Linear(n_embed, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.router_layer(x)

class MoRTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        
        # Create 4 distinct blocks, with blocks 2-4 reusing Q, K, V layers from block 1
        self.block1 = Block(n_embed, n_head=n_head)
        self.blocks = nn.ModuleList([
            self.block1,
            Block(n_embed, n_head=n_head, 
                  shared_query_layers=self.block1.sa.head_query_layers,
                  shared_key_layers=self.block1.sa.head_key_layers,
                  shared_value_layers=self.block1.sa.head_value_layers),
            Block(n_embed, n_head=n_head, 
                  shared_query_layers=self.block1.sa.head_query_layers,
                  shared_key_layers=self.block1.sa.head_key_layers,
                  shared_value_layers=self.block1.sa.head_value_layers),
            Block(n_embed, n_head=n_head, 
                  shared_query_layers=self.block1.sa.head_query_layers,
                  shared_key_layers=self.block1.sa.head_key_layers,
                  shared_value_layers=self.block1.sa.head_value_layers),
            Block(n_embed, n_head=n_head, 
                  shared_query_layers=self.block1.sa.head_query_layers,
                  shared_key_layers=self.block1.sa.head_key_layers,
                  shared_value_layers=self.block1.sa.head_value_layers)
        ])
        
        # The router to decide whether to continue
        self.router = Router(n_embed)
        
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        
        # Initial embeddings
        token_embed = self.token_embedding_table(idx)
        pos_embed = self.position_embedding_table(torch.arange(t, device=device))
        x = token_embed + pos_embed

        # --- Recursive Forward Pass ---
        # Initialize tensors to track state
        final_output = torch.zeros_like(x)
        remaining_prob = torch.ones(b, t, 1, device=device)
        
        current_x = x
        for depth in range(max_recursion_depth):
            # Use one of the 4 blocks, cycling through them
            block_idx = depth % 4
            processed_x = self.blocks[block_idx](current_x)
            
            # Calculate halt probabilities using the router
            halt_prob = self.router(processed_x) # Shape: (b, t, 1)
            
            # Probability of halting at this specific depth
            prob_halting_now = remaining_prob * halt_prob
            
            # Add the contribution of this layer to the final output
            final_output += prob_halting_now * processed_x
            
            # Update the remaining probability for tokens that did not halt
            remaining_prob = remaining_prob * (1.0 - halt_prob)
            
            # The input for the next iteration is the output of the current one
            current_x = processed_x

        # For any tokens that never halted, add their final state to the output
        # weighted by the probability that they never halted.
        final_output += remaining_prob * processed_x
        
        # Final projection to logits
        x = self.ln_f(final_output)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            b, t, c = logits.shape
            logits_view = logits.view(b * t, c)
            targets_view = targets.view(b * t)
            loss = F.cross_entropy(logits_view, targets_view)

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