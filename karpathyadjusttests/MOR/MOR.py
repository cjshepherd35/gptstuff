#not fully created yet, can't figure out the router. check gemini with mixture of recursions router as name

import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
import sys
sys.stdout.reconfigure(encoding="utf-8")
torch.autograd.set_detect_anomaly(True)

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
max_iters = 1
eval_iters = 1
eval_interval =  200  #500
n_embed = 64   
block_size = 32
batch_size = 16
learning_rate = 3e-4
n_head = 4  
n_layer = 6  
dropout = 0.2
max_recursion_depth = 3
halt_threshold = 0.5

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
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Router(nn.Module):
    """Calculates a halt probability for each token."""
    def __init__(self, block_size, n_embed):
        super().__init__()
        self.router_layer = nn.Sequential(
            nn.Linear(block_size*n_embed, block_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.router_layer(x)


# class TopKReduceExpand(nn.Module):
#     def __init__(self, keep_ratio: float = 0.25, op: nn.Module = None):
#         """
#         keep_ratio: fraction of tokens/channels to keep (0 < keep_ratio <= 1)
#         op: operation to apply on reduced set (e.g. nn.Linear, nn.Conv1d, etc.)
#         """
#         super().__init__()
#         self.keep_ratio = keep_ratio
#         self.op = op if op is not None else nn.Identity()

#     def forward(self, x: torch.Tensor, p: torch.Tensor):
#         """
#         x: (batch, C, L)
#         p: (batch, C, 1) probabilities
#         """
#         B, C, L = x.shape
#         k = max(1, int(C * self.keep_ratio))  # number of channels to keep

#         # 1. Top-k selection along dim=1
#         scores = p.squeeze(-1)                # (B, C)
#         _, idx = torch.topk(scores, k, dim=1) # (B, k)

#         # 2. Gather to reduced form
#         idx_exp = idx.unsqueeze(-1).expand(-1, -1, L)   # (B, k, L)
#         x_reduced = torch.gather(x, 1, idx_exp)         # (B, k, L)

#         # 3. Apply operation on reduced tokens
#         y_reduced = self.op(x_reduced)                  # (B, k, L) -> same shape assumed

#         # 4. Scatter back into full shape
#         y_full = torch.zeros_like(x)                    # (B, C, L)
#         y_full = y_full.scatter(1, idx_exp, y_reduced)  # restore original positions

#         return y_full, idx, y_reduced



class MoRTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.router = Router(block_size, n_embed)
        # self.routers = nn.Sequential(*[Router(block_size//(2**i)) for i in range(max_recursion_depth)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, x, targets=None):
        b, t = x.shape
        token_embed = self.token_embedding_table(x)
        pos_embed = self.position_embedding_table(torch.arange(t, device=device))
        x = token_embed + pos_embed

        # --- Hard Routing Forward Pass ---
        final_output = torch.zeros_like(x)
        # active_mask = torch.ones(b, t, dtype=torch.bool, device=device)
        
        current_x = x
        full_x = x
        idx = torch.arange(current_x.size(1))
        k = int(x.size(1)*halt_threshold)
        for depth in range(max_recursion_depth):
            # Process the current state through the transformer block
            processed_x = self.blocks(current_x)
            update_matrix = torch.zeros_like(full_x)
            update_matrix[:, idx, :] = processed_x
            full_x = full_x + update_matrix
            print("pro")
            print(processed_x.shape)
            processed_x = processed_x.view(n_embed,batch_size*block_size)
            # Calculate halt probabilities using the router
            halt_prob = self.router(processed_x) # Shape: (b, t, 1)
            # get top-k indices along dim=1
            print("halt")
            print(halt_prob.shape)
            _, idx = torch.topk(halt_prob.squeeze(-1), k, dim=1)  # (16, k)
            k = int(k*halt_threshold)
            # expand indices to match x’s last dim
            idx_expanded = idx.unsqueeze(-1).expand(-1, -1, x.size(-1))  # (16, k, 64)

            # gather top-k elements from x
            current_x = torch.gather(current_x, 1, idx_expanded)  # (16, k, 64)

            
        final_output = full_x
            
        # Final projection to logits
        x = self.ln_f(final_output)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            b, t, c = logits.shape
            loss = F.cross_entropy(logits.view(b*t, c), targets.view(b*t))

        return logits, loss
    
    # def generate(self, idx, max_new_tokens):
    #     for _ in range(max_new_tokens):
    #         # crop idx to the  last block_size tokens
    #         idx_cond = idx[:, -block_size:]
    #         logits, loss = self(idx_cond)
    #         logits = logits[:,-1,:]
    #         probs = F.softmax(logits, dim=-1)
    #         idx_next = torch.multinomial(probs, num_samples=1)
    #         idx = torch.cat((idx, idx_next), dim=1)
    #     return idx

model = MoRTransformer()
total_params = sum(p.numel() for p in model.parameters())
print('size of model',total_params)
m = model.to(device)

# logits, loss = m(xb,yb)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):

    #every once in awhile evaluate the loss on traon and val sets
    # if not iter % eval_interval:
    #     losses = estimate_loss()
    #     print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    #sample batch of data
    xb, yb = get_batch('train')
    
    #evaluate loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# context = idx=torch.zeros((1,1), dtype=torch.long, device=device)
# print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))