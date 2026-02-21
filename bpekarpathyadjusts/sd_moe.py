#i'm not finding this to work. should learn the code to see if it is right
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    text = "This is a fallback input text because input.txt was not found. " * 50

print('device is: ', device)

max_iters = 5_001
eval_iters = 100
eval_interval = 1_000
n_embed = 128
block_size = 64
batch_size = 12
#messing with this
learning_rate = 5e-3
n_head = 8
n_layer = 4
dropout = 0.2
num_experts = 4 # Number of experts in the MoE layer
top_k = 2 # Number of experts to route each token to

vocab_size = 500
num_merges = vocab_size - 256

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

ids = list(tokens)
merges = {}
for i in range(num_merges):
    stats = get_stats(ids)
    if not stats:
        break
    pair = max(stats, key=stats.get)
    idx = 256 + i
    ids = merge(ids, pair, idx)
    merges[pair] = idx

print("merged")
print('len: ',len(ids))

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0,p1), idx in merges.items():
    vocab[idx] = vocab.get(p0, b"") + vocab.get(p1, b"")

def decode(ids):
    tokens = b"".join(vocab.get(idx, b"") for idx in ids)
    return tokens.decode("utf-8", errors='replace')

def encode(text):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

if text.strip() == "":
    text = "Fallback "*50

data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]

torch.manual_seed(1337)

def get_batch(split):
    data = train_data if split == 'train' else test_data
    if len(data) <= block_size:
        ix = torch.zeros((batch_size,), dtype=torch.long)
    else:
        ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y = get_batch(split)
            logits, loss = model(x,y)
            losses[k] = loss.item() if loss is not None else 0
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
        wei = q @ k.transpose(-2,-1) * c**-0.5
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

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

class SDMoELinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W_c, W_u, P_U, P_V):
        W = W_c + W_u
        ctx.save_for_backward(x, W, P_U, P_V)
        return F.linear(x, W)

    @staticmethod
    def backward(ctx, grad_output):
        x, W, P_U, P_V = ctx.saved_tensors
        grad_x = grad_W_c = grad_W_u = None
        
        if ctx.needs_input_grad[0]:
            grad_x = grad_output @ W

        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            grad_W = grad_output.t() @ x
            
            if ctx.needs_input_grad[1]:
                grad_W_c = P_U @ grad_W @ P_V
                
            if ctx.needs_input_grad[2]:
                I_out = torch.eye(P_U.size(0), device=P_U.device, dtype=P_U.dtype)
                I_in = torch.eye(P_V.size(0), device=P_V.device, dtype=P_V.dtype)
                grad_W_u = (I_out - P_U) @ grad_W @ (I_in - P_V)
                
        return grad_x, grad_W_c, grad_W_u, None, None

class SDMoELinear(nn.Module):
    def __init__(self, in_features, out_features, num_experts, svd_rank=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.svd_rank = min(svd_rank, min(in_features, out_features))
        
        self.W_c = nn.Parameter(torch.empty(out_features, in_features))
        self.W_u = nn.Parameter(torch.empty(num_experts, out_features, in_features))
        
        nn.init.kaiming_uniform_(self.W_c, a=math.sqrt(5))
        nn.init.normal_(self.W_u, mean=0.0, std=0.02)
        
        self.register_buffer('P_U', torch.eye(out_features))
        self.register_buffer('P_V', torch.eye(in_features))
        self.update_svd_bases()

    @torch.no_grad()
    def update_svd_bases(self):
        U, S, Vh = torch.linalg.svd(self.W_c.data, full_matrices=False)
        k = self.svd_rank
        Uk = U[:, :k]
        Vk = Vh[:k, :].T
        
        self.P_U.copy_(Uk @ Uk.T)
        self.P_V.copy_(Vk @ Vk.T)
        
        I_out = torch.eye(self.out_features, device=self.W_c.device)
        I_in = torch.eye(self.in_features, device=self.W_c.device)
        P_U_perp = I_out - self.P_U
        P_V_perp = I_in - self.P_V
        
        for i in range(self.num_experts):
            self.W_u.data[i] = P_U_perp @ self.W_u.data[i] @ P_V_perp

    def forward(self, x, expert_idx):
        return SDMoELinearFunction.apply(x, self.W_c, self.W_u[expert_idx], self.P_U, self.P_V)

class SDMoEFeedForward(nn.Module):
    def __init__(self, n_embed, num_experts=num_experts, top_k=top_k, svd_rank=8):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.router = nn.Linear(n_embed, num_experts, bias=False)
        
        self.w1 = SDMoELinear(n_embed, 4 * n_embed, num_experts, svd_rank=svd_rank)
        self.w2 = SDMoELinear(4 * n_embed, n_embed, num_experts, svd_rank=svd_rank)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        
        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        final_output = torch.zeros_like(x_flat)
        
        for i in range(self.num_experts):
            expert_mask = (selected_experts == i)
            idx, nth_expert = torch.where(expert_mask)
            
            if len(idx) > 0:
                x_expert = x_flat[idx]
                h = self.w1(x_expert, i)
                h = F.relu(h)
                out = self.w2(h, i)
                out = self.dropout(out)
                
                weight = routing_weights[idx, nth_expert].unsqueeze(-1)
                final_output[idx] += out * weight
                
        return final_output.view(B, T, C)

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiheadAttention(n_head, head_size)
        self.ffwd = SDMoEFeedForward(n_embed, num_experts=4, top_k=2, svd_rank=8)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) 
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size) 

    def update_svd_bases(self):
        for module in self.modules():
            if isinstance(module, SDMoELinear):
                module.update_svd_bases()

    def forward(self, idx, targets=None):
        b,t = idx.shape
        token_embed = self.token_embedding_table(idx)
        pos_embed = self.position_embedding_table(torch.arange(t, device=device))
        x = pos_embed + token_embed
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            b,t,c = logits.shape
            logits = logits.view(b*t,c)
            targets = targets.view(b*t)
            targets = torch.clamp(targets, 0, vocab_size - 1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
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

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if not iter % eval_interval:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    if iter % 16 == 0:
        m.update_svd_bases()
        
    xb, yb = get_batch('train')
    
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    if loss is not None:
        loss.backward()
        optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))
