import torch
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print('device is: ', device)

#parameters to tweak
max_iters =  5_001  #1_001
eval_iters = 100
eval_interval =  1_000
n_embed = 128   #64
block_size = 64
batch_size = 12
learning_rate = 3e-3
n_head = 8  #4
n_layer = 8  #6
dropout = 0.2

# SD-MoE spectral decomposition hyperparameters
sd_rank = 4              # rank k of the shared dominant subspace per head weight
svd_refresh_interval = 16  # refresh SVD bases every N optimizer steps

vocab_size = 500 #my own preset number, may need changing
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

ids = list(tokens)
merges = {}
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    ids = merge(ids, pair, idx)
    merges[pair] = idx

print("merged")
print('len: ',len(ids))

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

def _orthogonal_complement_init(shared_U, shared_V, rows, cols, rank):
    """Initialize a (rows x cols) matrix in the orthogonal complement of shared_U / shared_V.
    Matches Algorithm 1 of the SD-MoE paper."""
    # --- Left basis orthogonal to shared_U (shape rows x rank) ---
    Z_left = torch.randn(rows, rank)
    if shared_U is not None:
        Z_left = Z_left - shared_U @ (shared_U.T @ Z_left)  # project out shared directions
    Q_left, _ = torch.linalg.qr(Z_left)
    Q_left = Q_left[:, :rank]

    # --- Right basis orthogonal to shared_V (shape cols x rank) ---
    Z_right = torch.randn(cols, rank)
    if shared_V is not None:
        Z_right = Z_right - shared_V @ (shared_V.T @ Z_right)
    Q_right, _ = torch.linalg.qr(Z_right)
    Q_right = Q_right[:, :rank]

    # Scale with small values for the "tail" singular spectrum
    scale = 0.02
    return scale * (Q_left @ Q_right.T)


class SpectralHead(nn.Module):
    """Single attention head using SD-MoE spectral decomposition.
    W_eff = W_c + W_u  where W_c is a shared low-rank common subspace
    (owned by SpectralMultiheadAttention) and W_u is this head's unique
    orthogonal component."""

    def __init__(self, head_size, Wq_c, Wk_c, Wv_c, Uq, Uk, Uv, Vq, Vk, Vv):
        super().__init__()
        self.head_size = head_size
        # References to shared W_c tensors (not parameters here; owned by parent)
        self.Wq_c = Wq_c
        self.Wk_c = Wk_c
        self.Wv_c = Wv_c

        # Head-specific unique components — initialized orthogonal to W_c
        self.Wq_u = nn.Parameter(_orthogonal_complement_init(Uq, Vq, n_embed, head_size, sd_rank))
        self.Wk_u = nn.Parameter(_orthogonal_complement_init(Uk, Vk, n_embed, head_size, sd_rank))
        self.Wv_u = nn.Parameter(_orthogonal_complement_init(Uv, Vv, n_embed, head_size, sd_rank))

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, t, c = x.shape
        # Effective weights = shared + unique
        Wq_eff = self.Wq_c + self.Wq_u   # (n_embed, head_size)
        Wk_eff = self.Wk_c + self.Wk_u
        Wv_eff = self.Wv_c + self.Wv_u
        # Linear projections: x @ W gives (b, t, head_size)
        q = x @ Wq_eff
        k = x @ Wk_eff
        v = x @ Wv_eff
        # Attention scores
        wei = q @ k.transpose(-2, -1) * c ** -0.5
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class SpectralMultiheadAttention(nn.Module):
    """Multi-head attention using SD-MoE spectral decomposition.
    Owns the shared W_c parameters for Q/K/V and schedules periodic SVD refresh."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self._step = 0

        # --- Shared W_c parameters (shape: n_embed x head_size) ---
        # Initialised with small normal values so early training is stable
        self.Wq_c = nn.Parameter(torch.randn(n_embed, head_size) * 0.02)
        self.Wk_c = nn.Parameter(torch.randn(n_embed, head_size) * 0.02)
        self.Wv_c = nn.Parameter(torch.randn(n_embed, head_size) * 0.02)

        # Compute initial SVD bases from W_c for orthogonal init of W_u
        Uq, Uk, Uv, Vq, Vk, Vv = self._compute_bases()

        # Build heads with unique orthogonal components
        self.heads = nn.ModuleList([
            SpectralHead(head_size, self.Wq_c, self.Wk_c, self.Wv_c,
                         Uq, Uk, Uv, Vq, Vk, Vv)
            for _ in range(num_heads)
        ])

        # Register gradient hooks for decoupled updates
        self._register_gradient_hooks(Uq, Uk, Uv, Vq, Vk, Vv)

        self.proj = nn.Linear(num_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # SVD helpers
    # ------------------------------------------------------------------
    def _compute_bases(self):
        """Run SVD on each W_c and return U_k, V_k (top-k bases)."""
        def top_k_bases(W):
            with torch.no_grad():
                U, S, Vh = torch.linalg.svd(W.detach(), full_matrices=False)
                k = min(sd_rank, U.shape[1])
                return U[:, :k].clone(), Vh[:k, :].T.clone()  # (rows,k), (cols,k)
        Uq, Vq = top_k_bases(self.Wq_c)
        Uk, Vk = top_k_bases(self.Wk_c)
        Uv, Vv = top_k_bases(self.Wv_c)
        return Uq, Uk, Uv, Vq, Vk, Vv

    def _register_gradient_hooks(self, Uq, Uk, Uv, Vq, Vk, Vv):
        """Attach gradient hooks to W_c and W_u params for spectral decoupling."""
        # Store bases as buffers so they persist
        self._Uq, self._Vq = Uq, Vq
        self._Uk, self._Vk = Uk, Vk
        self._Uv, self._Vv = Uv, Vv

        def make_shared_hook(U, V):
            """W_c receives only the shared (dominant) gradient component."""
            def hook(grad):
                d = grad.device
                U_ = U.to(d); V_ = V.to(d)
                PU = U_ @ U_.T
                PV = V_ @ V_.T
                Gc = PU @ grad + (torch.eye(PU.shape[0], device=d) - PU) @ grad @ PV
                return Gc
            return hook

        def make_unique_hook(U, V):
            """W_u receives only the orthogonal (unique) gradient component."""
            def hook(grad):
                d = grad.device
                U_ = U.to(d); V_ = V.to(d)
                PU = U_ @ U_.T
                PV = V_ @ V_.T
                I_rows = torch.eye(PU.shape[0], device=d)
                I_cols = torch.eye(PV.shape[0], device=d)
                Gu = (I_rows - PU) @ grad @ (I_cols - PV)
                return Gu
            return hook

        # Shared hooks on W_c
        if self.Wq_c.requires_grad:
            self.Wq_c.register_hook(make_shared_hook(Uq, Vq))
            self.Wk_c.register_hook(make_shared_hook(Uk, Vk))
            self.Wv_c.register_hook(make_shared_hook(Uv, Vv))

        # Unique hooks on each head's W_u
        for head in self.heads if hasattr(self, 'heads') else []:
            head.Wq_u.register_hook(make_unique_hook(Uq, Vq))
            head.Wk_u.register_hook(make_unique_hook(Uk, Vk))
            head.Wv_u.register_hook(make_unique_hook(Uv, Vv))

    def refresh_svd(self):
        """Recompute SVD bases from current W_c and re-register gradient hooks.
        Called every svd_refresh_interval optimizer steps."""
        Uq, Uk, Uv, Vq, Vk, Vv = self._compute_bases()
        # Remove old hooks by re-registering (PyTorch hooks are additive, so we
        # track handles and clear them)
        for h in getattr(self, '_hook_handles', []):
            h.remove()
        self._hook_handles = []

        def make_shared_hook(U, V):
            def hook(grad):
                d = grad.device
                U_ = U.to(d); V_ = V.to(d)
                PU = U_ @ U_.T
                PV = V_ @ V_.T
                Gc = PU @ grad + (torch.eye(PU.shape[0], device=d) - PU) @ grad @ PV
                return Gc
            return hook

        def make_unique_hook(U, V):
            def hook(grad):
                d = grad.device
                U_ = U.to(d); V_ = V.to(d)
                PU = U_ @ U_.T
                PV = V_ @ V_.T
                I_r = torch.eye(PU.shape[0], device=d)
                I_c = torch.eye(PV.shape[0], device=d)
                return (I_r - PU) @ grad @ (I_c - PV)
            return hook

        handles = []
        if self.Wq_c.requires_grad:
            handles.append(self.Wq_c.register_hook(make_shared_hook(Uq, Vq)))
            handles.append(self.Wk_c.register_hook(make_shared_hook(Uk, Vk)))
            handles.append(self.Wv_c.register_hook(make_shared_hook(Uv, Vv)))
        for head in self.heads:
            handles.append(head.Wq_u.register_hook(make_unique_hook(Uq, Vq)))
            handles.append(head.Wk_u.register_hook(make_unique_hook(Uk, Vk)))
            handles.append(head.Wv_u.register_hook(make_unique_hook(Uv, Vv)))
        self._hook_handles = handles
        self._Uq, self._Vq = Uq, Vq
        self._Uk, self._Vk = Uk, Vk
        self._Uv, self._Vv = Uv, Vv

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
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
        self.sa = SpectralMultiheadAttention(n_head, head_size)
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
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed) #final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size) 

    def forward(self, idx, targets=None):
        b,t = idx.shape
        #idx and targets are both (b,t) tensor of integers
        token_embed = self.token_embedding_table(idx) #(b,t,c)
        pos_embed = self.position_embedding_table(torch.arange(t, device=device)) #also (b,t,c)
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

# Collect all SpectralMultiheadAttention modules for SVD refresh scheduling
spectral_attn_modules = [
    module for module in m.modules()
    if isinstance(module, SpectralMultiheadAttention)
]

for iter in range(max_iters):

    #every once in awhile evaluate the loss on train and val sets
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

    # SD-MoE: refresh shared SVD bases every svd_refresh_interval steps
    if (iter + 1) % svd_refresh_interval == 0:
        for sa in spectral_attn_modules:
            sa.refresh_svd()



context = idx=torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))