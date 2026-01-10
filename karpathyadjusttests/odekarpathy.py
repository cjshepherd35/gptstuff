import torch
import torch.nn as nn
from torch.nn import functional as F
from torchdiffeq import odeint, odeint_adjoint
device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print('device is: ', device)
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(vocab_size)
#parameters to tweak
max_iters = 3001
eval_iters = 100
eval_interval =  500  #500
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
    
class ODEfunc(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiheadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        
    def forward(self, x):
        sa_term = self.sa(self.ln1(x))
        ff_term = self.ffwd(self.ln2(x))
        return sa_term + ff_term


class ODEBlock(nn.Module):
    """
    Wraps an ODEFunc and integrates from t0 -> t1.
    By default integrates from 0.0 -> 1.0.
    """
    def __init__(self, odefunc, t_span=(0.0, 1.0), solver='dopri5', rtol=1e-3, atol=1e-4, use_adjoint=False):
        super().__init__()
        self.odefunc = odefunc
        # t_span as torch tensor is created at forward-time so it's device-correct
        self.t0, self.t1 = float(t_span[0]), float(t_span[1])
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.use_adjoint = use_adjoint

    def forward(self, x):
        # ensure t is on same device/dtype as x
        t = torch.tensor([self.t0, self.t1], device=x.device, dtype=torch.float)
        # choose integration function
        integrate = odeint_adjoint if self.use_adjoint else odeint
        # odeint returns a tensor of shape (len(t), B, S, E)
        out = integrate(self.odefunc, x, t, rtol=self.rtol, atol=self.atol, method=self.solver)
        # take final timepoint
        x_t1 = out[-1]
        return x_t1


class ODETransformerBlock(nn.Module):
    """
    Drop-in replacement for Block(nn.Module) in the form:
        x -> ODE evolution of x under f(x)=sa(ln1(x)) + ffwd(ln2(x))
    Returns x_out (B, S, E)
    """
    def __init__(self, n_embed, n_head, ffwd_hidden=None, t_span=(0.0, 1.0),
                 solver='dopri5', rtol=1e-3, atol=1e-4, use_adjoint=False, sa_module=None):
        """
        n_embed: embedding dimension E
        n_head: number of heads for attention (if using built-in)
        ffwd_hidden: hidden dim inside feedforward MLP (defaults to 4*n_embed)
        sa_module: optional attention module that accepts (B,S,E) and returns (B,S,E).
                   If None, uses a simple nn.MultiheadAttention (adapts shapes).
        """
        super().__init__()
        head_size = n_embed // n_head
        if ffwd_hidden is None:
            ffwd_hidden = 4 * n_embed

        # Provide LayerNorms similar to original block
        ln1 = nn.LayerNorm(n_embed)
        ln2 = nn.LayerNorm(n_embed)

        # Provide attention module that accepts (B,S,E) -> (B,S,E)
        if sa_module is None:
            # Wrap nn.MultiheadAttention (which expects (S,B,E)) into callable
            mhatt = nn.MultiheadAttention(embed_dim=n_embed, num_heads=n_head, batch_first=False)
            def sa_fn(x):
                # x is (B, S, E) -> convert to (S, B, E)
                x_sbe = x.permute(1, 0, 2)
                # in PyTorch MHA: attn_output, _ = mhatt(query, key, value)
                out, _ = mhatt(x_sbe, x_sbe, x_sbe, need_weights=False)
                # convert back to (B, S, E)
                return out.permute(1, 0, 2)
            sa = sa_fn
        else:
            sa = sa_module

        # FeedForward module expects (B,S,E) in, (B,S,E) out
        ffwd = nn.Sequential(
            nn.Linear(n_embed, ffwd_hidden),
            nn.GELU(),
            nn.Linear(ffwd_hidden, n_embed),
        )

        

        odefunc = ODEfunc(n_embed=n_embed, n_head=n_head)
        self.odeblock = ODEBlock(odefunc, t_span=t_span, solver=solver, rtol=rtol, atol=atol, use_adjoint=use_adjoint)

    def forward(self, x, y):
        """
        x: (B, S, E)
        returns: (B, S, E)
        """
        return self.odeblock(x)
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the  last block_size tokens
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx, loss

model = ODETransformerBlock(n_embed, n_head, t_span=(0.0, 1.0), solver='dopri5', use_adjoint=False)
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