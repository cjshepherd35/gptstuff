'''this  does not seem to work, makes it bigger and worse performance
size of model 1170168
step 0: train loss 7.5619, val loss 7.5607
step 500: train loss 2.6173, val loss 2.6195
step 1000: train loss 2.4943, val loss 2.4963
step 1500: train loss 2.4334, val loss 2.4461
step 2000: train loss 2.3703, val loss 2.3762
step 2500: train loss 2.2919, val loss 2.3013
step 3000: train loss 2.2418, val loss 2.2465
step 3500: train loss 2.1860, val loss 2.2032
step 4000: train loss 2.1456, val loss 2.1461
step 4500: train loss 2.1072, val loss 2.1045'''
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
max_iters = 10_001
eval_iters = 100
eval_interval =  500
n_embed = 64   #64
block_size = 64
batch_size = 16
learning_rate = 3e-4
n_head = 4  #4
n_layer = 3  #6
dropout = 0.2
num_experts = 8 # Number of experts in the MoE layer
top_k = 2 # Number of experts to route each token to

# --- NEW MoE Hyperparameters for Attention Projections ---
num_experts_attn = 4  # Number of expert linear layers for K, Q, V
top_k_attn = 2        # Number of experts to use per token


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


# --- NEW: MoELinear Module ---
# This module replaces nn.Linear with a sparse MoE layer

class MoELinear(nn.Module):
    """
    A sparse MoE layer that mimics the nn.Linear interface.
    
    It routes each token to top_k expert linear layers.
    """
    def __init__(self, n_embed_in, n_embed_out, num_experts, top_k, bias=False):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.n_embed_out = n_embed_out
        
        # Create the expert linear layers
        self.experts = nn.ModuleList([
            nn.Linear(n_embed_in, n_embed_out, bias=bias) 
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(n_embed_in, num_experts)

    def forward(self, x):
        # x shape: (b, t, c_in) where c_in is n_embed_in
        b, t, c_in = x.shape
        
        # Flatten input for token-wise routing
        x_flat = x.view(-1, c_in) # (b*t, c_in)
        
        # 1. Get routing logits from the gate
        gate_logits = self.gate(x_flat) # (b*t, num_experts)
        
        # 2. Find the top_k experts and their weights
        top_k_logits, top_k_indices = gate_logits.topk(self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1) # (b*t, top_k)
        
        # 3. Process tokens with their assigned experts
        # Initialize flat output tensor
        final_output_flat = torch.zeros(
            b * t, self.n_embed_out, 
            device=x.device, dtype=x.dtype
        )
        
        # Get flattened indices for scatter-add
        flat_token_indices = torch.arange(b*t, device=x.device).repeat_interleave(self.top_k)
        flat_expert_indices = top_k_indices.view(-1)
        
        # Group inputs by expert for batch processing
        for i in range(self.num_experts):
            # Find which tokens are routed to this expert
            token_mask = (flat_expert_indices == i)
            if token_mask.any():
                # Get the indices of the tokens for this expert
                expert_token_indices = flat_token_indices[token_mask]
                
                # Get the inputs for this expert
                expert_input = x_flat[expert_token_indices]
                
                # Process the input with the expert
                expert_output = self.experts[i](expert_input)
                
                # Get the corresponding weights
                weights_for_expert = top_k_weights.view(-1)[token_mask]
                
                # Weight the expert's output
                weighted_output = expert_output * weights_for_expert.unsqueeze(1)
                
                # Efficiently add the weighted output to the final tensor
                final_output_flat.index_add_(0, expert_token_indices, weighted_output)

        # Reshape to original (b, t, c_out)
        return final_output_flat.view(b, t, self.n_embed_out)

# --- MODIFIED: Head Class ---

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        # Replace nn.Linear with our new MoELinear
        self.key = MoELinear(
            n_embed, head_size, num_experts_attn, top_k_attn, bias=False
        )
        self.query = MoELinear(
            n_embed, head_size, num_experts_attn, top_k_attn, bias=False
        )
        self.value = MoELinear(
            n_embed, head_size, num_experts_attn, top_k_attn, bias=False
        )
        
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b,t,c = x.shape
        
        # These calls now invoke the MoELinear forward pass
        k = self.key(x)
        q = self.query(x)
        
        # compute attention scores
        wei = q @ k.transpose(-2,-1) * c**-0.5
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # perform weighted aggregation of the values
        v = self.value(x) # This also uses MoELinear
        out = wei @ v
        return out

# --- MODIFIED: MultiheadAttention Class ---
# (Now passes args to Head; also fixed a copy-paste error in forward)

class MultiheadAttention(nn.Module):
    def __init__(self,num_heads, head_size):
        super().__init__()
        # Pass the MoE args down to each Head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed) # n_embed = num_heads * head_size
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        # (Removed the duplicated lines from your original code)
        return out


class Expert(nn.Module):
    """ An expert network, which is a simple feed-forward network. """
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class MixtureOfExperts(nn.Module):
    """
    A Mixture of Experts layer.

    Args:
        n_embed (int): The embedding dimension.
        num_experts (int): The total number of expert networks.
        top_k (int): The number of experts to route each token to.
    """
    def __init__(self, n_embed, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # A list of expert networks
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        
        # The gating network is a linear layer that outputs a logit for each expert
        self.gate = nn.Linear(n_embed, num_experts)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, n_embed) -> b, t, c
        b, t, c = x.shape
        
        # Flatten the input for token-wise processing
        x_flat = x.view(-1, c) # -> (b*t, c)

        # 1. Gating: Get logits for each token and each expert
        gate_logits = self.gate(x_flat) # -> (b*t, num_experts)
        
        # 2. Routing: Find the top_k experts with the highest logits for each token
        # topk returns a tuple of (values, indices)
        top_k_logits, top_k_indices = gate_logits.topk(self.top_k, dim=-1) # -> (b*t, top_k)
        
        # 3. Normalize the weights of the selected experts using softmax
        top_k_weights = F.softmax(top_k_logits, dim=-1) # -> (b*t, top_k)
        
        # 4. Combine results: Weighted sum of expert outputs
        final_output_flat = torch.zeros_like(x_flat)
        
        # Get the indices of tokens and experts to be processed
        flat_token_indices = torch.arange(x_flat.size(0), device=x.device).repeat_interleave(self.top_k)
        flat_expert_indices = top_k_indices.view(-1)
        
        # Group inputs by expert to process them in batches
        # This is more efficient than looping through each token
        for i in range(self.num_experts):
            # Find which tokens are routed to this expert
            token_mask = (flat_expert_indices == i)
            if token_mask.any():
                # Get the indices of the tokens for the current expert
                expert_token_indices = flat_token_indices[token_mask]

                # Get the input for this expert
                expert_input = x_flat[expert_token_indices]
                
                # Process the input with the expert
                expert_output = self.experts[i](expert_input)
                
                # Get the corresponding weights
                weights_for_expert = top_k_weights.view(-1)[token_mask]
                
                # Weight the expert's output
                weighted_output = expert_output * weights_for_expert.unsqueeze(1)
                
                # Add the weighted output back to the final result tensor
                # index_add_ is an efficient in-place scatter-add operation
                final_output_flat.index_add_(0, expert_token_indices, weighted_output)

        # Reshape the output back to the original input shape
        return final_output_flat.view(b, t, c)

# --- Updated Block ---
# The Transformer Block now uses the MixtureOfExperts layer instead of the single FeedForward layer.

class Block(nn.Module):
    """ Transformer block: communication followed by computation (with MoE) """
    def __init__(self, n_embed, n_head, num_experts, top_k):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiheadAttention(n_head, head_size)
        self.moe = MixtureOfExperts(n_embed, num_experts, top_k) # <-- CHANGED
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # The architecture with residual connections remains the same
        x = x + self.sa(self.ln1(x))
        x = x + self.moe(self.ln2(x)) # <-- CHANGED
        return x

class Transformer(nn.Module):
    def __init__(self, num_experts, top_k):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # Pass MoE parameters to each block
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head, num_experts=num_experts, top_k=top_k) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        token_embed = self.token_embedding_table(idx)
        pos_embed = self.position_embedding_table(torch.arange(t, device=device))
        x = pos_embed + token_embed
        x = self.blocks(x)
        x = self.ln_f(x)
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

model = Transformer(num_experts=num_experts, top_k=top_k)
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