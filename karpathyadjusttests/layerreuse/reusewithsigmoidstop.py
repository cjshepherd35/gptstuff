import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Bernoulli
# from torch.amp import GradScaler, autocast
from datasets import load_dataset
import sys
sys.stdout.reconfigure(encoding="utf-8")

textraw = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
sample = textraw['train'].select(range(400_000))
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
max_iters = 40_001
eval_iters = 100
eval_interval =  4000  #500
n_embed = 128   #64
block_size = 64
batch_size = 16
learning_rate = 3e-4 
n_head = 8  #4
n_layer = 4  #6
dropout = 0.2
max_recursion_depth = 8

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
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class RLSigstopTransformer(nn.Module):
    def __init__(self, max_loops=12, step_penalty=0.01):
        super().__init__()
        # --- Model Parameters ---
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

        # --- RL / Dynamic Loop Parameters ---
        self.max_loops = max_loops
        self.step_penalty = step_penalty # Cost for each computation step
        
        # This module is now our "policy network"
        self.policy_network = nn.Sequential(
            nn.Linear(2 * n_embed, 1),
            nn.Sigmoid()
        )

    def forward(self, idx, targets=None):
        b, t = idx.shape
        token_embed = self.token_embedding_table(idx)
        pos_embed = self.position_embedding_table(torch.arange(t, device=device))
        
        initial_x = token_embed + pos_embed
        x = initial_x

        log_probs = [] # To store log probabilities of actions taken
        loop_count = 0
        
        while loop_count < self.max_loops:
            # Get the probability of stopping from our policy network
            # The input is the "state": initial embedding + current embedding
            sigmoid_input = torch.cat((initial_x, x), dim=-1)
            stop_prob = self.policy_network(sigmoid_input)
            
            # Create a Bernoulli distribution to sample an action (0=continue, 1=stop)
            # We average the probability across the batch/sequence for a single decision
            dist = Bernoulli(stop_prob.mean())
            action = dist.sample() # Returns a tensor(0.) or tensor(1.)
            
            # Store the log probability of the action we just took. This is crucial for REINFORCE.
            log_probs.append(dist.log_prob(action))

            if action.item() == 1.0: # If we sampled "stop"
                break
            
            # If we sampled "continue", we perform another processing step
            x = self.blocks(x)
            x = self.ln_f(x)
            loop_count += 1
        
        # The remainder of the forward pass
        logits = self.lm_head(x)
        
        # If we are not training (e.g., during inference), we are done.
        if targets is None:
            return logits, None

        b, t, c = logits.shape
        task_loss = F.cross_entropy(logits.view(b*t, c), targets.view(b*t))

        num_steps = len(log_probs)
        reward = -task_loss - (num_steps * self.step_penalty)

        policy_loss = -torch.stack(log_probs).sum() * reward.detach()

        # 4. Combine the losses
        total_loss = task_loss + (policy_loss/2)
            
        return logits, total_loss
    
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


model = RLSigstopTransformer(max_loops=max_recursion_depth)
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