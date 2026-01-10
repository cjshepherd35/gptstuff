import numpy as np
import pandas as pd
import torch
from torch import nn
import os
import torch.nn.functional as F
# from torchtext import data, datasets
import tqdm
# import transformers
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
from datasets import load_dataset
import sys
import nltk
from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
from datasets import load_dataset
from gensim.models import Word2Vec
import re

embedding_dim = 64
max_iters = 10_001
eval_iters = 100
eval_interval =  500  #500
n_embed = 64   #64
block_size = 64
batch_size = 16
learning_rate = 3e-4
n_head = 4  #4
n_layer = 4  #6
dropout = 0.2
#vocab size is below


sys.stdout.reconfigure(encoding="utf-8")
pd.set_option('future.no_silent_downcasting', True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device is: ', device)

# Configure stdout for UTF-8 encoding
sys.stdout.reconfigure(encoding="utf-8")


# Download required NLTK data
try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")
    print("Please run: nltk.download('punkt_tab') and nltk.download('stopwords') manually.")
    sys.exit(1)
def preprocess_text(text, unique_tokens=None):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    # stop_words = set(stopwords.words('english'))
    # tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    # Add to unique tokens set if provided
    if unique_tokens is not None:
        unique_tokens.update(tokens)
    return tokens


# dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
# dataset = dataset['train'].select(range(50))

def load_wikitext_hf():
    # Load WikiText-103-raw-v1 dataset
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    #if changing this number then delete old model......... 
    dataset = dataset['train'].select(range(10_000))
    #.....................................................
    sentences = []
    unique_tokens = set()
    # Process the 'train' split (or modify for 'validation'/'test' if needed)
    for example in dataset:
        text = example['text']
        if text.strip() and not text.startswith(' ='):
            tokens = preprocess_text(text, unique_tokens)
            if tokens:
                sentences.append(tokens)
    return sentences, unique_tokens

def load_or_train_word2vec(sentences, model_path="karpathyadjusttests/wikitext_word2vec.model", vector_size=embedding_dim, window=5, min_count=1, workers=4):
    # Check if model exists
    if os.path.exists(model_path):
        print(f"Loading existing Word2Vec model from {model_path}...")
        try:
            model = Word2Vec.load(model_path)
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training a new model instead...")
    
    # Train new model if none exists or loading failed
    print("Training Word2Vec model...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers
    )
    model.save(model_path)
    print(f"Model saved as '{model_path}'")
    return model


# sentences, unique_tokens = load_wikitext_hf()
# print(sentences[:5])

model_path = "karpathyadjusttests/wikitext_word2vec.model"
sentences, unique_tokens = load_wikitext_hf()

# Load or train the model
model = load_or_train_word2vec(sentences, model_path)
    
vocab_size = len(unique_tokens)


sample_word = "unrecorded"
# if sample_word in model.wv:
#     print(f"Embedding for '{sample_word}':")
#     print(model.wv[sample_word])
# else:
#     print(f"Word '{sample_word}' not in vocabulary.")
textlist = []
wordvecs = []
for sample in sentences:
    textlist.extend(sample)
# print(model.wv['unrecorded'])
i = 0
for word in textlist:
    if word in model.wv:  # Check if word is in vocabulary
        vector = model.wv[word]  # Get the vector
        wordvecs.append(vector)
    else:
        # print(f"Word '{word}' not in vocabulary")
        i += 1
        # Optionally, append a zero vector or handle differently
        wordvecs.append(np.zeros(model.vector_size))  # Zero vector for unknown words
print('i; ', i)
print(len(wordvecs[0]))
# worddf = pd.DataFrame(wordvecs)
data = np.array(wordvecs)
data = torch.tensor(data, dtype=torch.float32)



train_data = data[:400_000]
test_data = data[400_000:]
# print(train_data.shape)







def get_batch(split):
    #generate a small batch of data of inputs x and y
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for  i in ix])
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

class Transformer(nn.Module):

    def __init__(self):
        super().__init__()
        #each token reads off the logits for the next tokenfrom a lookup table
        # self.token_embedding_table = nn.Embedding(vocab_size, n_embed) 
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed) #final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size) 

    def forward(self, idx, targets=None):
        b,t, c = idx.shape
        #idx and targets are both (b,t) tensor of integers
        # token_embed = self.token_embedding_table(idx) #(b,t,c)
        pos_embed = self.position_embedding_table(torch.arange(t, device=device)) #also (b,t,c)
        x = pos_embed + idx
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            b,t,c = logits.shape
            logits = logits.view(b*t,c)
            targets = targets.view(b*t,64)
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


# context = idx=torch.zeros((1,1), dtype=torch.long, device=device)
# print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))