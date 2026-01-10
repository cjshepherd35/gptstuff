#this is built from a tutorial
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

# from torchtext import data, datasets, vocab
# from torchtext import data, datasets
# import tqdm

class SelfAttention(nn.Module):
    def __init__(self, k, heads=4, mask=False):
        super().__init__()
        assert k % heads == 0
        self.k, self.heads = k, heads

        # heads
        self.tokeys    = nn.Linear(k, k, bias=False)
        self.toqueries = nn.Linear(k, k, bias=False)
        self.tovalues  = nn.Linear(k, k, bias=False)
        # This will be applied after the multi-head self-attention operation.
        self.unifyheads = nn.Linear(k, k)
    def forward(self, x):
        b, t, k = x.size()
        h = self.heads
        queries = self.toqueries(x)
        keys    = self.tokeys(x)   
        values  = self.tovalues(x)
        s = k // h
        keys    = keys.view(b, t, h, s)
        queries = queries.view(b, t, h, s)
        values  = values.view(b, t, h, s)
        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)
        # Get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        # -- dot has size (b*h, t, t) containing raw weights
        # scale the dot product
        dot = dot / (k ** (1/2))
        #mask the upper right triangle half of weights to mask future observations for gpt training.
        indices = torch.triu_indices(t, t, offset=1)
        dot[:, indices[0], indices[1]] = float('-inf')

        # normalize 
        dot = F.softmax(dot, dim=2)
        # - dot now contains row-wise normalized weights
        #C:\Users\cshep\virtenv\torchTransformer.py
        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, s)
        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)
        return self.unifyheads(out)

class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()
        self.attention = SelfAttention(k, heads=heads)
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, 4*k),
            nn.ReLU(),
            nn.Linear(4*k,k)
        )
    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        feedforward = self.ff(x)
        return self.norm2(feedforward + x)

class Transformer(nn.Module):
    def __init__(self, k, heads, depth, seq_length, num_tokens, next_seq):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, k)
        self.pos_emb = nn.Embedding(seq_length, k)

		# The sequence of transformer blocks that does all the
		# heavy lifting
        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(k=k, heads=heads))
        self.tblocks = nn.Sequential(*tblocks)

		# Maps the final output sequence to class logits
        self.toprobs = nn.Linear(k, next_seq)

    def forward(self, x):
        """
        :param x: A (b, t) tensor of integer values representing
                  words (in some predetermined vocabulary).
        :return: A (b, c) tensor of log-probabilities over the
                 classes (where c is the nr. of classes).
        """
		# generate token embeddings
        tokens = self.token_emb(x)
        b, t, k = tokens.size()
		# generate position embeddings
        positions = torch.arange(t).to('cuda:0')
        positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)

        x = tokens + positions
        x = self.tblocks(x)

        # Average-pool over the t dimension and project to class
        # probabilities
        x = self.toprobs(x.mean(dim=1))
        return F.log_softmax(x, dim=1)

