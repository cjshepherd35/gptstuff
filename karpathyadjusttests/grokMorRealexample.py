import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchtext
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import math

class ExpertChoiceRouter(nn.Module):
    def __init__(self, hidden_size, max_recursion_depth=3, beta_percentile=66.0):
        """
        Initialize the Expert-Choice router for Mixture-of-Recursions.
        
        Args:
            hidden_size (int): Dimension of token embeddings.
            max_recursion_depth (int): Maximum number of recursion iterations (e.g., 3).
            beta_percentile (float): Percentile threshold for top-k selection (e.g., 66 for top-1/3).
        """
        super(ExpertChoiceRouter, self).__init__()
        self.hidden_size = hidden_size
        self.max_depth = max_recursion_depth
        self.beta = beta_percentile / 100.0
        
        self.routing_params = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_size)) for _ in range(max_recursion_depth)
        ])
        self.activation = nn.Sigmoid()

    def compute_balancing_loss(self, scores_list):
        """
        Compute load balancing loss to ensure even distribution across depths.
        
        Args:
            scores_list (list of torch.Tensor): Scores for each recursion step.
            
        Returns:
            torch.Tensor: Scalar balancing loss.
        """
        scores = torch.stack(scores_list, dim=-1)
        probs = self.activation(scores).mean(dim=(0, 1))
        target_prob = torch.ones_like(probs) / self.max_depth
        return F.kl_div(probs.log(), target_prob, reduction='batchmean')

    def forward(self, hidden_states):
        """
        Route tokens to recursion depths using expert-choice routing.
        
        Args:
            hidden_states (torch.Tensor): Input embeddings of shape [batch_size, seq_len, hidden_size].
            
        Returns:
            torch.Tensor: Depth assignments of shape [batch_size, seq_len].
            torch.Tensor: Balancing loss for training.
            list of torch.Tensor: Masks for each recursion step.
        """
        batch_size, seq_len, _ = hidden_states.size()
        device = hidden_states.device
        
        depth_assignments = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        active_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        scores_list = []
        
        for r in range(self.max_depth):
            scores = torch.einsum('bsh,h->bs', hidden_states, self.routing_params[r])
            scores = self.activation(scores)
            scores_list.append(scores)
            scores = scores.masked_fill(~active_mask, float('-inf'))
            
            k = max(1, int(seq_len * (1.0 / self.max_depth)))
            _, top_k_indices = torch.topk(scores, k, dim=1)
            
            for b in range(batch_size):
                depth_assignments[b, top_k_indices[b]] = r + 1
            
            step_mask = torch.zeros_like(active_mask, dtype=torch.bool)
            for b in range(batch_size):
                step_mask[b, top_k_indices[b]] = True
            active_mask = active_mask & step_mask
            
            if not active_mask.any():
                break
        
        depth_assignments[depth_assignments == 0] = 1
        balancing_loss = self.compute_balancing_loss(scores_list)
        return depth_assignments, balancing_loss, [torch.ones_like(active_mask) if r == 0 else active_mask for r in range(self.max_depth)]

class MoRTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_recursion_depth=3):
        """
        Transformer with MoR routing for language modeling on WikiText-2.
        
        Args:
            vocab_size (int): Size of the vocabulary.
            hidden_size (int): Dimension of token embeddings.
            max_recursion_depth (int): Maximum recursion depth.
        """
        super(MoRTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.max_depth = max_recursion_depth
        
        # Token and positional embeddings
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Parameter(torch.randn(1, 128, hidden_size))  # Fixed seq_len=128
        self.dropout = nn.Dropout(0.1)
        
        # Shared transformer block
        self.transformer_block = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=8, dim_feedforward=hidden_size * 4
        )
        self.router = ExpertChoiceRouter(hidden_size, max_recursion_depth)
        
        # Language modeling head
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, targets=None):
        """
        Process text tokens with dynamic recursion depths.
        
        Args:
            input_ids (torch.Tensor): Input token IDs of shape [batch_size, seq_len].
            targets (torch.Tensor, optional): Target token IDs for loss calculation.
            
        Returns:
            torch.Tensor: Logits of shape [batch_size, seq_len, vocab_size].
            torch.Tensor: Total loss (LM loss + balancing loss) if targets provided, else None.
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Embed tokens
        x = self.token_embed(input_ids) + self.pos_embed[:, :seq_len, :]
        x = self.dropout(x)
        
        # Route tokens
        depth_assignments, balancing_loss, step_masks = self.router(x)
        
        # Apply transformer block
        output = x.clone()
        for depth in range(1, self.max_depth + 1):
            mask = (depth_assignments == depth) & step_masks[depth - 1]
            if mask.sum() == 0:
                continue
            active_tokens = output[mask].reshape(-1, self.hidden_size)
            active_output = self.transformer_block(active_tokens)
            output[mask] = active_output.reshape(-1, self.hidden_size)
        
        # Compute logits
        logits = self.lm_head(output)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
            loss = lm_loss + 0.1 * balancing_loss  # Weight balancing loss
        
        return logits, loss

def get_wikitext2_data(seq_len=128, batch_size=32):
    """
    Load and preprocess WikiText-2 dataset.
    
    Args:
        seq_len (int): Sequence length for training.
        batch_size (int): Batch size for DataLoader.
        
    Returns:
        DataLoader: Training DataLoader.
        DataLoader: Validation DataLoader.
        torchtext.vocab.Vocab: Vocabulary object.
    """
    # Tokenizer and vocabulary
    tokenizer = get_tokenizer('basic_english')
    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenizer(text)
    
    train_iter = WikiText2(split='train')
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<pad>', '<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    
    # Process dataset into sequences
    def process_data(split, seq_len):
        data_iter = WikiText2(split=split)
        data = []
        for text in data_iter:
            tokens = vocab(tokenizer(text))
            for i in range(0, len(tokens) - seq_len, seq_len):
                data.append(tokens[i:i + seq_len + 1])  # Include target token
        return torch.tensor(data, dtype=torch.long)
    
    train_data = process_data('train', seq_len)
    valid_data = process_data('valid', seq_len)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, vocab

def train_model(model, train_loader, valid_loader, epochs=10, device='cuda'):
    """
    Train and evaluate the MoRTransformer on WikiText-2.
    
    Args:
        model: MoRTransformer model.
        train_loader: DataLoader for training data.
        valid_loader: DataLoader for validation data.
        epochs: Number of training epochs.
        device: Device to run on.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids, targets = batch[:, :-1].to(device), batch[:, 1:].to(device)
            
            optimizer.zero_grad()
            _, loss = model(input_ids, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Perplexity: {math.exp(train_loss):.2f}')
        
        # Evaluate
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                input_ids, targets = batch[:, :-1].to(device), batch[:, 1:].to(device)
                _, loss = model(input_ids, targets)
                valid_loss += loss.item()
        
        valid_loss /= len(valid_loader)
        print(f'Validation Loss: {valid_loss:.4f}, Validation Perplexity: {math.exp(valid_loss):.2f}')

def main():
    # Hyperparameters
    hidden_size = 256
    max_depth = 3
    seq_len = 128
    batch_size = 32
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load WikiText-2
    train_loader, valid_loader, vocab = get_wikitext2_data(seq_len=seq_len, batch_size=batch_size)
    
    # Initialize model
    model = MoRTransformer(vocab_size=len(vocab), hidden_size=hidden_size, max_recursion_depth=max_depth)
    
    # Train and evaluate
    train_model(model, train_loader, valid_loader, epochs=epochs, device=device)

if __name__ == "__main__":
    main()