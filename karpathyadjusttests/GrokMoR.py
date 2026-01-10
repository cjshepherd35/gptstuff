import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertChoiceRouter(nn.Module):
    def __init__(self, hidden_size, max_recursion_depth=3, beta_percentile=66.0):
        """
        Initialize the Expert-Choice router for Mixture-of-Recursions.
        
        Args:
            hidden_size (int): Dimension of token embeddings (transformer hidden size).
            max_recursion_depth (int): Maximum number of recursion iterations (e.g., 3).
            beta_percentile (float): Percentile threshold for top-k selection (e.g., 66 for top-1/3).
        """
        super(ExpertChoiceRouter, self).__init__()
        self.hidden_size = hidden_size
        self.max_depth = max_recursion_depth
        self.beta = beta_percentile / 100.0
        
        # Routing parameters for each recursion step
        self.routing_params = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_size)) for _ in range(max_recursion_depth)
        ])
        
        self.activation = nn.Sigmoid()  # G function, as per paper

    def compute_balancing_loss(self, scores_list):
        """
        Compute load balancing loss to ensure even distribution across depths.
        
        Args:
            scores_list (list of torch.Tensor): Scores for each recursion step.
            
        Returns:
            torch.Tensor: Scalar balancing loss.
        """
        # Concatenate scores across depths
        scores = torch.stack(scores_list, dim=-1)  # Shape: [batch_size, seq_len, max_depth]
        probs = self.activation(scores).mean(dim=(0, 1))  # Average probability per depth
        target_prob = torch.ones_like(probs) / self.max_depth  # Uniform target
        return F.kl_div(probs.log(), target_prob, reduction='batchmean')

    def forward(self, hidden_states):
        """
        Route tokens to recursion depths using expert-choice routing with hierarchical filtering.
        
        Args:
            hidden_states (torch.Tensor): Input embeddings of shape [batch_size, seq_len, hidden_size].
            
        Returns:
            torch.Tensor: Depth assignments of shape [batch_size, seq_len].
            torch.Tensor: Balancing loss for training.
            list of torch.Tensor: Masks for each recursion step.
        """
        batch_size, seq_len, _ = hidden_states.size()
        device = hidden_states.device
        
        # Initialize depth assignments and masks
        depth_assignments = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        active_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)  # All tokens start active
        scores_list = []
        
        for r in range(self.max_depth):
            # Compute scores: g_r^t = G(theta_r^T * H_r^t)
            theta_r = self.routing_params[r]  # Shape: [hidden_size]
            scores = torch.einsum('bsh,h->bs', hidden_states, theta_r)  # Shape: [batch_size, seq_len]
            scores = self.activation(scores)  # Apply sigmoid
            scores_list.append(scores)
            
            # Apply hierarchical filtering: only active tokens are scored
            scores = scores.masked_fill(~active_mask, float('-inf'))
            
            # Compute beta-percentile threshold
            k = max(1, int(seq_len * (1.0 / self.max_depth)))  # Top-k per expert
            _, top_k_indices = torch.topk(scores, k, dim=1)  # Shape: [batch_size, k]
            
            # Update depth assignments
            for b in range(batch_size):
                # Tokens selected at this step get depth r+1
                depth_assignments[b, top_k_indices[b]] = r + 1
            
            # Update active mask for hierarchical filtering (only selected tokens continue)
            step_mask = torch.zeros_like(active_mask, dtype=torch.bool)
            for b in range(batch_size):
                step_mask[b, top_k_indices[b]] = True
            active_mask = active_mask & step_mask  # Only selected tokens remain active
            
            # Early stop if no tokens remain
            if not active_mask.any():
                break
        
        # Assign depth 1 to any unassigned tokens (ensures all tokens processed at least once)
        depth_assignments[depth_assignments == 0] = 1
        
        # Compute balancing loss
        balancing_loss = self.compute_balancing_loss(scores_list)
        
        return depth_assignments, balancing_loss, [torch.ones_like(active_mask) if r == 0 else active_mask for r in range(self.max_depth)]

class MoRTransformer(nn.Module):
    def __init__(self, hidden_size, max_recursion_depth=3):
        """
        Simplified transformer with expert-choice MoR routing.
        
        Args:
            hidden_size (int): Dimension of token embeddings.
            max_recursion_depth (int): Maximum recursion depth.
        """
        super(MoRTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.max_depth = max_recursion_depth
        
        # Shared transformer block (simplified)
        self.transformer_block = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=8, dim_feedforward=hidden_size * 4
        )
        self.router = ExpertChoiceRouter(hidden_size, max_recursion_depth)
        
    def forward(self, x):
        """
        Process tokens with dynamic recursion depths.
        
        Args:
            x (torch.Tensor): Input embeddings of shape [batch_size, seq_len, hidden_size].
            
        Returns:
            torch.Tensor: Processed embeddings of shape [batch_size, seq_len, hidden_size].
            torch.Tensor: Balancing loss for training.
        """
        batch_size, seq_len, _ = x.size()
        output = x.clone()
        depth_assignments, balancing_loss, step_masks = self.router(x)
        
        # Apply transformer block according to assigned depths
        for depth in range(1, self.max_depth + 1):
            # Mask for tokens requiring this depth
            mask = (depth_assignments == depth) & step_masks[depth - 1]
            if mask.sum() == 0:
                continue
                
            # Apply transformer block to selected tokens
            active_tokens = output[mask].reshape(-1, self.hidden_size)
            active_output = self.transformer_block(active_tokens)
            
            # Update output for active tokens
            output[mask] = active_output.reshape(-1, self.hidden_size)
        
        return output, balancing_loss

# Example usage
def main():
    hidden_size = 512
    seq_len = 128
    batch_size = 4
    max_depth = 3
    
    # Initialize model
    model = MoRTransformer(hidden_size=hidden_size, max_recursion_depth=max_depth)
    model.eval()
    
    # Dummy input
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # Forward pass
    output, balancing_loss = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Balancing loss: {balancing_loss.item()}")

if __name__ == "__main__":
    main()