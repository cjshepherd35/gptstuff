import torch

n_heads = 4
seq_len = 6

def get_retnet_gammas(n_heads):
    # The heads have different decay rates based on the formula:
    # gamma = 1 - 2^(-5 - i)  where i is the head index
    
    # Create index vector [0, 1, ..., n_heads-1]
    indices = torch.arange(0, n_heads).float()
    
    # Apply the formula
    gammas = 1 - 2 ** (-5 - indices)
    
    return gammas.tolist()

def create_retention_mask(seq_len, gamma, device='cpu'):
    
    # 1. Create indices for rows (n) and columns (m)
    # n: shape [seq_len, 1] -> [[0], [1], [2], ...]
    n = torch.arange(seq_len, device=device).unsqueeze(1)
    # m: shape [1, seq_len] -> [[0, 1, 2, ...]]
    m = torch.arange(seq_len, device=device).unsqueeze(0)
    
    # 2. Compute the relative distance (n - m)
    # This broadcasts to shape [seq_len, seq_len]
    distance = n - m
    
    # 3. Create the causal mask (lower triangular)
    # We only care about positions where n >= m (past and present)
    causal_mask = distance >= 0
    
    # 4. Calculate the decay values: gamma^(n-m)
    # We initialize a matrix of zeros
    D = torch.zeros(seq_len, seq_len, device=device)
    
    # Fill the valid lower-triangular part with calculated decay values
    # Note: We only compute power for valid positions to ensure numerical stability/correctness,
    # though gamma^(negative) would just be handled by the mask zeroing it out anyway.
    D[causal_mask] = gamma ** distance[causal_mask].float()
    
    return D
gammas = get_retnet_gammas(n_heads)
# print(get_retnet_gammas(n_heads))
print(create_retention_mask(seq_len, gammas[1]))