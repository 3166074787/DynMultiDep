import torch

def diff_softmax(logits, tau=1.0, hard=False, dim=-1):
    """
    Differentiable softmax function with temperature and optional hard sampling.
    
    Args:
        logits: Input logits
        tau: Temperature parameter (controls softness)
        hard: Whether to use hard sampling
        dim: Dimension along which to apply softmax
        
    Returns:
        Softmax output with optional hard sampling
    """
    y_soft = (logits / tau).softmax(dim)
    
    if hard:
        # Hard sampling with straight-through estimator
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        # Straight-through estimator
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft 