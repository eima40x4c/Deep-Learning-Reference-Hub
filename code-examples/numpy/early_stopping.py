def early_stopping(val_losses, patience=10, min_delta=1e-4, verbose=1):
    """
    Early stopping with detailed tracking and optional verbose output.
    
    Args:
        val_losses (list): Validation losses from training history
        patience (int): Number of epochs to wait after last improvement
        min_delta (float): Minimum change required for improvement
        verbose (bool): Whether to print detailed information
    
    Returns:
        tuple: (should_stop: bool, info: dict) where info contains:
            - 'best_loss': Best validation loss seen so far
            - 'epochs_since_improvement': Number of epochs since last improvement
            - 'current_loss': Most recent validation loss
            - 'improvement_needed': Minimum loss needed to count as improvement
    """
    if len(val_losses) < 2:
        return False, {'message': 'Need at least 2 epochs to evaluate'}
    
    best_loss = min(val_losses)
    best_epoch = val_losses.index(best_loss)
    current_epoch = len(val_losses) - 1
    epochs_since_improvement = current_epoch - best_epoch
    current_loss = val_losses[-1]
    improvement_needed = best_loss - min_delta
    
    info = {
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'current_loss': current_loss,
        'improvement_needed': improvement_needed,
        'patience_remaining': max(0, patience - epochs_since_improvement)
    }
    
    should_stop = epochs_since_improvement >= patience
    
    if verbose:
        print(f"Early Stopping Check:")
        print(f"  Current Loss: {current_loss:.6f}")
        print(f"  Best Loss: {best_loss:.6f} (epoch {best_epoch})")
        print(f"  Epochs since improvement: {epochs_since_improvement}")
        print(f"  Patience remaining: {info['patience_remaining']}")
        print(f"  Should stop: {should_stop}")
    
    return should_stop, info