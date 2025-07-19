"""
Early Stopping Utility
======================

Implements a mechanism to halt model training when a monitored validation metric
(such as loss or accuracy) ceases to improve after a specified number of epochs.

This prevents overfitting and saves compute by terminating training once performance plateaus.

References
----------
- Prechelt, L. (2012). Early Stopping — But When? In *Neural Networks: Tricks of the Trade*. Springer.
  https://link.springer.com/chapter/10.1007/978-3-642-35289-8_5

Author
------
Deep Learning Reference Hub

License
-------
MIT License

Notes
-----
- Works by monitoring "no improvement" for `patience` consecutive epochs.
"""

from typing import Tuple


def early_stopping(
    val_losses: list, patience: int = 10, min_delta: float = 1e-4, verbose: bool = 1
) -> Tuple[bool, dict]:
    """
    Early stopping with detailed tracking and optional verbose output.

    Parameters
    ----------
    val_losses : list
        Validation losses from training history
    patience : int, defualt=10
        Number of epochs to wait after last improvement
    min_delta : float, default=1e-4
        Minimum change required for improvement
    verbose : bool, default=1
        Whether to print detailed information

    Returns
    -------
        Tuple[bool, Dict]
            whether to stop and info dict, which contains:
            - 'best_loss': Best validation loss seen so far
            - 'epochs_since_improvement': Number of epochs since last improvement
            - 'current_loss': Most recent validation loss
            - 'improvement_needed': Minimum loss needed to count as improvement
    """
    if len(val_losses) < 2:
        return False, {"message": "Need at least 2 epochs to evaluate"}

    best_loss = min(val_losses)
    best_epoch = val_losses.index(best_loss)
    current_epoch = len(val_losses) - 1
    epochs_since_improvement = current_epoch - best_epoch
    current_loss = val_losses[-1]
    improvement_needed = best_loss - min_delta

    info = {
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "epochs_since_improvement": epochs_since_improvement,
        "current_loss": current_loss,
        "improvement_needed": improvement_needed,
        "patience_remaining": max(0, patience - epochs_since_improvement),
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
