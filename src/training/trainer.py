"""PyTorch model trainer with validation early stopping and checkpointing."""

import copy
import logging
from typing import Any, Callable, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip_norm: float = 5.0,
    task: str = "scalar"  # "scalar" or "tensor"
) -> float:
    """Trains model for one epoch."""
    model.train()
    total_loss = 0.0
    total_items = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        if task == "scalar":
            out = model(batch)
            loss = criterion(out, batch.y_scalar.squeeze(-1))
            num_items = batch.num_graphs
        elif task == "tensor":
            _, V_pred = model(batch)
            loss = criterion(V_pred, batch.y_tensor)
            num_items = batch.num_nodes
        else:
            raise ValueError(f"Unknown task: {task}")

        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += loss.item() * num_items
        total_items += num_items

    return total_loss / max(1, total_items)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task: str = "scalar"
) -> float:
    """Evaluates model on validation data."""
    model.eval()
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if task == "scalar":
                out = model(batch)
                loss = criterion(out, batch.y_scalar.squeeze(-1))
                num_items = batch.num_graphs
            elif task == "tensor":
                _, V_pred = model(batch)
                loss = criterion(V_pred, batch.y_tensor)
                num_items = batch.num_nodes

            total_loss += loss.item() * num_items
            total_items += num_items

    return total_loss / max(1, total_items)


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int = 50,
    patience: int = 12,
    grad_clip_norm: float = 5.0,
    scheduler: Optional[Any] = None,
    task: str = "scalar",
    logger: Optional[logging.Logger] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Fits model with early stopping on validation loss."""
    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, grad_clip_norm, task)
        val_loss = evaluate(model, val_loader, criterion, device, task)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if logger:
            logger.info(f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} (Best: {best_val_loss:.4f})")

        if patience_counter >= patience:
            if logger:
                logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    model.load_state_dict(best_weights)
    return model, {"history": history, "best_val_loss": best_val_loss}
