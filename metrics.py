"""
Evaluation Metrics.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import torch

def mae_loss(y_true, y_pred, mask=None, reduce="mean"):
    '''MAE Loss.'''
    assert reduce in ["mean", "sum", "none"], f"reduce={reduce} is not defined! ('mean'|'sum'|'none')"
    
    if type(y_true) == torch.Tensor: y_true = y_true.numpy()
    if type(y_pred) == torch.Tensor: y_pred = y_pred.numpy()

    if mask is not None:
        if type(mask) == torch.Tensor: mask = mask.numpy()
        y_true, y_pred = y_true[mask==1], y_pred[mask==1]
    
    mae = np.abs(y_true - y_pred)

    if reduce=="mean": return np.mean(mae)
    elif reduce=="sum": return np.sum(mae)
    else: return mae


def mse_loss(y_true, y_pred, mask=None, reduce="mean"):
    '''MSE Loss.'''
    assert reduce in ["mean", "sum", "none"], f"reduce={reduce} is not defined! ('mean'|'sum'|'none')"
    
    if type(y_true) == torch.Tensor: y_true = y_true.numpy()
    if type(y_pred) == torch.Tensor: y_pred = y_pred.numpy()

    if mask is not None:
        if type(mask) == torch.Tensor: mask = mask.numpy()
        y_true, y_pred = y_true[mask==1], y_pred[mask==1]

    mse = (y_true - y_pred) ** 2
    
    if reduce=="mean": return np.mean(mse)
    elif reduce=="sum": return np.sum(mse)
    else: return mse.item()


def r2_loss(y_true, y_pred, mask=None):
    '''R2 Score.'''
    if type(y_true) == torch.Tensor: y_true = y_true.numpy()
    if type(y_pred) == torch.Tensor: y_pred = y_pred.numpy()

    if mask is not None:
        if type(mask) == torch.Tensor: mask = mask.numpy()
        y_true, y_pred = y_true[mask==1], y_pred[mask==1]
    
    r2 = r2_score(y_true, y_pred)
    
    return r2


def prediction_summary(y_true, y_pred, mask=None, verbose=True):
    """
    Calculate various metrics for the model's predictions.

    Parameters
    ----------
    y_true : np.ndarray or torch.Tensor.
        Groundtruth values.
    y_pred : np.ndarray or torch Tensor.
        Predicted values.
    verbose : bool.
        If printing evaluation results.
    """
    
    assert y_true.shape == y_pred.shape, "y_true and y_pred have different shapes!"
    if mask is not None:
        assert mask.shape == y_true.shape, "mask and y have different shapes!"

    mse = mse_loss(y_true, y_pred, mask)
    rmse = np.sqrt(mse_loss)
    mae = mae_loss(y_true, y_pred, mask)
    r2 = r2_loss(y_true, y_pred, mask)

    if verbose:
        print(f'MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f},')
    else:
        return mse, rmse, mae, r2


