import numpy as np
import pandas as pd

def iou_score_total (y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    score = round(np.sum(intersection) / np.sum(union),4)
    return score

def dice_score_total (y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)        
    score = round(2 * np.sum(intersection) / (np.sum(y_true) + np.sum(y_pred)), 4)    
    return score


def iou_score_piecewise(y_true, y_pred):
    scores = []
    for yt, pred in zip(y_true, y_pred):
        intersection = np.logical_and(yt, pred)
        union = np.logical_or(yt, pred)
        scores.append(round(np.sum(intersection) / np.sum(union),4))
    return scores

def dice_score_piecewise(y_true, y_pred):
    scores = []
    for yt, pred in zip(y_true, y_pred):
        intersection = np.logical_and(yt, pred)        
        scores.append(round(2 * np.sum(intersection) / (np.sum(yt) + np.sum(pred)), 4)  )
    return scores