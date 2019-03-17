import torch 
from os.path import join
import pandas as pd 
import numpy as np 
from time import time

SUBMISSION_COLS = ['series_id', 'surface']


def format_submision(preds):
    # format numpy array of predictions into DataFrame
    n = len(preds)
    series_ids = list(range(n))
    data = list(zip(series_ids, preds))
    submission = pd.DataFrame(data, columns=SUBMISSION_COLS)
    return submission


def make_submission(model, data, save_file = 'submission.csv'):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1, keepdim=False)
        pred = pred.numpy()
    
    submission = format_submision(pred)
    submission.to_csv(save_file, index=False)
