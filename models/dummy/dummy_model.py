from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
import random

CLASSES = ['fine_concrete', 'concrete', 'soft_tiles', 'tiled', 'soft_pvc',
           'hard_tiles_large_space', 'carpet', 'hard_tiles', 'wood']

SUBMISSION_COLS = ['series_id', 'surface']

class DummyClassifier(BaseEstimator):
    def __init__(self, classes = CLASSES):
        self.classes = classes

    def fit(self, x, y):
        return self

    def predict(self, x):
        n = len(x)
        preds = []
        for i in range(n):
            preds.append([i, random.choice(self.classes)])
        preds = pd.DataFrame(preds, columns=SUBMISSION_COLS)
        return preds
