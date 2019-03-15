import pandas as pd
import torch
import numpy as np 
from os.path import join
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder



X_TRAIN = 'X_train.csv'
X_TEST = 'X_test.csv'
Y_TRAIN = 'y_train.csv'

FEATURE_COLS = ['orientation_X',
                'orientation_Y', 'orientation_Z', 'orientation_W', 'angular_velocity_X',
                'angular_velocity_Y', 'angular_velocity_Z', 'linear_acceleration_X',
                'linear_acceleration_Y', 'linear_acceleration_Z']


def x_to_array(df, feature_cols=FEATURE_COLS):
    # Convert tabular data into 3D numpy array
    n_series = len(df.series_id.unique())
    n_features = len(feature_cols)
    n_samples = max(df.measurement_number+1)
    x_trans = np.empty((n_series, n_samples, n_features))
    for i, idee in enumerate(df.series_id.unique()):
        x_trans[i] = np.array(df[df.series_id == idee][feature_cols])
    return x_trans
    

def load_and_format_data(path):
    x_train = pd.read_csv(join(path,X_TRAIN))
    x_test = pd.read_csv(join(path,X_TEST))
    y_train = pd.read_csv(join(path,Y_TRAIN))

    x_test = x_to_array(x_test)
    x_train = x_to_array(x_train)
    y_train = y_train.surface # for now just return unencoded values, let Tensorflow take care of this

    return x_train, y_train, x_test


class RobotNavDataset(Dataset):
    # TODO add transformations here
    # TODO add option to one hot labels
    def __init__(self, path_dir):
        self.train, self.labels, self.test = self.load_and_format_data(path_dir)
        
    def load_and_format_data(path):
        x_train = pd.read_csv(join(path,X_TRAIN))
        x_test = pd.read_csv(join(path,X_TEST))
        y_train = pd.read_csv(join(path,Y_TRAIN))

        x_test = x_to_array(x_test)
        x_train = x_to_array(x_train)
        y_train = y_train.surface # for now just return unencoded values, let Tensorflow take care of this
        y_train = LabelEncoder().fit_transform(y_train)

        return torch.tensor(x_train), torch.tensor(y_train), torch.tensor(x_test)
        