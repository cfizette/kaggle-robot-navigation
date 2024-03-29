import pandas as pd
import torch
import numpy as np 
from os.path import join
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, scale
from functools import partial

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
    x_trans = np.empty((n_series, n_features, n_samples))
    for i, idee in enumerate(df.series_id.unique()):
        x_trans[i] = np.array(df[df.series_id == idee][feature_cols]).T
    return x_trans


class RobotNavDataset(Dataset):
    # TODO add transformations here
    def __init__(self, path_dir, normalize=True):
        self.normalize = normalize
        self.label_encoder = LabelEncoder()
        #self.train, self.labels, self.test = self.load_and_format_data(path_dir)
        self.data, self.labels = self.load_and_format_data(path_dir)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        X = self.data[index]
        y = self.labels[index]
        return X, y
        
    def load_and_format_data(self, path):
        x_train = pd.read_csv(join(path,X_TRAIN))
        #x_test = pd.read_csv(join(path,X_TEST))
        y_train = pd.read_csv(join(path,Y_TRAIN))
        
        n = len(y_train)

        #x_test = x_to_array(x_test)
        x_train = x_to_array(x_train)
        
        # Normalize each channel
        if self.normalize:
            scale_trans = partial(scale, axis=1)
            # TODO: Find better/faster way to do this
            x_train = np.array([scale_trans(samp) for samp in x_train])
            #x_test = np.array([scale_trans(samp) for samp in x_test])

        y_train = self.label_encoder.fit_transform(y_train.surface)
        
        #x_train = torch.tensor(x_train).type(torch.LongTensor)
        #y_train = torch.tensor(y_train).type(torch.LongTensor)
        x_train = torch.tensor(x_train)
        y_train = torch.tensor(y_train)

        return x_train.view((n,1,10,128)), y_train 


class SubmissionDataset(Dataset):
    def __init__(self, path_dir, normalize=True):
        self.normalize = normalize
        self.data = self.load_and_format_data(path_dir)

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        X = self.train[index]
        return X

    def load_and_format_data(self, path):
        x_test = pd.read_csv(join(path,X_TEST))      
        x_test = x_to_array(x_test)
        n = len(x_test)  
        
        # Normalize each channel
        if self.normalize:
            scale_trans = partial(scale, axis=1)
            # TODO: Find better/faster way to do this
            x_test = np.array([scale_trans(samp) for samp in x_test])
            #x_test = np.array([scale_trans(samp) for samp in x_test])
        x_test = torch.tensor(x_test)
        return x_test.view((n,1,10,128))