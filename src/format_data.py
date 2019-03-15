import pandas as pd
import numpy as np 

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
    