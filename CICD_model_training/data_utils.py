import numpy as np
import tensorflow as tf

def preprocess(data_array: np.ndarray, train_size: float, val_size: float):
    num_time_steps = data_array.shape[0]
    num_train = int(num_time_steps * train_size)
    num_val = int(num_time_steps * val_size)
    train_array = data_array[:num_train]
    mean, std = train_array.mean(axis=0), train_array.std(axis=0)
    train_array = (train_array - mean) / std
    val_array = (data_array[num_train:num_train+num_val] - mean) / std
    test_array = (data_array[(num_train+num_val):] - mean) / std
    return train_array, val_array, test_array

def create_tf_dataset(
    data_array: np.ndarray,
    input_sequence_length: int,
    forecast_horizon: int,
    batch_size: int = 128,
    shuffle=True,
    multi_horizon=True,
):
    inputs = tf.keras.utils.timeseries_dataset_from_array(
        np.expand_dims(data_array[:-forecast_horizon], axis=-1),
        None,
        sequence_length=input_sequence_length,
        shuffle=False,
        batch_size=batch_size,
    )
    target_offset = (
        input_sequence_length if multi_horizon else input_sequence_length + forecast_horizon - 1
    )
    target_seq_length = forecast_horizon if multi_horizon else 1
    targets = tf.keras.utils.timeseries_dataset_from_array(
        data_array[target_offset:],
        None,
        sequence_length=target_seq_length,
        shuffle=False,
        batch_size=batch_size,
    )
    dataset = tf.data.Dataset.zip((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(100)
    return dataset.prefetch(16).cache()

def compute_adjacency_matrix(route_distances: np.ndarray, sigma2: float, epsilon: float):
    num_routes = route_distances.shape[0]
    route_distances = route_distances / 10000.0
    w2, w_mask = route_distances * route_distances, np.ones([num_routes, num_routes]) - np.identity(num_routes)
    return (np.exp(-w2 / sigma2) >= epsilon) * w_mask

class GraphInfo:
    def __init__(self, edges, num_nodes):
        self.edges = edges
        self.num_nodes = num_nodes
