import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from data_utils import preprocess, create_tf_dataset, compute_adjacency_matrix, GraphInfo
from model_layers import LSTMGC

SAMPLE_ROUTES = [
    0, 1, 4, 7, 8, 11, 15, 108, 109, 114, 115, 118, 120, 123,
    124, 126, 127, 129, 130, 132, 133, 136, 139, 144, 147, 216
]

def main(args):
    route_distances = pd.read_csv(args.route_distances).to_numpy()
    speeds_array = pd.read_csv(args.speeds_array).to_numpy()
    route_distances = route_distances[np.ix_(SAMPLE_ROUTES, SAMPLE_ROUTES)]
    speeds_array = speeds_array[:, SAMPLE_ROUTES]

    train_array, val_array, test_array = preprocess(speeds_array, args.train_size, args.val_size)
    train_dataset = create_tf_dataset(train_array, args.input_seq_len, args.forecast_horizon, args.batch_size)
    val_dataset = create_tf_dataset(val_array, args.input_seq_len, args.forecast_horizon, args.batch_size)
    adj = compute_adjacency_matrix(route_distances, args.sigma2, args.epsilon)
    ni, nj = np.where(adj == 1)
    graph = GraphInfo(edges=(ni.tolist(), nj.tolist()), num_nodes=adj.shape[0])
    from keras import layers, models, optimizers, losses, callbacks
    st_gcn = LSTMGC(
        in_feat=1,
        out_feat=args.out_feat,
        lstm_units=args.lstm_units,
        input_seq_len=args.input_seq_len,
        output_seq_len=args.forecast_horizon,
        graph_info=graph,
        graph_conv_params={
            "aggregation_type": args.aggregation_type,
            "combination_type": args.combination_type,
            "activation": args.gcn_activation,
        },
    )
    inputs = layers.Input((args.input_seq_len, graph.num_nodes, 1))
    outputs = st_gcn(inputs)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.RMSprop(learning_rate=args.learning_rate),
        loss=losses.MeanSquaredError(),
    )
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=[callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    )
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "stgcn_model.keras")
    model.export(model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--route_distances", required=True)
    parser.add_argument("--speeds_array", required=True)
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--input_seq_len", type=int, default=12)
    parser.add_argument("--forecast_horizon", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sigma2", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--out_feat", type=int, default=8)
    parser.add_argument("--lstm_units", type=int, default=32)
    parser.add_argument("--aggregation_type", default="mean")
    parser.add_argument("--combination_type", default="concat")
    parser.add_argument("--gcn_activation", default=None)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--model_dir", required=True, help="GCS path (gs://...) or local dir to save the trained model")
    args = parser.parse_args()
    main(args)
