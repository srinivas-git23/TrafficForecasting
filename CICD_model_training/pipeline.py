from kfp import dsl
from kfp.dsl import component

@component(
    base_image="python:3.11",
    packages_to_install=["pandas", "numpy", "tensorflow", "keras"]
)
def train_stgcn_model(
    route_distances_path: str,
    speeds_array_path: str,
    model_dir: str,
):
    import subprocess
    import sys

    cmd = [
        sys.executable, "train.py",
        "--route_distances", route_distances_path,
        "--speeds_array", speeds_array_path,
        "--model_dir", model_dir,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

@dsl.pipeline(
    name="STGCN Traffic Prediction Pipeline",
    description="Trains STGCN model on traffic data and saves model to GCS"
)
def stgcn_pipeline(
    route_distances_path: str = "gs://traffic_prediction_25/PeMSD7_W_228.csv",
    speeds_array_path: str = "gs://traffic_prediction_25/PeMSD7_V_228.csv",
    model_dir: str = "gs://traffic_prediction_25/model_output/",
):
    train_task = train_stgcn_model(
        route_distances_path=route_distances_path,
        speeds_array_path=speeds_array_path,
        model_dir=model_dir,
    )

if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(stgcn_pipeline, "stgcn_pipeline.yaml")
