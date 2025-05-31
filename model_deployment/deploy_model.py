from google.cloud import aiplatform

PROJECT = "canvas-joy-456715-b1"
REGION = "us-east1"
MODEL_PATH = "gs://traffic_prediction_25/model_output/stgcn_model.keras/"  # The SavedModel directory you want to deploy

aiplatform.init(
    project=PROJECT,
    location=REGION,
    staging_bucket="gs://traffic_prediction_25"
)

# Upload the model from GCS
model = aiplatform.Model.upload(
    display_name="traffic-lstm-graphcnn-model",
    artifact_uri=MODEL_PATH,
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-11:latest",
    sync=True
)

# Deploy to a new or existing endpoint
endpoint = model.deploy(
    deployed_model_display_name="traffic-prediction-endpoint",
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=1,
    traffic_split={"0": 100}
)

print(f"Model deployed at endpoint: {endpoint.resource_name}")
print(f"Endpoint URL: {endpoint._gca_resource.name}")
