import os
from google.cloud import aiplatform

PIPELINE_PACKAGE_PATH = "stgcn_pipeline.yaml"
PIPELINE_DISPLAY_NAME = "STGCN-Traffic-Prediction"
PROJECT_ID = "canvas-joy-456715-b1"
REGION = "us-east1"

def main():
    aiplatform.init(project=PROJECT_ID, location=REGION)
    pipeline_job = aiplatform.PipelineJob(
        display_name=PIPELINE_DISPLAY_NAME,
        template_path=PIPELINE_PACKAGE_PATH,
        parameter_values={
            "route_distances_path": "gs://traffic_prediction_25/PeMSD7_W_228.csv",
            "speeds_array_path": "gs://traffic_prediction_25/PeMSD7_V_228.csv",
            "model_dir": "gs://traffic_prediction_25/model_output/",
        },
    )
    pipeline_job.run(sync=True)

if __name__ == "__main__":
    main()
