from flask import Flask, render_template_string, request
import requests
import os
import json
import numpy as np
from google.auth.transport.requests import Request as GoogleRequest
import google.auth

app = Flask(__name__)

# Replace with your actual endpoint info
#ENDPOINT_URL = "https://us-east1-aiplatform.googleapis.com/v1/projects/canvas-joy-456715-b1/locations/us-east1/endpoints/2149936658441568256:predict"
ENDPOINT_URL = os.environ.get("VERTEX_ENDPOINT_URL") 
# Load mean and std for normalization
MEAN = np.array([
    52.69398185, 53.90552486, 57.80827743, 52.7523086, 53.40117403, 56.78225138,
    53.1226914, 64.08870363, 56.68529006, 53.1551105, 60.72802881, 53.21778808,
    57.20292028, 60.75633386, 60.09406077, 49.55187451, 63.18920679, 55.88162983,
    64.11615036, 61.95404499, 56.26108919, 57.08177782, 61.94263023, 61.83439227,
    55.74215667, 60.00659037
])
STD = np.array([
    18.95678552, 22.63782019, 11.61472907, 20.85760142, 13.56871364, 18.1111499,
    18.55411899, 10.67993086, 10.60895735, 18.22740348, 8.48041052, 17.23787313,
    14.63585896, 9.39328996, 11.05922825, 22.25900655, 12.54500973, 18.05319553,
    5.27975133, 14.83901396, 12.88447352, 17.53805178, 7.06272208, 6.79413435,
    17.2553262, 16.1712089
])
ROUTES = ["t+1", "t+2", "t+3"]
TIMESTEPS = [
    0, 1, 4, 7, 8, 11, 15, 108, 109, 114, 115, 118, 120, 123,
    124, 126, 127, 129, 130, 132, 133, 136, 139, 144, 147, 216
]

def get_access_token():
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(GoogleRequest())
    return credentials.token

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Traffic Prediction UI</title>
    <style>
        textarea, input[type=submit] {
            font-family: monospace;
            font-size: 14px;
        }
        table {
            border-collapse: collapse;
            margin-top: 20px;
        }
        td, th {
            border: 1px solid #ccc;
            padding: 6px;
            text-align: center;
        }
        .heat {
            color: white;
        }
    </style>
</head>
<body>
    <h2>Traffic Forecast Input</h2>
    <form method="post">
        <label for="input_data">Paste 12x26 comma-separated values (no brackets):</label><br>
        <textarea id="input_data" name="input_data" rows="15" cols="100">{{ input_data }}</textarea><br><br>
        <input type="submit" value="Send Request">
    </form>
    {% if table_data %}
        <h3>Input Sequence (Last 12 Time Steps)</h3>
        <table>
            <tr><th>Time Step</th>{% for i in route_labels %}<th>{{ i }}</th>{% endfor %}</tr>
            {% for i in range(12) %}
                <tr>
                    <td>t-{{ 11 - i }}</td>
                    {% for val in input_matrix[i] %}
                        <td>{{ '%.2f' % val }}</td>
                    {% endfor %}
                </tr>
            {% endfor %}
        </table>
        <h3>Prediction Result (Denormalized)</h3>
        <table>
            <tr>
                <th>Time / Route</th>
                {% for r in route_labels %}
                    <th>{{ r }}</th>
                {% endfor %}
            </tr>
            {% for i in range(table_data|length) %}
                <tr>
                    <th>{{ timesteps[i] }}</th>
                    {% for val in table_data[i] %}
                        {% set shade = val / max_value * 255 %}
                        <td style="background-color: rgb({{ 255 - shade|int }}, 100, {{ shade|int }}); color: white;">
                            {{ '%.2f' % val }}
                        </td>
                    {% endfor %}
                </tr>
            {% endfor %}
        </table>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def predict():
    input_data = ""
    table_data = []
    input_matrix = []
    max_value = 1

    if request.method == 'POST':
        input_data = request.form['input_data']
        try:
            rows = input_data.strip().split("\n")
            arr = np.array([[float(x) for x in row.strip().split(",")] for row in rows])

            if arr.shape != (12, 26):
                raise ValueError("Input must be 12 rows of 26 values each.")

            input_matrix = arr.tolist()
            norm_input = (arr - MEAN) / STD
            reshaped = norm_input[..., None]
            payload = json.dumps({"instances": [reshaped.tolist()]})

            headers = {
                "Authorization": f"Bearer {get_access_token()}",
                "Content-Type": "application/json"
            }
            response = requests.post(ENDPOINT_URL, headers=headers, data=payload)
            response_json = response.json()

            pred = np.array(response_json['predictions'][0])
            denorm_pred = (pred * STD) + MEAN
            table_data = denorm_pred.tolist()
            max_value = np.max(denorm_pred)

        except Exception as e:
            table_data = [[f"Error: {str(e)}"]]

    return render_template_string(
        HTML_PAGE,
        input_data=input_data,
        input_matrix=input_matrix,
        table_data=table_data,
        max_value=max_value,
        timesteps=ROUTES,
        route_labels=TIMESTEPS
    )

if __name__ == '__main__':
    app.run(debug=True, port=8080)
