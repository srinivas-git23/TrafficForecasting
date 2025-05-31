FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY deploy_model.py .

CMD ["python", "deploy_model.py"]
