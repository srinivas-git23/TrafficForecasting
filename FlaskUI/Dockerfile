# Use Python 3.11 base image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

# Set port (Cloud Run uses PORT env variable)
ENV PORT 8080
EXPOSE 8080

# Run Flask app using gunicorn for production
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
