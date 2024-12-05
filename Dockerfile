FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    GOOGLE_API_KEY=AIzaSyCWmWlwM4R3Otqp0Go51z9EVCNfEgWa2rM  # Optional, or use secrets

# Set the working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN apt-get update && apt-get install -y gcc libpq-dev && \
    pip install --no-cache-dir -r requirements.txt

# Expose Cloud Run's default port
EXPOSE 8080

# Command to run the Streamlit app
CMD ["streamlit", "run", "college.py", "--server.port=8080", "--server.enableCORS=false"]
