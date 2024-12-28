# Use the official Airflow image as a parent image
FROM apache/airflow:2.10.4

# Switch to root user to install system dependencies
USER root

# Install system dependencies (wget, unzip, etc.)
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    ffmpeg \
    libsndfile1 \
    && apt-get clean

# Switch to airflow user
USER airflow

# Install dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: Add commands to download and unzip as airflow user
RUN mkdir -p /opt/airflow/vosk-models && \
    cd /opt/airflow/vosk-models && \
    wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip && \
    unzip vosk-model-en-us-0.22-lgraph.zip

# Copy the Streamlit application files to the container
COPY app.py /opt/airflow/app.py

# Expose the port Streamlit runs on
EXPOSE 8501

# Set the working directory
WORKDIR /opt/airflow

# Command to run the Streamlit app
CMD ["streamlit", "run", "/opt/airflow/app.py"]

# Optional: Verify installations
#RUN pip show vosk pydub google-generativeai langchain faiss transformers torch && which wget && which unzip