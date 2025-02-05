# Use the official Python 3.10.11 base image from Docker Hub
FROM python:3.10.11-slim

# Set the working directory inside the container
WORKDIR /app

# Update package lists and install sqlite3 and libsqlite3-dev (required by Chroma)
RUN apt-get update && \
    apt-get install -y sqlite3 libsqlite3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application files into the container
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]
