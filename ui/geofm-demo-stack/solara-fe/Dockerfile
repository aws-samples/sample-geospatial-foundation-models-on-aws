# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install localtileserver deps. NOT NEEDED anymore
#RUN apt-get update && \
#    apt-get install -y libexpat1 libexpat1-dev && \
#    apt-get clean && \
#    rm -rf /var/lib/apt/lists/*

# Upgreade pip
RUN pip install --no-cache-dir --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Disable telemetry
ENV SOLARA_TELEMETRY_MIXPANEL_ENABLE=False

# Check if "--auto-restart" is required/preferred
CMD ["solara", "run", "./pages", "--host=0.0.0.0", "--port", "8000", "--production"]
# Run app.py when the container launches