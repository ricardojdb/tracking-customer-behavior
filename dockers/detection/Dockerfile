# Use pytorch image with Cuda for GPU support
FROM pytorch/pytorch

LABEL GPU.run="docker run --runtime=nvidia  --name=detection-service --rm -dit -v <PATH>:/app -p 7003:7000 dockers_yolodetection" \
      CPU.run="docker run --name=detection-service --rm -dit -v <PATH>:/app -p 7000:7000 dockers_yolodetection"

# Create app/ directory
RUN mkdir /app

# Set the working directory to /app
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app

# Set local directory as a volume
VOLUME .:/app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port available to the world outside this container
EXPOSE 7000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]