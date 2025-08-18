# Dockerfile
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /milliman

# Copy the dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy over the directory with the code
COPY . .

# command to run the app
CMD [ "python", "./app.py" ]