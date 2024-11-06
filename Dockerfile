# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Expose port 5000 (if your app runs on a specific port)
EXPOSE 5000

# Define environment variables (optional)
ENV PYTHONUNBUFFERED 1

# Run the application (adjust according to your app's entry point)
CMD ["python", "app.py"]
