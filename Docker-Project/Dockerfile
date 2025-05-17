# Use official Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy everything from your current directory to /app inside the container
COPY . .

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]

# check the app.py filename as per your filename

