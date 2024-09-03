# Using the official Python image from docker
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy the requirements into the container
COPY requirements.txt .

# Install the required dependencies
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# Copying the rest of the application to the working directory
COPY . .

# Expose the port that the Streamlit app runs on
EXPOSE 8080

# Running the Streamlit app on port 8080
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]

