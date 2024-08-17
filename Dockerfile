# Using the official Python image from docker
FROM python:3.9-slim

#Set working directory
WORKDIR /app

#Copy the requirements into the container
COPY requirements.txt .

#Install the required dependencies
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

#Copying the rest of the application to the working directory
COPY . .

#Expose the port that the Streamlit app runs on
EXPOSE 8501

#Running the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

