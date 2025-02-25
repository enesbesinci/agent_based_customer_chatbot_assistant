# Using Python 3.9 image
FROM python:3.9-slim

# Setting the working directory
WORKDIR /app

# Copying the requirements file to install dependencies
COPY requirements.txt .

# Copying the Flask application files
COPY . .

# Installing dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Starting the Flask application
CMD ["python", "app.py"]