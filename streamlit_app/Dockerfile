# Using Python 3.8 image
FROM python:3.8-slim

# Setting the working directory
WORKDIR /app

# Copying the requirements file to install dependencies
COPY requirements.txt .

# Copying the Streamlit application files
COPY . .

# Installing dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Starting the Streamlit application
CMD ["streamlit", "run", "streamlit.py"]