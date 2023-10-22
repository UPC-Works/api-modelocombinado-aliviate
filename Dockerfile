# Use an official Python runtime as a parent image
FROM python:3.11.4

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app


# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Instala las dependencias
RUN pip install uvicorn

# Instala las dependencias
RUN pip install fastapi

# Copy the current directory contents into the container
COPY . /app

# Command to run on container start
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]