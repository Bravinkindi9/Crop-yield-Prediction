FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trainer package
COPY trainer ./trainer

# Set the entry point for the container
ENTRYPOINT ["python", "-m", "trainer.train"]