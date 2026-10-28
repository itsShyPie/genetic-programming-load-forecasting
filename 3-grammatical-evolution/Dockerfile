FROM python:3.12-slim

WORKDIR /app

# Install numpy
RUN pip install --no-cache-dir numpy==1.26.4

# Copy project files
COPY data/ data/
COPY src/  src/

WORKDIR /app/src

CMD ["python3", "main.py"]
