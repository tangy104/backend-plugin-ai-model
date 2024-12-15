# Use the official Python slim image for smaller size
FROM python:3.12.4-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file first for dependency caching
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the application code to the container
COPY . /app/ /code/app/

# Expose the port FastAPI will run on
EXPOSE 8000

# specify default commands
CMD ["fastapi", "run", "app/main.py", "--port", "8000"]
# CMD ["uvicorn", "app.main:app", "--reload" "--host", "0.0.0.0", "--port", "8000"]
