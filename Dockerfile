FROM python:3.11-slim

# Set up user with UID 1000 to avoid permission issues in Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements and install dependencies
COPY --chown=user:user requirements.txt $HOME/app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r $HOME/app/requirements.txt

# Copy application files
COPY --chown=user:user . $HOME/app

# Run training to build model pickle
RUN python models/train_model.py

# Expose port 7860 which Hugging Face uses
EXPOSE 7860

# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
