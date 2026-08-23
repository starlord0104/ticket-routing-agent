FROM python:3.11-slim

WORKDIR /app

# Force UTF-8 output so the box-drawing characters in train.py / evaluate.py
# banners don't crash on systems where the default locale is ASCII.
ENV PYTHONIOENCODING=utf-8
ENV PYTHONUTF8=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/     ./src/
COPY app/     ./app/
COPY train.py evaluate.py ./

# Create directories
RUN mkdir -p data models plots

# Pre-download the embedding model so the container starts fast
RUN python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('all-MiniLM-L6-v2')"

# Default: run FastAPI backend
# Override with: docker run ... streamlit run app/streamlit_app.py
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

EXPOSE 8000 8501
