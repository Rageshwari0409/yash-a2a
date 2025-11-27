# ---- Build stage ----
    FROM python:3.12-slim AS builder

    WORKDIR /app
    
    # Install build tools only in builder stage
    RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        && rm -rf /var/lib/apt/lists/*
    
    # Install uv and dependencies
    RUN pip install --no-cache-dir uv
    
    COPY pyproject.toml ./        
    
    RUN uv pip install --system --no-cache-dir .    
    
    # ---- Final runtime stage ----
    FROM python:3.12-slim
    
    WORKDIR /app
    
    # Copy only installed packages and code from builder
    COPY --from=builder /usr/local /usr/local
    COPY . .
    
    EXPOSE 5000
    
    CMD ["python", "-m", "agents.obligation_user_story", "--host", "0.0.0.0", "--port", "5000"]

    