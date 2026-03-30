# Use a lightweight Python base image
FROM python:3.12-slim

# Install uv directly from Astral's official Docker image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1

# Tell uv to build the virtual environment OUTSIDE of the /app folder
# so that our docker-compose volume mount doesn't overwrite it!
ENV UV_PROJECT_ENVIRONMENT="/opt/venv"
ENV PATH="/opt/venv/bin:$PATH"

# Copy dependency files FIRST to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv into the /opt/venv environment
RUN uv sync --frozen --no-dev

# Copy the rest of your application code
COPY . .