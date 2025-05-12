# Base image with Go and Python
FROM golang:1.24-alpine AS builder

# Install Python, pip, git, and build tools
RUN apk add --no-cache python3 py3-pip git make build-base python3-dev gcc

WORKDIR /app

# Create a directory for NLTK data
RUN mkdir -p /nltk_data
ENV NLTK_DATA=/nltk_data

# Create and activate a Python virtual environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Copy and install Python dependencies
COPY requirements.txt . 
RUN pip3 install --no-cache-dir -r requirements.txt

# Pre-download NLTK 'punkt' model to the specified directory
RUN python3 -c "import nltk; nltk.download('punkt', download_dir='${NLTK_DATA}')"

# Copy Go module files and download Go dependencies
COPY go.mod go.sum ./ 
RUN go mod download && go mod vendor

# Copy the rest of the application source code
COPY . .

# Build the Go application
RUN CGO_ENABLED=1 GOOS=linux go build -v -o /app/pdf_qna_service .

# --- Final Stage ---
FROM alpine:latest

# Install curl for health checks and other utilities
RUN apk add --no-cache curl tar gzip wget make

# Install CA certificates for HTTPS calls
# Also add build-base for packages that might need to compile C code, and python3-dev for C extensions
RUN apk --no-cache add ca-certificates python3 py3-pip make build-base python3-dev

# Environment variables
# Qdrant will run locally within the same container
ENV QDRANT_URL="http://qdrant:6333"
# GEMINI_API_KEY should be provided at runtime, e.g., docker run -e GEMINI_API_KEY="your_key"
ENV GEMINI_API_KEY=""
ENV GRPC_PORT="50051"
ENV DB_PATH="/app/database/pdf_qna.db"
ENV NLTK_DATA=/app/nltk_data
# Define Qdrant version - choose a version compatible with x86_64-unknown-linux-musl
ENV QDRANT_VERSION="v1.9.2"
ENV QDRANT_DIST_URL="https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-x86_64-unknown-linux-musl.tar.gz"

WORKDIR /app

# Download and install Qdrant
RUN mkdir -p /app/qdrant_dist && \
    wget -O qdrant.tar.gz "${QDRANT_DIST_URL}" && \
    tar -xzf qdrant.tar.gz -C /app/qdrant_dist && \
    rm qdrant.tar.gz
# The Qdrant binary will be at /app/qdrant_dist/qdrant

# Create a non-root user and group
RUN addgroup -S appgroup && adduser -S appuser -G appgroup

# Copy built Go application from builder stage
COPY --from=builder /app/pdf_qna_service /app/pdf_qna_service
# Copy Python script and NLTK data from builder stage
COPY --from=builder /app/pdf_processor.py /app/pdf_processor.py
COPY --from=builder /nltk_data ${NLTK_DATA}
# Copy config.json (its values for API key, Qdrant URL, DB path will be overridden by ENV vars)
COPY config.json /app/config.json
# Copy sample.pdf (optional, for default CLI behavior if no args given)
COPY sample.pdf /app/sample.pdf
# Copy static directory
COPY static ./static/
# Copy static assets
COPY static ./static/
# Copy the startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh
# Copy Python script and requirements
COPY pdf_processor.py /app/pdf_processor.py
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r /app/requirements.txt --break-system-packages

# Create directories and set permissions
# Qdrant data directory (start.sh expects /qdrant_data)
RUN mkdir -p /qdrant_data && chown -R appuser:appgroup /qdrant_data && chmod -R 770 /qdrant_data
# App-specific directories
RUN mkdir -p /app/database && chown -R appuser:appgroup /app/database && chmod -R 770 /app/database
RUN mkdir -p /app/uploads && chown -R appuser:appgroup /app/uploads && chmod -R 770 /app/uploads

# Ensure the NLTK_DATA directory is also owned by appuser
RUN chown -R appuser:appgroup ${NLTK_DATA}

# Change ownership of the app directory to the new user
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose gRPC port (value from GRPC_PORT env var)
EXPOSE ${GRPC_PORT}
# Expose Qdrant's default gRPC and HTTP ports
EXPOSE 6333
EXPOSE 6334
# Expose HTTP port
EXPOSE 8080

# Command to run the application
CMD ["/app/start.sh"]
