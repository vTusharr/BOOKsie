#!/bin/sh
set -e

# Qdrant is now managed by docker-compose, so we don't start it here.
# The Go application will connect to the 'qdrant' service at QDRANT_URL 
# (e.g., http://qdrant:6333), which is set as an environment variable 
# in docker-compose.yml.

echo "Starting PDF QnA Service..."
# The Go application (pdf_qna_service) has its own internal retry logic 
# for connecting to Qdrant.
/app/pdf_qna_service -serve &
APP_PID=$!

# Trap SIGINT and SIGTERM to gracefully shut down the Go application
trap 'echo "Received shutdown signal. Terminating PDF QnA Service..."; kill -TERM $APP_PID 2>/dev/null; wait $APP_PID' SIGINT SIGTERM

# Wait for the Go application to exit.
# Using `wait $APP_PID` directly makes the script wait for that specific process.
wait $APP_PID
EXIT_CODE=$?

echo "PDF QnA Service exited with code $EXIT_CODE. Script is ending."
# An explicit kill might be redundant if `wait` successfully caught the exit,
# but it ensures cleanup if the trap or process termination was unusual.
kill -TERM $APP_PID 2>/dev/null || true
sleep 1 # Give a brief moment for graceful termination
kill -KILL $APP_PID 2>/dev/null || true # Force kill if still running

exit $EXIT_CODE
