#!/bin/bash

# Start Ollama in background
echo "🔄 Starting Ollama server..."
ollama serve &

# Wait until Ollama is ready to serve LLaMA3
until curl -s http://localhost:11434/api/tags | grep -q "llama3"; do
    echo "⏳ Waiting for llama3 model to be available..."
    sleep 3
done

# Confirm model available
echo "✅ Llama3 model ready."

# Run Streamlit app
echo "🚀 Launching Streamlit application..."
streamlit run ui/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
