#!/bin/bash

# Make sure script exits if any command fails
set -e

echo "🔁 Downloading spaCy model: en_core_web_sm..."
python -m spacy download en_core_web_sm
echo "✅ spaCy model downloaded successfully."
