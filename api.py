#!/usr/bin/env python3
"""
Medical RAG API

This script provides a Flask API for searching the medical RAG system.
"""

from flask import Flask, request, jsonify
import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import search functionality
from search_medical import (
    connect_to_qdrant, 
    load_models, 
    search_medical_database, 
    create_text_embedding
)

app = Flask(__name__)

# Global variables for models and client
client = None
text_model = None
clip_model = None
clip_processor = None
device = None

@app.before_first_request
def initialize():
    """Initialize the models and client before the first request."""
    global client, text_model, clip_model, clip_processor, device
    
    logger.info("Initializing Qdrant client and models...")
    try:
        client = connect_to_qdrant()
        text_model, clip_model, clip_processor, device = load_models()
        logger.info("Initialization successful!")
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        raise

@app.route('/search', methods=['POST'])
def search():
    """
    Handle search requests.
    
    Expected JSON payload:
    {
        "query": "text query",
        "image_query": "optional image query",
        "limit": 3
    }
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Extract parameters
        query = data.get('query', '')
        image_query = data.get('image_query', '')
        limit = int(data.get('limit', 3))
        
        if not query and not image_query:
            return jsonify({"error": "At least one of 'query' or 'image_query' must be provided"}), 400
        
        logger.info(f"Received search request - Text: '{query}', Image: '{image_query}'")
        
        # Search the database
        results = search_medical_database(
            client=client,
            text_query=query,
            image_query=image_query,
            text_model=text_model,
            clip_model=clip_model,
            clip_processor=clip_processor,
            device=device,
            limit=limit
        )
        
        # Format results for JSON response
        formatted_results = {
            "text_results": [],
            "image_results": []
        }
        
        for result in results.get("text", []):
            formatted_results["text_results"].append({
                "score": float(result.score),
                "source": result.payload.get("source", "Unknown"),
                "page": result.payload.get("page_num", "Unknown"),
                "original_id": result.payload.get("original_id", "Unknown"),
                "content": result.payload.get("text", "")
            })
        
        for result in results.get("image", []):
            formatted_results["image_results"].append({
                "score": float(result.score),
                "source": result.payload.get("source", "Unknown"),
                "page": result.payload.get("page_num", "Unknown"),
                "original_id": result.payload.get("original_id", "Unknown")
            })
        
        logger.info(f"Returning {len(formatted_results['text_results'])} text results and {len(formatted_results['image_results'])} image results")
        return jsonify(formatted_results)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the API is running."""
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    # Initialize on startup
    initialize()
    
    # Get port from environment variable or use 8000 as default
    port = int(os.environ.get("PORT", 8000))
    
    # Run the app
    app.run(host='0.0.0.0', port=port) 