#!/usr/bin/env python3
"""
Search Medical Literature

This script searches for text and images in the medical literature database.

Usage:
    python3 search_medical.py --query "your search query" [--limit 5] [--image-query "your image search"]

Example:
    python3 search_medical.py --query "What are the symptoms of diabetes?"
    python3 search_medical.py --query "Show me images of skin rashes" --image-query "skin rash"
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

import torch
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchRequest
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

# Constants
QDRANT_URL = os.getenv("QDRANT_URL", "https://66fbeec8-f3a4-4a70-b9b9-b50452eaa25e.us-east4-0.gcp.cloud.qdrant.io")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.L0saX3GlmkbEqxMQqQvSTqh5UUexyijROQkdyEfNYKQ")
TEXT_COLLECTION = "medical_literature"
IMAGE_COLLECTION = "medical_images"
TEXT_EMBEDDINGS_DIM = 384
IMAGE_EMBEDDINGS_DIM = 512

def connect_to_qdrant() -> QdrantClient:
    """Connect to Qdrant Cloud and return client."""
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=120
    )
    return client

def load_models():
    """Load models for text and image embeddings."""
    print("Loading models...")
    # Determine device
    device_name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(device_name)
    print(f"Using device: {device_name}")
    
    # Load text embedding model
    text_model = SentenceTransformer('all-MiniLM-L6-v2', device=device_name)
    
    # Load CLIP model for image search (if needed)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.to(device)
    
    return text_model, clip_model, clip_processor, device

def create_text_embedding(query: str, model) -> List[float]:
    """Create embedding for text query."""
    return model.encode(query).tolist()

def create_image_embedding(query: str, model, processor, device) -> List[float]:
    """Create embedding for image query using text."""
    inputs = processor(text=query, images=None, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "pixel_values"}
    
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    
    # Normalize
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()[0].tolist()

def format_result(result: Dict[str, Any], query_type: str) -> str:
    """Format search result for display."""
    payload = result.payload
    score = round(float(result.score), 4)
    
    source = payload.get("source", "Unknown")
    content_type = "Text" if query_type == "text" else "Image"
    original_id = payload.get("original_id", "Unknown")
    page_num = payload.get("page_num", "Unknown")
    
    formatted = f"\n{content_type} Result (Score: {score}):\n"
    formatted += f"Source: {source}\n"
    formatted += f"Page: {page_num}\n"
    formatted += f"Original ID: {original_id}\n"
    
    if query_type == "text":
        content = payload.get("text", "")
        formatted += f"\nContent:\n{content}\n"
    else:
        # For images, we only have the reference since we can't display them directly
        formatted += "\nImage found in document. Use the source and page information to locate it.\n"
    
    return formatted

def search_medical_database(client: QdrantClient, 
                           text_query: Optional[str], 
                           image_query: Optional[str], 
                           text_model, 
                           clip_model, 
                           clip_processor,
                           device,
                           limit: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """Search both text and image collections based on the queries."""
    results = {"text": [], "image": []}
    
    # Search text if query provided
    if text_query:
        text_vector = create_text_embedding(text_query, text_model)
        try:
            # Try the new API method
            text_results = client.query_points(
                collection_name=TEXT_COLLECTION,
                vector=text_vector,
                limit=limit
            )
            results["text"] = text_results.points
        except Exception as e:
            print(f"Error with query_points: {e}")
            # Fall back to the deprecated method
            try:
                text_results = client.search(
                    collection_name=TEXT_COLLECTION,
                    query_vector=text_vector,
                    limit=limit
                )
                results["text"] = text_results
            except Exception as e2:
                print(f"Error with search fallback: {e2}")
    
    # Search images if query provided
    if image_query:
        image_vector = create_image_embedding(image_query, clip_model, clip_processor, device)
        try:
            # Try the new API method
            image_results = client.query_points(
                collection_name=IMAGE_COLLECTION,
                vector=image_vector,
                limit=limit
            )
            results["image"] = image_results.points
        except Exception as e:
            print(f"Error with query_points for images: {e}")
            # Fall back to the deprecated method
            try:
                image_results = client.search(
                    collection_name=IMAGE_COLLECTION,
                    query_vector=image_vector,
                    limit=limit
                )
                results["image"] = image_results
            except Exception as e2:
                print(f"Error with search fallback for images: {e2}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Search medical literature")
    parser.add_argument("--query", type=str, help="Text query")
    parser.add_argument("--image-query", type=str, help="Image query (text describing the image)")
    parser.add_argument("--limit", type=int, default=3, help="Maximum number of results to return")
    args = parser.parse_args()
    
    if not args.query and not args.image_query:
        parser.print_help()
        sys.exit(1)
    
    # Connect to Qdrant
    client = connect_to_qdrant()
    print(f"Connected to Qdrant at {QDRANT_URL}")
    
    # Load models
    text_model, clip_model, clip_processor, device = load_models()
    
    # Search
    results = search_medical_database(
        client=client,
        text_query=args.query,
        image_query=args.image_query,
        text_model=text_model,
        clip_model=clip_model,
        clip_processor=clip_processor,
        device=device,
        limit=args.limit
    )
    
    # Display results
    print(f"\n{'='*50}")
    print(f"SEARCH RESULTS")
    print(f"{'='*50}")
    
    if args.query:
        print(f"\nText Query: '{args.query}'")
        if results["text"]:
            for result in results["text"]:
                print(format_result(result, "text"))
        else:
            print("No text results found.")
    
    if args.image_query:
        print(f"\nImage Query: '{args.image_query}'")
        if results["image"]:
            for result in results["image"]:
                print(format_result(result, "image"))
        else:
            print("No image results found.")
    
    print(f"\n{'='*50}")

if __name__ == "__main__":
    main() 