# Medical RAG API

This API serves as an interface to the medical RAG system, allowing Custom GPTs and other applications to search through medical literature using both text and image queries.

## API Endpoints

### POST /search
Search the medical database with text or image queries.

**Request format:**
```json
{
  "query": "What are the treatment options for atrial fibrillation?",
  "image_query": "eczema skin rash",
  "limit": 5
}
```

**Response format:**
```json
{
  "text_results": [
    {
      "score": 0.85,
      "source": "DSM 5 TR",
      "page": "245",
      "original_id": "dsm_5_tr_chunk_1039",
      "content": "Text content from the medical document..."
    }
  ],
  "image_results": [
    {
      "score": 0.78,
      "source": "Fitzpatricks Color Atlas",
      "page": "15",
      "original_id": "Fitzpatricks_Color_Atlas_page_15_image_1"
    }
  ]
}
```

### GET /health
Health check endpoint to verify the API is running.

**Response:**
```json
{
  "status": "ok"
}
```

## Deployment on Render

1. Connect to your GitHub repository
2. Use the following settings:
   - Runtime: Python 3
   - Build Command: `pip install -r requirements_api.txt`
   - Start Command: `gunicorn api:app`

## Custom GPT Integration

Configure your Custom GPT with the following schema:

```json
{
  "openapi": "3.0.0",
  "info": {
    "title": "Medical Search API",
    "version": "1.0.0"
  },
  "paths": {
    "/search": {
      "post": {
        "summary": "Search medical literature",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "query": {
                    "type": "string",
                    "description": "Text search query"
                  },
                  "image_query": {
                    "type": "string", 
                    "description": "Image search query (optional)"
                  },
                  "limit": {
                    "type": "integer",
                    "description": "Maximum results to return (default: 3)"
                  }
                },
                "required": ["query"]
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Search results"
          }
        }
      }
    }
  }
}
``` 