import requests
import json
from pathlib import Path

def get_llm_description(centiloid_score, suvr_values, model="deepseek-coder:latest"):
    """Get image description using LLM."""
    
    # Prepare context for LLM
    prompt = f"""
    Analyze the following Amyloid PET scan results:
    - Centiloid Score: {centiloid_score}
    - Regional SUVr values: {json.dumps(suvr_values, indent=2)}
    
    Provide a clinical interpretation including:
    1. Overall amyloid burden assessment
    2. Regional distribution patterns
    3. Clinical significance
    """
    
    # Call OLLAMA API
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': model,
            'prompt': prompt,
            'stream': False
        }
    )
    
    return response.json()['response']