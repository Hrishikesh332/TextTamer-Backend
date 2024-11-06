from flask import Flask, request, jsonify
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting
from functools import wraps
import os

app = Flask(__name__)

GENERATION_CONFIG = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}


def initialize_vertex_ai():
    vertexai.init(project="aihack24ber-8540", location="us-central1")
    return GenerativeModel("gemini-1.5-flash-002")

def validate_input(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        data = request.get_json()
        required_fields = ['text', 'language', 'level']
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
            
        return f(*args, **kwargs)
    return decorated_function

@app.route('/transcribe', methods=['POST'])
@validate_input
def transcribe():
    """
    Expects JSON with format
    {
        "text": "Text to transcribe",
        "language": "Target language",
        "level": "Language level (e.g., beginner, intermediate, advanced)"
    }
    """
    try:
        data = request.get_json()
        model = initialize_vertex_ai()
        
        prompt = f"""
        You are a transcriber. Transcribe the following text:
        
        Text: {data['text']}
        Target Language: {data['language']}
        Language Level: {data['level']}
        
        Please provide the transcription while maintaining the specified language level.
        """
        
        response = model.generate_content(
            prompt,
            generation_config=GENERATION_CONFIG,
        )
        
        transcribed_text = ""
        if hasattr(response, '__iter__'):
            for chunk in response:
                transcribed_text += chunk.text
        else:
            transcribed_text = response.text
            
        return jsonify({
            'status': 'success',
            'transcribed_text': transcribed_text,
            'source_language': 'auto-detected',
            'target_language': data['language'],
            'level': data['level']
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))