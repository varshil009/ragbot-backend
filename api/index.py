from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# Add parent directory to path so we can import base_rag2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_rag2 import rag_process

app = Flask(__name__)

# Allow frontend - Vercel serverless is stateless, no sessions
CORS(app, supports_credentials=True, origins=["*"])

REFERENCE_BOOKS = [
    'deep learning with python',
    'python data science handbook'
]

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "RAGBOT backend is running on Vercel"})


@app.route('/select_book', methods=['POST', 'OPTIONS'])
def select_book():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        data = request.get_json()
        book_name = data.get('book_name') if data else None

        if not book_name or book_name not in REFERENCE_BOOKS:
            return jsonify({'error': 'Invalid book selection'}), 400

        # In serverless, we just validate - client must send book_name with each /ask request
        return jsonify({
            'message': f'{book_name} selected successfully',
            'book_name': book_name  # Return so client can store it
        })

    except Exception as e:
        print(f"Error in select_book: {e}")
        return jsonify({'error': 'Server error'}), 500


@app.route('/status', methods=['GET'])
def status():
    # Serverless is always ready
    return jsonify({'status': 'ready'})


@app.route('/ask', methods=['POST', 'OPTIONS'])
def ask():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        data = request.get_json()
        question = data.get('question') if data else None
        book_name = data.get('book_name') if data else None

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        if not book_name or book_name not in REFERENCE_BOOKS:
            return jsonify({'error': 'No valid book selected. Please include book_name in request.'}), 400

        print(f"Processing question for book: {book_name}")
        rag_model = rag_process(book_name)
        answer = rag_model.execute(question)

        return jsonify({'answer': answer})

    except Exception as e:
        print(f"Error in ask: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/end_session', methods=['POST', 'OPTIONS'])
def end_session():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    # No session to clear in serverless
    return jsonify({'message': 'Session ended successfully'})


# Vercel serverless handler
app = app
