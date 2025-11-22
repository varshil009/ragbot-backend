from flask import Flask, request, jsonify, session
from flask_session import Session
from flask_cors import CORS
import os
import traceback
from base_rag2 import rag_process
app = Flask(__name__)

app.secret_key = os.urandom(24)

app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './session_data'
app.config['SESSION_COOKIE_NAME'] = 'ragbot_session'

# IMPORTANT FIX FOR GITHUB PAGES → RAILWAY
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = True

os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

Session(app)

# Allow frontend
CORS(app, supports_credentials=True)


REFERENCE_BOOKS = [
    'deep learning with python',
    'python data science handbook'
]

# ----------------------------
# API ROUTES ONLY (NO HTML)
# ----------------------------

@app.route('/')
def home():
    return jsonify({"message": "RAGBOT backend is running"})


@app.route('/select_book', methods=['POST'])
def select_book():
    try:
        book_name = request.json.get('book_name')

        if not book_name or book_name not in REFERENCE_BOOKS:
            return jsonify({'error': 'Invalid book selection'}), 400

        session['current_book'] = book_name
        print(f"Book '{book_name}' selected successfully")

        return jsonify({'message': f'{book_name} selected successfully'})

    except Exception as e:
        print("Error in select_book:", str(e))
        return jsonify({'error': 'Server error'}), 500


@app.route('/status', methods=['GET'])
def status():
    if 'current_book' in session:
        return jsonify({'status': 'ready'})
    return jsonify({'status': 'processing'})


@app.route('/ask', methods=['POST'])
def ask():
    try:
        question = request.json.get('question')

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        if 'current_book' not in session:
            return jsonify({'error': 'No book selected'}), 400

        book_name = session['current_book']

        print(f"Processing question for book: {book_name}")
        rag_model = rag_process(book_name)
        answer = rag_model.execute(question)

        return jsonify({'answer': answer})

    except Exception as err:
        print("Error:", str(err))
        traceback.print_exc()
        return jsonify({'error': str(err)}), 500


@app.route('/end_session', methods=['POST'])
def end_session():
    session.pop('current_book', None)
    return jsonify({'message': 'Session ended successfully'})


# ----------------------------
# Local development
# ----------------------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
