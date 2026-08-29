import os
import json
import re
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from quiz_engine import process_document_and_generate_quiz

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Ollama LLM with native JSON output mode
llm = ChatOllama(
    model="llama3.2",
    temperature=0.2,
    format="json"
)

def extract_json_from_response(raw_text: str) -> list:
    """Robust parser to extract JSON arrays from LLM outputs."""
    try:
        data = json.loads(raw_text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ["questions", "quiz", "data", "mcqs", "results"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
    except Exception:
        pass

    # Regex extraction fallback
    match = re.search(r'\[\s*\{.*\}\s*\]', raw_text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    
    raise ValueError("Failed to parse structured JSON quiz output from Ollama.")

def process_and_generate(file_path: str, num_questions: int = 5, difficulty: str = "Medium"):
    # 1. Document Extraction
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        docs = docs[:10]  # Cap to top 10 pages for optimum local LLM speed
    else:
        loader = TextLoader(file_path, encoding='utf-8')
        docs = loader.load()

    valid_docs = [d for d in docs if d.page_content and d.page_content.strip()]
    if not valid_docs:
        raise ValueError("The uploaded document contains no readable text content.")

    # 2. Text Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        strip_whitespace=True
    )
    splits = text_splitter.split_documents(valid_docs)
    
    # Take representative text chunks
    selected_chunks = splits[:5] if len(splits) > 5 else splits
    context_text = "\n\n".join([doc.page_content for doc in selected_chunks])

    # 3. Prompt Construction
    prompt = f"""You are an expert assessment creation AI. Generate exactly {num_questions} multiple-choice questions (MCQs) at a {difficulty} difficulty level based strictly on the provided context.

Context:
{context_text}

Requirements:
- Return ONLY a JSON array of objects.
- Each object MUST contain:
  1. "question": A clear, concise question string.
  2. "options": An array of exactly 4 plausible option strings.
  3. "correct_answer": The exact string of the correct option from the options list.
  4. "explanation": A 1-2 sentence explanation of why the correct answer is right.

JSON Format Example:
[
  {{
    "question": "What is the primary function of...?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": "Option A",
    "explanation": "Option A is correct because..."
  }}
]"""

    response = llm.invoke(prompt)
    quiz = extract_json_from_response(response.content.strip())
    return quiz

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-quiz', methods=['POST'])
def generate_quiz():
    if 'file' not in request.files:
        return jsonify({'error': 'No document uploaded.'}), 400
    
    file = request.files['file']
    num_questions = int(request.form.get('num_questions', 5))
    difficulty = request.form.get('difficulty', 'Medium')

    if file.filename == '':
        return jsonify({'error': 'Selected file is empty.'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        quiz_data = process_and_generate(file_path, num_questions=num_questions, difficulty=difficulty)
        return jsonify({'success': True, 'quiz': quiz_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True, port=5000)