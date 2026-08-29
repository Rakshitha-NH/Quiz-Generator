import json
import re
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama

# Optimized configuration for low CPU/GPU inference latency
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.1,
    format="json",
    num_ctx=2048,       # Keep attention window small for high processing speed
    num_predict=1024    # Strict token cap to prevent generation lag
)

def clean_json_response(raw_text: str) -> list:
    """Safely parses structured JSON response."""
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

    match = re.search(r'\[\s*\{.*\}\s*\]', raw_text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    
    raise ValueError("Failed to parse questions into JSON. Try generating fewer questions.")

def process_document_and_generate_quiz(file_path: str, num_questions: int = 3):
    # 1. Fast Load Document (Caps processing to the first 4 pages)
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        docs = loader.load()[:4]
    else:
        loader = TextLoader(file_path, encoding='utf-8')
        docs = loader.load()

    valid_docs = [d for d in docs if d.page_content and d.page_content.strip()]
    if not valid_docs:
        raise ValueError("The uploaded document contains no readable text.")

    # 2. Fast Chunking & Concise Context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50,
        strip_whitespace=True
    )
    splits = text_splitter.split_documents(valid_docs)
    
    # Take top 2 high-density chunks to minimize compute time
    selected_chunks = splits[:2] if len(splits) > 2 else splits
    context_text = "\n\n".join([doc.page_content for doc in selected_chunks])

    # 3. Compact MCQ Generation Prompt
    prompt_message = f"""Create exactly {num_questions} multiple-choice questions from this text.

Context:
{context_text}

Respond ONLY with this JSON structure:
[
  {{
    "question": "Question text here",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": "Option A",
    "explanation": "Brief explanation"
  }}
]"""

    response = llm.invoke(prompt_message)
    return clean_json_response(response.content.strip())