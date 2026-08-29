# QuizMatrix — AI-Powered Assessment Generator

An intelligent, local assessment generation web application that automatically parses uploaded documents (PDF/TXT) and generates multiple-choice questions (MCQs) with options, correct answers, and detailed explanations.

## ✨ Features
* **Zero API Cost & 100% Private**: Runs completely on your local machine using Ollama.
* **Document Ingestion**: Efficiently extracts and chunks content from PDF and TXT files using LangChain.
* **Low-Latency Generation**: Uses optimized inference settings with structured JSON output formatting.
* **Instant Auto-Grading**: Interactive candidate assessment UI that evaluates choices, tracks scores in real time, and shows explanations.
* **Modern Cyber-Glass UI**: Responsive, dark glassmorphic design built with smooth gradients and custom interactive components.

## 🛠️ Tech Stack
* **Backend**: Python, Flask
* **AI Orchestration**: LangChain, ChromaDB
* **LLM Engine**: Ollama (`llama3.2:1b` / `llama3.2`)
* **Embeddings**: `nomic-embed-text`
* **Frontend**: HTML5, CSS3, JavaScript (Fetch API)

## ⚙️ Prerequisites
Before getting started, make sure you have installed:
1. **Python 3.8+**
2. **[Ollama](https://ollama.com/)** running locally

## 🚀 Installation & Setup

**1. Clone the repository**
```bash
