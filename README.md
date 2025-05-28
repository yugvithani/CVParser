# CVParser 

A helpful Streamlit web app that allows users to upload a resume (PDF), processes the content (including inline hyperlinks), and enables question-answering over the resume using LLMs and vector embeddings.

## Process

- 🗂 Upload PDF resume
- 🔍 Extracts visible text and clickable URLs
- 🤖 Uses AI for LLM-based question answering
- 🧠 Embedding powered for fast semantic search
- 💬 Ask natural language questions about the resume content

## Tech Stack

- Python
- Streamlit
- LlamaIndex
- Fireworks AI (LLM + Embeddings)
- PyMuPDF (fitz)

## Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/yugvithani/CVParser.git
cd CVParser
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Environment Variables**

```bash
FIREWORKS_API_KEY=your_fireworks_api_key
FIREWORKS_MODEL=your_model_url_from_fireork
```

4. **Run the App**

```bash
streamlit run app.py
```

## Example

Upload a resume like:

> **John Doe**  
> GitHub [john-doe](https://github.com/john-doe)  
> Skills: Python, ML, Data Analysis

Then ask questions like:

- “What is the candidate’s GitHub?”
- “What are the listed skills?”
- “Where did the candidate study?”