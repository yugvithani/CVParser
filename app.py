import os
from dotenv import load_dotenv
import streamlit as st
import fitz
from llama_index.readers.file import PyMuPDFReader
from llama_index.embeddings.fireworks import FireworksEmbedding
from llama_index.llms.fireworks import Fireworks
from llama_index.core import VectorStoreIndex, Document

# Load env variables
load_dotenv(override=True)
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY").strip()
FIREWORKS_MODEL = os.getenv("FIREWORKS_MODEL").strip()

st.set_page_config(page_title="Resume Q&A", layout="centered")
st.title("ParserCV")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

file_path = None
if uploaded_file:
    # Save uploaded file to local 'resumes' directory
    # resumes_dir = "resumes"
    # os.makedirs(resumes_dir, exist_ok=True)
    # file_path = os.path.join(resumes_dir, uploaded_file.name)
    # with open(file_path, "wb") as f:
    #     f.write(uploaded_file.getbuffer())

    with st.spinner("Processing resume..."):
        # # Load PDF
        # loader = PyMuPDFReader()
        # documents = loader.load(file_path=file_path)  # use saved file path
        
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        # Prepare document for LlamaIndex (as a list of dicts)
        documents = [Document(text=text)]

        # Setup embedding and LLM
        embed_model = FireworksEmbedding(api_key=FIREWORKS_API_KEY, embed_batch_size=10)
        llm = Fireworks(model=FIREWORKS_MODEL, api_key=FIREWORKS_API_KEY)

        # Build index
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model, llm=llm)
        query_engine = index.as_query_engine(llm=llm)
        st.success("Resume indexed. Ask a question below!")

        # Chat interface
        user_question = st.text_input("Ask a question about the resume:")
        if user_question:
            with st.spinner("Thinking..."):
                response = query_engine.query(user_question)
                st.markdown(f"**Answer:** {response}")
