import os
from dotenv import load_dotenv
import streamlit as st
import fitz
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

# Function to extract text with inline links
def extract_text_with_inline_links(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text_lines = []

    for page in doc:
        words = page.get_text("words")
        links = page.get_links()

        link_map = [
            (fitz.Rect(link["from"]), link["uri"])
            for link in links if "uri" in link
        ]

        page_words = []
        for word in words:
            word_rect = fitz.Rect(word[:4])
            word_text = word[4]

            for link_rect, uri in link_map:
                if link_rect.intersects(word_rect):
                    word_text += f" <{uri}>"
                    break

            page_words.append(word_text)

        full_text_lines.append(" ".join(page_words))

    return "\n".join(full_text_lines)

file_path = None
if uploaded_file:
    with st.spinner("Processing resume..."):
        pdf_bytes = uploaded_file.read()
        text = extract_text_with_inline_links(pdf_bytes)

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
