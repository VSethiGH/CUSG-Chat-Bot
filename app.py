import streamlit as st
import os
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from huggingface_hub import InferenceClient
import pickle
from similarity import cosineSimilarity, euclideanSimilarity, manhattanSimilarity

HF_TOKEN = "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"
LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"        
EMBED_MODEL = "all-MiniLM-L6-v2"                        
TOP_K = 10                                    
FAISS_PATH = str(Path("index") / "faiss.index")
CHUNKS_PATH = str(Path("index") / "chunks.jsonl")

def answer(user_input, embedder, index, chunks, client):


    # Encode (like with the chunks)
    vec = embedder.encode([user_input], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    if isinstance(index, faiss.IndexFlatL2):
        _, indices = index.search(vec, TOP_K)          # (1, 384)
        top_indices = indices[0].tolist()
    else:
        _, indices = index.search(vec.squeeze(), TOP_K) # (384,)
        top_indices = indices.flatten().tolist()


    important_chunks = [chunks[i] for i in top_indices if i < len(chunks)]


    # Goes through each chunk and extract all the information
    chunks_info = []
    for i, chunk in enumerate(important_chunks):
        source = chunk.get("source_file", "")
        doc_type = chunk.get("document_type", "")
        date = chunk.get("date", "")
        content = chunk["content"]
        chunks_info.append(f"Document Type: {doc_type}, Source: {source}, Date: {date} - \n{content}")

    # All Chunks combined
    chunk_info_concat = "\n\n- Next Source -\n\n".join(chunks_info)

    # Generate a response given a list of messages
    completion  = client.chat_completion(
        model= LLM_MODEL,

        # User Information (Context)
        # then Model Prompt
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional chatbot for the Clemson University Undergraduate Student Government (CUSG). "
                    "You assist students, senators, and administrators with questions about bills, legislation, meeting minutes, rules, and procedures."
                    "Answer questions strictly using the provided context. Do not use outside knowledge or make assumptions. "
                    "If the context does not contain enough information to answer, say so clearly. Do not guess!"
                    "Be concise and clear. Summarize key information rather than quoting large blocks of text. "
                    "When relevant, reference the bill ID, document type, or date from the source. "
                )
            },

            {
                "role": "user", 
                "content": f"CONTEXT:\n{chunk_info_concat}\nQUESTION: {user_input}"
            },
        ],
    )
    return completion.choices[0].message.content

# streamlit run app.py
# Main / UI
def main():
    st.title("SenateGPT")
    st.caption("Ask questions about Clemson University Student Senate 69th Term")

    embedder = SentenceTransformer(EMBED_MODEL)
    chunks = []

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))


    client = InferenceClient(token=HF_TOKEN, provider="auto")

    index_choice = st.selectbox(
        "Select Similarity Method",
        ["FAISS (L2)", "Cosine", "Euclidean", "Manhattan"]
    )

    if index_choice == "FAISS (L2)":
        index = faiss.read_index(str(FAISS_PATH))
    elif index_choice == "Cosine":
        with open(str(Path("index") / "cosine_index.pkl"), "rb") as f:
            index = pickle.load(f)
    elif index_choice == "Euclidean":
        with open(str(Path("index") / "euclidean_index.pkl"), "rb") as f:
            index = pickle.load(f)
    elif index_choice == "Manhattan":
        with open(str(Path("index") / "manhattan_index.pkl"), "rb") as f:
            index = pickle.load(f)


    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.write(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            
            # Generate Response
            with st.spinner("Thinking..."):
                response = answer(prompt, embedder, index, chunks, client)
            st.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()