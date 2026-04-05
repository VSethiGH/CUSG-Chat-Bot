# import os
# import json
# import numpy as np
# from pathlib import Path

# import streamlit as st

# from sentence_transformers import SentenceTransformer
# import faiss
# from huggingface_hub import InferenceClient

# HF_TOKEN    = "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"
# LLM_MODEL   = "meta-llama/Meta-Llama-3-8B-Instruct"        
# EMBED_MODEL = "all-MiniLM-L6-v2"                        
# TOP_K       = 3                                     
# FAISS_PATH  = str(Path("index") / "faiss.index")
# CHUNKS_PATH = str(Path("index") / "chunks.jsonl")



# class RAGApp:
#     def __init__(self):
#         self.embedder = SentenceTransformer(EMBED_MODEL)
#         self.faiss_index = faiss.read_index(FAISS_PATH)

#         print("Loading chunks...")
#         self.chunks = []

#         with open(CHUNKS_PATH, encoding="utf-8") as f:
#             for line in f:
#                 self.chunks.append(json.loads(line))

#         self.client = InferenceClient(token=HF_TOKEN, provider="auto")

#     # Embed Question
#     def embed_query(self, question: str) -> np.ndarray:
#         vec = self.embedder.encode([question], convert_to_numpy=True)
#         return vec.astype("float32")

#     # Retrieve Relevant Chunks
#     def retrieve(self, question: str, top_k: int = TOP_K):
#         query_vec = self.embed_query(question)
#         distances, indices = self.faiss_index.search(query_vec, top_k)

#         results = []
#         for idx, dist in zip(indices[0], distances[0]):
#             if idx < len(self.chunks):
#                 chunk = self.chunks[idx].copy()
#                 chunk["_score"] = float(dist)
#                 results.append(chunk)
#         return results

#     # Call LLM
#     def answer(self, user_input: str) -> str:
#         relevant_chunks = self.retrieve(user_input)

#         context_parts = []
#         for i, chunk in enumerate(relevant_chunks, 1):
#             source = chunk.get("source_file", "unknown")
#             doc_type = chunk.get("document_type", "")
#             date = chunk.get("date", chunk.get("bill_id", ""))
#             context_parts.append(
#                 f"[Source {i}: {doc_type} | {source} | {date}]\n{chunk['content']}"
#             )
#         context = "\n\n---\n\n".join(context_parts)

#         response = self.client.chat_completion(
#             model=LLM_MODEL,
#             messages=[
#                 {"role": "system", "content": (
#                     "You are SenateGPT, an assistant for Clemson University Student Senate. "
#                     "Answer concisely using the provided context. "
#                     "Summarize agenda content rather than quoting it directly. "
#                     "If the answer is not in the context, say 'I don't have that information.'"
#                 )},
#                 {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {user_input}"}
#             ],
#             max_tokens=512,
#             temperature=0.3,
#         )
#         return response.choices[0].message.content.strip()

#     def run(self):
#         print("=" * 60)
#         print("Welcome to SenateGPT")
#         print("Type 'quit' to exit")
#         print("=" * 60 + "\n")

#         last_chunks = []

#         while True:
#             try:
#                 user_input = input("You: ").strip()
#             except (EOFError, KeyboardInterrupt):
#                 print("\nGoodbye!")
#                 break

#             if not user_input:
#                 continue

#             if user_input.lower() == "quit":
#                 print("Goodbye!")
#                 break

#             last_chunks = self.retrieve(user_input)

#             answer = self.answer(user_input)
#             print(f"\nAssistant: {answer}\n")


# def main():
#     app = RAGApp()
#     app.run()


# if __name__ == "__main__":
#     main()

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
                "role": "user", 
                "content": f"CONTEXT:\n{chunk_info_concat}\nQUESTION: {user_input}"
            },

            {
                "role": "system",
                "content": (
                    "You are a helpful and professional chatbot for Clemson University Undergraduate Student Government (CUSG). "
                    "You assist students, senators, and administrators with questions about bills, legislation, meeting minutes, rules, and procedures. "
                    "Answer questions strictly using the provided context. Do not use outside knowledge or make assumptions. "
                    "If the context does not contain enough information to answer, say so clearly — do not guess. "
                    "Be concise and clear. Summarize key information rather than quoting large blocks of text. "
                    "When relevant, reference the bill ID, document type, or date from the source. "
                )
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