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

HF_TOKEN    = "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"
LLM_MODEL   = "meta-llama/Meta-Llama-3-8B-Instruct"        
EMBED_MODEL = "all-MiniLM-L6-v2"                        
TOP_K       = 5                                     
FAISS_PATH  = str(Path("index") / "faiss.index")
CHUNKS_PATH = str(Path("index") / "chunks.jsonl")


@st.cache_resource
def load_app():
    embedder = SentenceTransformer(EMBED_MODEL)
    faiss_index = faiss.read_index(FAISS_PATH)
    chunks = []

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    client = InferenceClient(token=HF_TOKEN, provider="auto")
    return embedder, faiss_index, chunks, client


def retrieve(question, embedder, faiss_index, chunks, top_k=TOP_K):
    vec = embedder.encode([question], convert_to_numpy=True).astype("float32")
    _, indices = faiss_index.search(vec, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


def answer(user_input, embedder, faiss_index, chunks, client):
    relevant_chunks = retrieve(user_input, embedder, faiss_index, chunks)

    context_parts = []
    for i, chunk in enumerate(relevant_chunks, 1):
        source = chunk.get("source_file", "unknown")
        doc_type = chunk.get("document_type", "")
        date = chunk.get("date", chunk.get("bill_id", ""))
        text = chunk["content"][:1000]
        context_parts.append(f"[Source {i}: {doc_type} | {source} | {date}]\n{text}")
    context = "\n\n---\n\n".join(context_parts)

    response = client.chat_completion(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": (
                "You are SenateGPT, an assistant for Clemson University Student Senate. "
                "Answer concisely using the provided context. "
                "Summarize agenda content rather than quoting it directly. "
                "If the answer is not in the context, say 'I don't have that information.'"
            )},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {user_input}"}
        ],
        max_tokens=512,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# Main / UI
st.title("SenateGPT")
st.caption("Ask questions about Clemson University Student Senate 69th Term")

embedder, faiss_index, chunks, client = load_app()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = answer(prompt, embedder, faiss_index, chunks, client)
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})