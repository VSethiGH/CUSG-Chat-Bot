import os
import json
from pathlib import Path
from docx import Document
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


# Read each Document, and put each paragraph into the array
# Returns huge string of the entire document
def docx_to_text(file_path):
    doc = Document(file_path)
    full_text = []
    # Reads each paragraph
    for para in doc.paragraphs:
        if para.text.strip():                    # Makes sure there is content
            full_text.append(para.text.strip()) 
    return '\n'.join(full_text)                  # Combnes paragraph in a long string


# Creates Chunk based on above string
def create_chunks(text):
    chunks = []
    index = 0       
    text_len = len(text)

    # If the text is smaller than the chunk_size
    if text_len <= CHUNK_SIZE:
        return [text]

    while index < text_len:
        end = min(index + CHUNK_SIZE, text_len)       # text cannot be above length of text
        
        #  Completes the current sentence
        finish_sentence = text[index:end].rfind('.')
        if finish_sentence > 0:
            end = index + finish_sentence + 1
            end = min(end, text_len) 

        chunks.append(text[index:end])
        
        # Reached last chunk
        if end >= text_len:
            break

        next_index = end - OVERLAP
        if next_index <= index: 
            next_index = end
        index = next_index
        
    return chunks

def process_agenda_file(file_path):
    print("Processing: ", file_path)
    
    # Get the Date (last bit of content)
    # File name = data/agenda/Senate Agenda 1-12-2026.docx
    # 2nd index = 1-12-2026.docx (rempve the .docx)
    filename = Path(file_path).name
    date = filename.split()[2].replace(".docx", "")

    full_text = docx_to_text(file_path)
    chunks = create_chunks(full_text)
    
    formatted_chunks = []
    for i, chunk_text in enumerate(chunks):
        chunk_data = {
            'id': f"agenda_{date}_{i}",
            'source_file': str(file_path),
            'date': date,
            'document_type': 'senate_agenda',
            'content': chunk_text.strip(),
            'chunk_index': i,
            'total_chunks': len(chunks),
            'chunk_size': len(chunk_text)
        }
        formatted_chunks.append(chunk_data)
    
    return formatted_chunks

def process_bill_file(file_path):
    print("Processing: ", file_path)
    

    # Example 70_SB_03.docx
    # Example <senate term>_<bill_type>_<bill number>
    filename = Path(file_path).name
    file = filename.replace(".docx", "").split("_")
    senate_term = file[0]
    bill_type = file[1] 
    bill_number = file[2]

    full_text = docx_to_text(file_path)
    chunks = create_chunks(full_text)
    
    formatted_chunks = []
    for i, chunk_text in enumerate(chunks):
        chunk_data = {
            'id': f"{senate_term}_{bill_type}_{bill_number}_{i}",
            'source_file': str(file_path),
            'bill_id': bill_number,
            'document_type': f"{bill_type}_bill",
            'content': chunk_text.strip(),
            'chunk_index': i,
            'total_chunks': len(chunks),
            'chunk_size': len(chunk_text)
        }
        formatted_chunks.append(chunk_data)
    
    return formatted_chunks

def process_minutes_file(file_path):
    print("Processing: ", file_path)
    
    # Get the Date (last bit of content)
    # File name = data/agenda/Senate Agenda 1-12-2026.docx
    # 2nd index = 1-12-2026.docx (rempve the .docx)
    filename = Path(file_path).name
    date = filename.split()[2].replace(".docx", "")

    full_text = docx_to_text(file_path)
    chunks = create_chunks(full_text)
    
    formatted_chunks = []
    for i, chunk_text in enumerate(chunks):
        chunk_data = {
            'id': f"minutes_{date}_{i}",
            'source_file': str(file_path),
            'date': date,
            'document_type': 'senate_minutes',
            'content': chunk_text.strip(),
            'chunk_index': i,
            'total_chunks': len(chunks),
            'chunk_size': len(chunk_text)
        }
        formatted_chunks.append(chunk_data)
    
    return formatted_chunks

def process_bylaws_file(file_path):
    print("Processing: ", file_path)
    
    file_id = file_path.stem.replace(" ", "_").lower()
    full_text = docx_to_text(file_path)
    chunks = create_chunks(full_text)
    
    formatted_chunks = []
    for i, chunk_text in enumerate(chunks):
        chunk_data = {
            'id': f"bylaws_{file_id}_{i}",
            'source_file': str(file_path),
            # No Date
            'document_type': 'bylaws',
            'content': chunk_text.strip(),
            'chunk_index': i,
            'total_chunks': len(chunks),
            'chunk_size': len(chunk_text)
        }
        formatted_chunks.append(chunk_data)
    
    return formatted_chunks


DATA_DIR = Path("data")
OUTPUT_DIR = Path("index")
CHUNK_SIZE = 1000
OVERLAP = 250

def main():

    # Create Output Directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    all_chunks = []
    
    # Process agenda files
    agenda_dir = DATA_DIR / "agenda"
    if agenda_dir.exists():
        for file_path in agenda_dir.glob("*.docx"):
            chunks = process_agenda_file(file_path)
            all_chunks.extend(chunks)
    
    # Process bill files  
    bills_dir = DATA_DIR / "bills"
    if bills_dir.exists():
        for file_path in bills_dir.glob("*.docx"):
            chunks = process_bill_file(file_path)
            all_chunks.extend(chunks)
    
    # Process minutes files
    minutes_dir = DATA_DIR / "minutes"
    if minutes_dir.exists():
        for file_path in minutes_dir.glob("*.docx"):
            chunks = process_minutes_file(file_path)
            all_chunks.extend(chunks)
    
    # Process bylaws files
    bylaws_dir = DATA_DIR / "bylaws"
    if bylaws_dir.exists():
        for file_path in bylaws_dir.glob("*.docx"):
            chunks = process_bylaws_file(file_path)
            all_chunks.extend(chunks)
    
    # Write to JSONL file
    output_file = OUTPUT_DIR / "chunks.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    print(f"Processed {len(all_chunks)} chunks")
    print(f"Index saved to: {output_file}")


    print(f"Embedding Chunks...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [chunk["content"] for chunk in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, str(OUTPUT_DIR / "faiss.index"))

if __name__ == "__main__":
    main()
