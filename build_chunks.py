import os
import json
from pathlib import Path
from docx import Document
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class DocumentIndexBuilder:
    def __init__(self, data_dir="data", output_dir="index", chunk_size=1500, overlap=150):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size  
        self.overlap = overlap       
        self.output_dir.mkdir(exist_ok=True)
        
    # Extract Date 
    def extract_date_from_filename(self, filename):
        fileDate = filename.split()[2].replace(".docx", "")     # Read File name (2nd index is date and remove .docx)
        return fileDate
    
    # Extract bill info
    def extract_bill_info(self, filename):
        parts = filename.replace(".docx", "").split("_")
        session = parts[0]
        bill_type = parts[1] 
        bill_number = parts[2]
        return f"{session}_{bill_type}_{bill_number}"
    
    # Extract all file content into one string
    def extract_text_from_docx(self, file_path):
        doc = Document(file_path)
        full_text = []
        # Reads each paragraph
        for para in doc.paragraphs:
            if para.text.strip():   # Makes sure there is content
                full_text.append(para.text.strip()) 
        return '\n'.join(full_text)
    
    # Creates varying chunks
    def create_chunks(self, text):
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        index = 0       
        
        # Go through each index
        while index < len(text):
            end = index + self.chunk_size
            
            # If at the end of the index
            if end >= len(text):
                chunks.append(text[index:])
                break
            
            # Find last space to avoid breaking words
            last_space = text[index:end].rfind(' ')
            if last_space > 0:
                end = index + last_space   
            
            # New Chunk
            chunks.append(text[index:end])
            index = end - self.overlap      # Account for overlap
            
        return chunks
    
    def process_agenda_file(self, file_path):
        print("Processing: ", file_path.name)
        
        date = self.extract_date_from_filename(file_path.name)
        full_text = self.extract_text_from_docx(file_path)
        text_chunks = self.create_chunks(full_text)
        
        # Format chunks for JSONL with metadata
        formatted_chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_data = {
                'id': f"{date}_{i:03d}",
                'source_file': file_path.name,
                'date': date,
                'document_type': 'senate_agenda',
                'content': chunk_text.strip(),
                'chunk_index': i,
                'total_chunks': len(text_chunks),
                'chunk_size': len(chunk_text)
            }
            formatted_chunks.append(chunk_data)
        
        return formatted_chunks
    
    def process_bill_file(self, file_path):
        print("Processing: ", file_path.name)
        
        bill_id = self.extract_bill_info(file_path.name)
        full_text = self.extract_text_from_docx(file_path)
        text_chunks = self.create_chunks(full_text)
        
        formatted_chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_data = {
                'id': f"bill_{bill_id}_{i:03d}",
                'source_file': file_path.name,
                'bill_id': bill_id,
                'document_type': 'senate_bill',
                'content': chunk_text.strip(),
                'chunk_index': i,
                'total_chunks': len(text_chunks),
                'chunk_size': len(chunk_text)
            }
            formatted_chunks.append(chunk_data)
        
        return formatted_chunks
    
    def process_minutes_file(self, file_path):
        print("Processing: ", file_path.name)
        
        date = self.extract_date_from_filename(file_path.name)
        full_text = self.extract_text_from_docx(file_path)
        text_chunks = self.create_chunks(full_text)
        
        formatted_chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_data = {
                'id': f"minutes_{date}_{i:03d}",
                'source_file': file_path.name,
                'date': date,
                'document_type': 'senate_minutes',
                'content': chunk_text.strip(),
                'chunk_index': i,
                'total_chunks': len(text_chunks),
                'chunk_size': len(chunk_text)
            }
            formatted_chunks.append(chunk_data)
        
        return formatted_chunks
    
    def process_bylaws_file(self, file_path):
        print("Processing: ", file_path.name)
        
        file_id = file_path.stem.replace(" ", "_").lower()
        full_text = self.extract_text_from_docx(file_path)
        text_chunks = self.create_chunks(full_text)
        
        formatted_chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_data = {
                'id': f"bylaws_{file_id}_{i:03d}",
                'source_file': file_path.name,
                'document_type': 'bylaws',
                'content': chunk_text.strip(),
                'chunk_index': i,
                'total_chunks': len(text_chunks),
                'chunk_size': len(chunk_text)
            }
            formatted_chunks.append(chunk_data)
        
        return formatted_chunks
    
    def build_index(self):
        all_chunks = []
        
        # Process agenda files
        agenda_dir = self.data_dir / "agenda"
        if agenda_dir.exists():
            for file_path in agenda_dir.glob("*.docx"):
                chunks = self.process_agenda_file(file_path)
                all_chunks.extend(chunks)
        
        # Process bill files  
        bills_dir = self.data_dir / "bills"
        if bills_dir.exists():
            for file_path in bills_dir.glob("*.docx"):
                chunks = self.process_bill_file(file_path)
                all_chunks.extend(chunks)
        
        # Process minutes files
        minutes_dir = self.data_dir / "minutes"
        if minutes_dir.exists():
            for file_path in minutes_dir.glob("*.docx"):
                chunks = self.process_minutes_file(file_path)
                all_chunks.extend(chunks)
        
        # Process bylaws files
        bylaws_dir = self.data_dir / "bylaws"
        if bylaws_dir.exists():
            for file_path in bylaws_dir.glob("*.docx"):
                chunks = self.process_bylaws_file(file_path)
                all_chunks.extend(chunks)
        
        # Write to JSONL file
        output_file = self.output_dir / "chunks.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        
        print(f"\nProcessed {len(all_chunks)} total chunks")
        print(f"Index saved to: {output_file}")


        print(f"Embedding Chunks...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        texts = [chunk["content"] for chunk in all_chunks]
        embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.array(embeddings).astype("float32")
        dim = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dim)
        faiss_index.add(embeddings)
        faiss.write_index(faiss_index, str(self.output_dir / "faiss.index"))
        return all_chunks

def main():
    builder = DocumentIndexBuilder(chunk_size=1000, overlap=250)
    
    print("Building document index...")
    chunks = builder.build_index()


if __name__ == "__main__":
    main()
