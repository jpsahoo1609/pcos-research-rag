"""
Run this ONCE to fetch papers, chunk, embed, and store in Supabase.
Usage: python ingest.py
"""
import os
from dotenv import load_dotenv
load_dotenv()

from src.fetch_papers import download_papers
from src.chunker import load_and_chunk
from src.vectorstore import store_chunks

print("Step 1/3: Fetching papers from PubMed...")
papers = download_papers(disease="PCOS", max_papers_per_query=10)

print("\nStep 2/3: Chunking documents...")
chunks = load_and_chunk()

print("\nStep 3/3: Embedding + storing in Supabase...")
store_chunks(chunks)

print("\n✅ Ingestion complete! Run: streamlit run app.py")
