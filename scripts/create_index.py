import argparse
import pandas as pd
import numpy as np
import faiss
import os
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("create_index")

def create_faiss_index(
    data_path,
    index_path,
    embeddings_model='all-MiniLM-L6-v2',
    text_column='abstract',
    batch_size=64
):
    """
    Create a FAISS index from text data.
    
    Args:
        data_path: Path to the CSV file containing the data
        index_path: Path to save the FAISS index
        embeddings_model: Name of the sentence transformer model to use
        text_column: Name of the column containing the text to index
        batch_size: Batch size for embedding generation
    """
    start_time = time.time()
    logger.info(f"Loading data from {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in data file")
    
    # Filter out rows with missing or empty text
    df = df.dropna(subset=[text_column]).query(f"{text_column}.str.strip().str.len() > 0")
    
    logger.info(f"Loaded {len(df)} documents")
    
    # Initialize model
    logger.info(f"Loading embedding model: {embeddings_model}")
    model = SentenceTransformer(embeddings_model)
    
    # Get dimension of embeddings
    test_embedding = model.encode(["test"])
    dimension = test_embedding.shape[1]
    logger.info(f"Embedding dimension: {dimension}")
    
    # Create FAISS index - use IVF for faster search on large datasets
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dimension), dimension, min(100, len(df)//100))
    
    # Process data in batches
    total_batches = (len(df) + batch_size - 1) // batch_size
    all_texts = df[text_column].tolist()
    training_embeddings = model.encode(all_texts[:min(10000, len(all_texts))], normalize_embeddings=True).astype('float32')
    index.train(training_embeddings)
    
    logger.info("Generating embeddings and adding to index")
    
    for i in tqdm(range(0, len(df), batch_size), total=total_batches, desc="Creating embeddings"):
        # Get batch of texts
        batch_texts = all_texts[i:i+batch_size]
        
        # Generate embeddings and normalize in one step
        embeddings = model.encode(batch_texts, show_progress_bar=False, normalize_embeddings=True).astype('float32')
        
        # Add to index
        index.add(embeddings)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
    
    # Save index
    logger.info(f"Saving index to {index_path}")
    faiss.write_index(index, index_path)
    
    # Save metadata
    metadata = {
        "data_path": data_path,
        "embeddings_model": embeddings_model,
        "text_column": text_column,
        "dimension": dimension,
        "num_vectors": index.ntotal,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.splitext(index_path)[0] + "_metadata.txt", 'w') as f:
        f.write('\n'.join(f"{k}: {v}" for k, v in metadata.items()))
    
    elapsed_time = time.time() - start_time
    logger.info(f"Index created successfully with {index.ntotal} vectors")
    logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
    
    return index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a FAISS index from text data")
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to the CSV file containing the data")
    parser.add_argument("--index_path", type=str, required=True, 
                        help="Path to save the FAISS index")
    parser.add_argument("--embeddings_model", type=str, default="all-MiniLM-L6-v2", 
                        help="Name of the sentence transformer model to use")
    parser.add_argument("--text_column", type=str, default="abstract", 
                        help="Name of the column containing the text to index")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Batch size for embedding generation")
    
    args = parser.parse_args()
    
    create_faiss_index(
        data_path=args.data_path,
        index_path=args.index_path,
        embeddings_model=args.embeddings_model,
        text_column=args.text_column,
        batch_size=args.batch_size
    ) 