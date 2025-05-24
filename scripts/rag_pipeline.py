import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import requests
import os
import time
import json
from typing import List, Tuple, Dict, Optional, Union, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import pickle
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("sciquery")

@dataclass
class Document:
    """Class representing a scientific document with its metadata."""
    content: str
    metadata: Dict[str, Any]
    similarity: float = 0.0
    
    def __str__(self) -> str:
        return f"{self.content[:100]}... (similarity: {self.similarity:.4f})"

@dataclass
class QueryResult:
    """Class representing the result of a query."""
    answer: str
    documents: List[Document]
    confidence: float
    query_time: float
    
    def formatted_answer(self, include_citations: bool = True) -> str:
        """Format the answer with citations if requested."""
        if not include_citations:
            return self.answer
            
        answer = self.answer
        # Add citation footer
        answer += "\n\nSources:\n"
        for i, doc in enumerate(self.documents):
            answer += f"[{i+1}] {doc.metadata.get('title', 'Unknown')} "
            answer += f"({doc.metadata.get('date', 'Unknown')}) - {doc.metadata.get('arxiv_id', 'Unknown')}\n"
        
        return answer

class SciQueryRAG:
    """
    Retrieval-Augmented Generation system for scientific papers.
    
    This class implements a RAG pipeline that retrieves relevant scientific papers
    based on a query and uses an LLM to generate an answer.
    """
    
    def __init__(
        self,
        embeddings_model: str = 'all-MiniLM-L6-v2',
        llm_model: str = 'deepseek/deepseek-v3-0324',
        data_path: str = 'data/arxiv_papers_cs.AI.csv',
        index_path: str = 'data/sciquery_index.faiss',
        cache_dir: str = 'cache',
        hf_token: Optional[str] = None,
        use_cache: bool = True,
    ):
        """
        Initialize the SciQueryRAG system.
        
        Args:
            embeddings_model: Name of the sentence transformer model to use
            llm_model: Name of the LLM model to use for generation
            data_path: Path to the CSV file containing the papers
            index_path: Path to the FAISS index
            cache_dir: Directory to store cache files
            hf_token: HuggingFace API token (defaults to HF_TOKEN env variable)
            use_cache: Whether to use caching for queries
        """
        self.embeddings_model_name = embeddings_model
        self.llm_model_name = llm_model
        self.data_path = data_path
        self.index_path = index_path
        self.cache_dir = Path(cache_dir)
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.use_cache = use_cache
        
        # Validate token
        if not self.hf_token:
            logger.warning("No HuggingFace token found. Set HF_TOKEN environment variable or pass token to constructor.")
        
        # Create cache directory if needed
        if self.use_cache:
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            self.cache_file = self.cache_dir / "query_cache.pkl"
            self.load_cache()
        
        # Load components
        self._load_components()
    
    def _load_components(self):
        """Load all required components (model, index, data)."""
        try:
            logger.info(f"Loading embedding model: {self.embeddings_model_name}")
            self.embeddings_model = SentenceTransformer(self.embeddings_model_name)
            
            logger.info(f"Loading FAISS index from: {self.index_path}")
            self.index = faiss.read_index(self.index_path)
            
            logger.info(f"Loading paper data from: {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            
            # Check if there are required columns
            required_cols = ['abstract', 'title']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                logger.warning(f"Missing columns in data: {missing_cols}")

        except Exception as e:
            logger.error(f"Error loading components: {str(e)}")
            raise RuntimeError(f"Failed to initialize SciQueryRAG: {str(e)}")
    
    def load_cache(self):
        """Load query cache from disk."""
        if hasattr(self, 'cache') or not self.use_cache:
            return
            
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"Loaded {len(self.cache)} cached queries")
            except Exception as e:
                logger.warning(f"Failed to load cache: {str(e)}")
                self.cache = {}
        else:
            self.cache = {}
    
    def save_cache(self):
        """Save query cache to disk."""
        if not hasattr(self, 'cache') or not self.use_cache:
            return
            
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            logger.info(f"Saved {len(self.cache)} queries to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {str(e)}")
    
    def retrieve(
        self, 
        query: str, 
        k: int = 5, 
        similarity_threshold: float = 0.4
    ) -> List[Document]:
        """
        Retrieve relevant documents based on a query.
        
        Args:
            query: The search query
            k: Maximum number of results to retrieve
            similarity_threshold: Minimum similarity score to include a document
            
        Returns:
            List of Document objects with content and metadata
        """
        if not query or not isinstance(query, str):
            logger.error("Query must be a non-empty string")
            return []

        try:
            # Encode query
            query_embedding = self.embeddings_model.encode([query]).astype('float32')
            
            # Search index
            distances, indices = self.index.search(query_embedding, k)
            
            results = []
            if indices.size > 0:
                for i, dist in enumerate(distances[0]):
                    if indices[0][i] == -1:
                        continue
                        
                    # Calculate similarity from L2 distance
                    similarity = max(0.0, min(1.0, 1.0 - (dist**2) / 2.0))
                    
                    if similarity >= similarity_threshold:
                        doc_idx = indices[0][i]
                        paper_row = self.df.iloc[doc_idx]
                        
                        # Get content and metadata
                        content = paper_row.get('abstract', '')
                        
                        # Extract all available metadata
                        metadata = {col: paper_row.get(col, '') for col in self.df.columns 
                                   if col != 'abstract' and not pd.isna(paper_row.get(col, ''))}
                        
                        doc = Document(
                            content=content,
                            metadata=metadata,
                            similarity=similarity
                        )
                        results.append(doc)
                        
            return results
            
        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}")
            return []

    def compute_confidence(self, documents: List[Document]) -> float:
        """
        Compute confidence score based on retrieved documents.
        
        Args:
            documents: List of retrieved Document objects
            
        Returns:
            Confidence score (0-100)
        """
        if not documents:
            return 0.0
            
        # Extract similarities and weight by position
        similarities = [doc.similarity for doc in documents]
        
        # Apply position-based weighting (higher weight to top results)
        weights = np.linspace(1.0, 0.5, len(similarities))
        weighted_similarities = np.multiply(similarities, weights)
        
        # Calculate weighted average similarity
        weighted_avg = np.sum(weighted_similarities) / np.sum(weights)
        return weighted_avg * 100.0

    def generate_answer(self, context: str, query: str) -> str:
        """
        Generate an answer using the LLM based on context and query.
        
        Args:
            context: The context from retrieved documents
            query: The original user query
            
        Returns:
            Generated answer
        """
        if not self.hf_token:
            return "Error: No HuggingFace token available for API access."
            
        API_URL = "https://router.huggingface.co/novita/v3/openai/chat/completions"
        headers = {"Authorization": f"Bearer {self.hf_token}", "Content-Type": "application/json"}
        
        system_prompt = (
            "You are SciQuery, a scientific research assistant specialized in AI research. "
            "Answer the user's question based only on the provided context. "
            "If the context doesn't contain relevant information to answer the question, "
            "say that you don't have enough information. "
            "Provide accurate, well-reasoned answers with clear explanations. "
            "Use an academic, objective tone."
        )
        
        prompt = f"Based on the following research paper extracts, please answer the question: {query}"

        try:
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                    {
                        "role": "assistant",
                        "content": "I'll help answer this based on the research papers."
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}"
                    }
                ],
                "model": self.llm_model_name,
                "temperature": 0.3,  # Lower temperature for more factual responses
                "max_tokens": 1024,  # Adjust based on response needs
            }
            
            # Send request to API
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            return response.json()["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return f"Error generating answer: {str(e)}"
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Error parsing API response: {str(e)}")
            return "Error: Failed to parse the model's response."
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return "An unexpected error occurred while generating the answer."

    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a context string for the LLM.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(documents):
            # Format with metadata
            context_part = f"[DOCUMENT {i+1}]\n"
            context_part += f"Title: {doc.metadata.get('title', 'Unknown')}\n"
            if 'authors' in doc.metadata:
                context_part += f"Authors: {doc.metadata.get('authors', 'Unknown')}\n"
            if 'date' in doc.metadata:
                context_part += f"Date: {doc.metadata.get('date', 'Unknown')}\n"
            context_part += f"Content: {doc.content}\n"
            
            context_parts.append(context_part)
            
        return "\n".join(context_parts)

    def query(
        self, 
        query: str, 
        k: int = 5,
        similarity_threshold: float = 0.4,
        use_cache: Optional[bool] = None
    ) -> QueryResult:
        """
        Process a query through the full RAG pipeline.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            similarity_threshold: Minimum similarity for retrieval
            use_cache: Whether to use cached results (defaults to instance setting)
            
        Returns:
            QueryResult containing answer, documents, confidence, and timing info
        """
        start_time = time.time()
        
        # Normalize query
        query = query.strip()
        
        # Check cache if enabled
        use_cache = self.use_cache if use_cache is None else use_cache
        cache_key = f"{query}_{k}_{similarity_threshold}"
        
        if use_cache and hasattr(self, 'cache') and cache_key in self.cache:
            logger.info(f"Using cached result for query: {query}")
            result = self.cache[cache_key]
            # Update query time
            result.query_time = time.time() - start_time
            return result
        
        # Retrieve relevant documents
        documents = self.retrieve(query, k, similarity_threshold)
        
        if not documents:
            logger.warning(f"No documents found for query: {query}")
            return QueryResult(
                answer="I couldn't find relevant research papers to answer your question. Please try rephrasing or asking a different question.",
                documents=[],
                confidence=0.0,
                query_time=time.time() - start_time
            )
        
        # Format context from documents
        context = self.format_context(documents)
        
        # Generate answer
        answer = self.generate_answer(context, query)
        
        # Compute confidence
        confidence = self.compute_confidence(documents)
        
        # Create result
        result = QueryResult(
            answer=answer,
            documents=documents,
            confidence=confidence,
            query_time=time.time() - start_time
        )
        
        # Cache result if caching is enabled
        if use_cache and hasattr(self, 'cache'):
            self.cache[cache_key] = result
            # Periodically save cache
            if len(self.cache) % 10 == 0:
                self.save_cache()
        
        return result

# Example usage
if __name__ == "__main__":
    # Initialize the RAG system
    rag = SciQueryRAG(
        embeddings_model='all-MiniLM-L6-v2',
        llm_model='deepseek/deepseek-v3-0324',
        data_path='data/arxiv_papers_cs.AI.csv',
        index_path='data/sciquery_index.faiss'
    )
    
    # Process a query
    query = "What are the latest advances in neural network optimization?"
    result = rag.query(query)
    
    # Print results
    print(f"Query: {query}")
    print(f"Confidence: {result.confidence:.2f}%")
    print(f"Response time: {result.query_time:.2f}s")
    print("\nAnswer:")
    print(result.formatted_answer(include_citations=True))