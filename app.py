import os
import gradio as gr
from scripts.rag_pipeline import SciQueryRAG

# Initialize the RAG system
rag = SciQueryRAG(
    embeddings_model='all-MiniLM-L6-v2',
    llm_model='deepseek/deepseek-v3-0324',
    data_path='data/arxiv_papers_cs.AI.csv',
    index_path='data/sciquery_index.faiss',
    use_cache=True
)

def process_query(query, include_citations, k, similarity_threshold):
    try:
        if not query or not query.strip():
            return "Please enter a question to continue.", "", 0, []
        
        k = max(1, int(k)) if k >= 1 else 5
        similarity_threshold = max(0, min(1, float(similarity_threshold))) if 0 <= similarity_threshold <= 1 else 0.4
            
        result = rag.query(query=query, k=k, similarity_threshold=similarity_threshold)
        answer = result.formatted_answer(include_citations=include_citations)
        
        sources = [[
            doc.metadata.get('title', 'Unknown'),
            doc.metadata.get('authors', 'Unknown'), 
            doc.metadata.get('date', 'Unknown'),
            doc.metadata.get('arxiv_id', 'Unknown'),
            f"{doc.similarity:.2f}"
        ] for doc in result.documents]
            
        return answer, f"{result.confidence:.1f}%", result.query_time, sources
        
    except Exception as e:
        return f"Error: {str(e)}", "0.0%", 0, []

def build_interface():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as interface:
        gr.Markdown(
            """
            # SciQuery: AI Research Assistant
            
            Ask questions about AI research and get answers backed by scientific papers.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=4):
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., What are the latest advances in neural network optimization?",
                    lines=2
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    clear_btn = gr.Button("Clear")
                
                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        include_citations = gr.Checkbox(
                            label="Include Citations",
                            value=True,
                            info="Show paper references at the end of the answer"
                        )
                        k_value = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Number of Papers",
                            info="How many relevant papers to retrieve"
                        )
                        similarity_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.4,
                            step=0.05,
                            label="Similarity Threshold",
                            info="Minimum relevance score (higher = more strict)"
                        )
            
            with gr.Column(scale=1):
                confidence_output = gr.Textbox(label="Confidence Score")
                time_output = gr.Number(label="Response Time (seconds)")
                
        answer_output = gr.Markdown(label="Answer")
        
        sources_output = gr.Dataframe(
            headers=["Title", "Authors", "Date", "arXiv ID", "Similarity"],
            label="Source Papers",
            visible=True,
            wrap=True  # Allow text wrapping for better display
        )
        
        # Examples
        gr.Examples([
            ["What are the latest advances in neural network optimization?"],
            ["How does deep reinforcement learning work?"],
            ["Explain the transformer architecture in simple terms"],
            ["What are some applications of generative adversarial networks?"],
            ["Compare and contrast supervised and unsupervised learning"]
        ], inputs=query_input)
        
        # Set up event handlers
        inputs = [query_input, include_citations, k_value, similarity_threshold]
        outputs = [answer_output, confidence_output, time_output, sources_output]
        
        submit_btn.click(process_query, inputs=inputs, outputs=outputs)
        query_input.submit(process_query, inputs=inputs, outputs=outputs)
        clear_btn.click(lambda: ("", "", 0, []), outputs=outputs)
                
        gr.Markdown(
            """
            ### About SciQuery
            
            SciQuery uses Retrieval-Augmented Generation (RAG) to find relevant AI research papers 
            and generate accurate answers to your questions. The system retrieves papers from arXiv 
            and uses a language model to synthesize information from them.
            
            **Note**: This is a research tool and may not always provide complete or accurate information.
            """
        )
        
    return interface

if __name__ == "__main__":
    build_interface().launch(share=False)
