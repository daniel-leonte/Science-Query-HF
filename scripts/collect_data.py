import arxiv
import pandas as pd
import os
import re
import argparse
from tqdm import tqdm

def clean_text(text):
    """Basic text cleaning for abstracts and titles"""
    if not text:
        return ""
    # Remove extra whitespace and LaTeX/special characters in one pass
    text = re.sub(r'\\[a-zA-Z]+|\s+', ' ', text)
    return text.strip()

def collect_arxiv_data(categories=["cs.AI"], max_results=500, output_dir="data"):
    """
    Collect papers from arXiv for specified categories
    
    Args:
        categories (list): List of arXiv categories to collect papers from
        max_results (int): Maximum number of papers to collect per category
        output_dir (str): Directory to save the collected data
    
    Returns:
        DataFrame containing the collected papers
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_papers = []
    
    for category in categories:
        print(f"Collecting papers for category: {category}")
        
        client = arxiv.Client(page_size=100, delay_seconds=1.0)
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        for result in tqdm(client.results(search), total=max_results, desc=f"Collecting {category}"):
            try:
                paper_data = {
                    'title': clean_text(result.title),
                    'abstract': clean_text(result.summary),
                    'date': result.published,
                    'authors': ', '.join(author.name for author in result.authors),
                    'arxiv_id': result.entry_id.split('/')[-1],
                    'pdf_url': result.pdf_url,
                    'primary_category': result.primary_category,
                    'categories': ', '.join(result.categories),
                    'doi': getattr(result, 'doi', None),
                    'journal_ref': getattr(result, 'journal_ref', None)
                }
                
                if paper_data['title'] and paper_data['abstract']:
                    all_papers.append(paper_data)
            except (AttributeError, TypeError) as e:
                print(f"Error processing paper: {e}")
                continue
    
    df = pd.DataFrame(all_papers).drop_duplicates(subset=['arxiv_id'])
    
    df['abstract_word_count'] = df['abstract'].str.split().str.len().fillna(0)
    
    output_file = os.path.join(output_dir, f"arxiv_papers_{'_'.join(categories)}.csv")
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} papers to {output_file}")
    
    preview_df = df[['title', 'abstract', 'date', 'authors', 'arxiv_id', 'primary_category']]
    preview_df.to_csv(os.path.join(output_dir, f"arxiv_papers_{'_'.join(categories)}_preview.csv"), index=False)
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect papers from arXiv")
    parser.add_argument("--categories", nargs="+", default=["cs.AI"], 
                        help="arXiv categories to collect papers from")
    parser.add_argument("--max_results", type=int, default=500,
                        help="Maximum number of papers to collect per category")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Directory to save the collected data")
    
    args = parser.parse_args()
    
    df = collect_arxiv_data(categories=args.categories, 
                            max_results=args.max_results,
                            output_dir=args.output_dir)

    print(f"\nCollected {len(df)} papers")
    print("\nData statistics:")
    print(f"- Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"- Average abstract length: {df['abstract_word_count'].mean():.1f} words")
    print(f"- Number of unique authors: {df['authors'].nunique()}")
    
    print("\nFirst few entries:")
    print(df[['title', 'abstract', 'date']].head()) 