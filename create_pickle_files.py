"""
Script to create pickle files from PDF documents.
This processes PDFs, chunks them, generates embeddings, and saves to pickle format.

Usage:
    python create_pickle_files.py <pdf_path> <output_filename>
    
Example:
    python create_pickle_files.py "Deep Learning with Python.pdf" data_dl.pkl
    python create_pickle_files.py "Python Data Science Handbook.pdf" data_ds.pkl
"""

import sys
import pickle
import numpy as np
from base_rag2 import pdf_processing, embedd

def create_pickle_from_pdf(pdf_path, output_filename, chunk_size=256, overlap_size=50):
    """
    Process a PDF file and create a pickle file with embeddings and chunks.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_filename (str): Name of the output pickle file (e.g., 'data_dl.pkl')
        chunk_size (int): Size of each text chunk in tokens (default: 256)
        overlap_size (int): Size of overlap between chunks (default: 50)
    """
    print(f"\n{'='*60}")
    print(f"Processing PDF: {pdf_path}")
    print(f"Output file: {output_filename}")
    print(f"{'='*60}\n")
    
    # Step 1: Process PDF and create chunks
    print("Step 1: Processing PDF and creating chunks...")
    pdf_processor = pdf_processing(pdf_path, chunk_size, overlap_size)
    chunks, chunk_page = pdf_processor.chunk_pdf_with_pages()
    
    print(f"✓ Created {len(chunks)} chunks from PDF")
    print(f"  Chunk size: {chunk_size} tokens")
    print(f"  Overlap: {overlap_size} tokens\n")
    
    # Step 2: Generate embeddings
    print("Step 2: Generating embeddings...")
    print("  (This may take a while depending on PDF size...)\n")
    embedding_model = embedd()
    embeddings = embedding_model.generate_embeddings(chunks)
    
    print(f"✓ Generated {len(embeddings)} embeddings")
    print(f"  Embedding dimension: {embeddings[0].shape[0]}\n")
    
    # Step 3: Prepare data structure
    print("Step 3: Preparing data structure...")
    # Convert numpy arrays to lists for better pickle compatibility
    # Keep chunks as numpy array of strings (as used in the codebase)
    data = {
        'embeddings': embeddings.tolist(),  # Convert to list for pickle
        'chunks': chunks.tolist() if isinstance(chunks, np.ndarray) else list(chunks)
    }
    
    # Step 4: Save to pickle file
    print(f"Step 4: Saving to {output_filename}...")
    with open(output_filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✓ Successfully saved to {output_filename}")
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  - Total chunks: {len(chunks)}")
    print(f"  - Total embeddings: {len(embeddings)}")
    print(f"  - Embedding dimension: {embeddings[0].shape[0]}")
    print(f"  - Output file: {output_filename}")
    print(f"{'='*60}\n")
    
    return data

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 3:
        print("Usage: python create_pickle_files.py <pdf_path> <output_filename>")
        print("\nExample:")
        print('  python create_pickle_files.py "Deep Learning with Python.pdf" data_dl.pkl')
        print('  python create_pickle_files.py "Python Data Science Handbook.pdf" data_ds.pkl')
        print("\nOptional parameters (can be modified in script):")
        print("  - chunk_size: 256 (default)")
        print("  - overlap_size: 50 (default)")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_filename = sys.argv[2]
    
    # Check if PDF file exists
    import os
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    try:
        create_pickle_from_pdf(pdf_path, output_filename)
        print("✓ Process completed successfully!")
    except Exception as e:
        print(f"\n✗ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

