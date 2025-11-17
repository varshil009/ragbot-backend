"""
Simple script to create pickle files for the two reference books.
This is a simplified version that processes the books used in the application.

Make sure you have the PDF files in the 'uploads' folder or update the paths below.
"""

import pickle
import numpy as np
from base_rag2 import pdf_processing, embedd
import os

# Configuration
CHUNK_SIZE = 256
OVERLAP_SIZE = 50

# Book configurations
BOOKS = {
    'deep learning with python': {
        'pdf_path': 'uploads/Deep Learning with Python.pdf',  # Update this path
        'output_file': 'data_dl.pkl'
    },
    'python data science handbook': {
        'pdf_path': 'uploads/Python_Data_Science_Handbook-1-200.pdf',
        'output_file': 'data_ds.pkl'
    }
}

def process_book(book_name, pdf_path, output_file):
    """Process a single book and create its pickle file."""
    print(f"\n{'='*70}")
    print(f"Processing: {book_name}")
    print(f"PDF: {pdf_path}")
    print(f"Output: {output_file}")
    print(f"{'='*70}\n")
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"⚠ Warning: PDF file not found: {pdf_path}")
        print(f"  Please update the path in the script or place the PDF in the correct location.\n")
        return False
    
    try:
        # Step 1: Process PDF
        print("📄 Step 1: Processing PDF and creating chunks...")
        pdf_processor = pdf_processing(pdf_path, CHUNK_SIZE, OVERLAP_SIZE)
        chunks, chunk_page = pdf_processor.chunk_pdf_with_pages()
        print(f"   ✓ Created {len(chunks)} chunks\n")
        
        # Step 2: Generate embeddings
        print("🧠 Step 2: Generating embeddings (this may take several minutes)...")
        embedding_model = embedd()
        embeddings = embedding_model.generate_embeddings(chunks)
        print(f"   ✓ Generated {len(embeddings)} embeddings")
        print(f"   ✓ Embedding dimension: {embeddings[0].shape[0]}\n")
        
        # Step 3: Prepare data
        print("💾 Step 3: Preparing data structure...")
        data = {
            'embeddings': embeddings.tolist(),
            'chunks': chunks.tolist() if isinstance(chunks, np.ndarray) else list(chunks)
        }
        
        # Step 4: Save pickle file
        print(f"💾 Step 4: Saving to {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"   ✓ Successfully saved to {output_file}")
        print(f"\n{'='*70}")
        print(f"✅ Completed: {book_name}")
        print(f"   - Chunks: {len(chunks)}")
        print(f"   - Embeddings: {len(embeddings)}")
        print(f"   - File: {output_file}")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error processing {book_name}:")
        print(f"   {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Process all books."""
    print("\n" + "="*70)
    print("PDF to Pickle Converter")
    print("="*70)
    print("\nThis script will:")
    print("  1. Process PDF files and create text chunks")
    print("  2. Generate embeddings using CodeBERT model")
    print("  3. Save embeddings and chunks to pickle files")
    print("\n" + "="*70)
    
    results = {}
    
    for book_name, config in BOOKS.items():
        success = process_book(book_name, config['pdf_path'], config['output_file'])
        results[book_name] = success
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for book_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {book_name}: {status}")
    print("="*70 + "\n")
    
    if all(results.values()):
        print("🎉 All books processed successfully!")
        print("\nNext steps:")
        print("  1. Run 'python qdrant_save.py' to load data into Qdrant")
        print("  2. Start your Flask app with 'python app.py'")
    else:
        print("⚠ Some books failed to process. Please check the errors above.")

if __name__ == "__main__":
    main()

