import numpy as np
#import matplotlib.pyplot as plt

import voyageai
# for tokenizers and reading pdf
import fitz  # PyMuPDF
import pdfplumber


# Display the variable in Markdown format
#from IPython.display import Markdown, display
#from pymilvus import connections, FieldSchema, CollectionSchema, Collection, DataType
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pickle

from typing import Any
from google import genai
import os


# for api (requests no longer needed with new google.genai client)
import json

# to calculate similarities


class pdf_processing:
    def __init__(self, path, chunk_size, overlap_size):
        self.path = path
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    def chunk_pdf_with_pages(self):
        """
        Chunk PDF text with overlapping while preserving page numbers.
        
        Args:
            pdf: PDF document object
            chunk_size: Size of each chunk in tokens
            overlap_size: Size of overlap between chunks
        
        Returns:
            Dict mapping text chunks to their page number(s)
        """
        textwise_page = {}
        pdf = pdfplumber.open(self.path)
        pages = pdf.pages
        
        incomplete_chunk = {
            "text": np.array([], dtype='object'),
            "pages": []
        }
        
        for i in range(len(pages)):
            if i %50 == 0: print(f"page num {i}")
            try:
                page_text = np.array(pages[i].extract_text().split(), dtype='object')
                if len(page_text) == 0:
                    continue
                    
                next_index = 0
                #o = 0
                while True:
                    # Handle text from previous page
                    current_incomplete_len = len(incomplete_chunk["text"])
                    if current_incomplete_len > 0:
                        stop_index = next_index + self.chunk_size - current_incomplete_len
                    else:
                        stop_index = next_index + self.chunk_size
                        
                    # Create new chunk
                    new_chunk = np.append(incomplete_chunk["text"], 
                                        page_text[next_index:min(stop_index, len(page_text))])
                    
                    
                    # Handle the chunk
                    if len(new_chunk) < self.chunk_size and i < len(pages) - 1:
                        # Save incomplete chunk for next iteration
                        incomplete_chunk["text"] = new_chunk
                        if i not in incomplete_chunk["pages"]:
                            incomplete_chunk["pages"].append(i)
                    else:
                        # Process complete chunk
                        chunk_text = " ".join(new_chunk.tolist()).lower()
                        if current_incomplete_len > 0:
                            # Chunk spans multiple pages
                            textwise_page[chunk_text] = incomplete_chunk["pages"] + [i]
                            # Reset incomplete chunk
                            incomplete_chunk["text"] = np.array([], dtype='object')
                            incomplete_chunk["pages"] = []
                        else:
                            # Chunk from single page
                            textwise_page[chunk_text] = i
                    #print("diff", stop_index, self.overlap_size)
                    next_index = stop_index - self.overlap_size  
                        # Update next starting position with overlap
                    #print("next before", next_index)
                    if next_index < 0: 
                        next_index = 0 
                    #print(f"next : {next_index}\n last : {stop_index} ")
                    #print("page ", i, f"len : {len(page_text)}", "\n", new_chunk)

                    if stop_index >= len(page_text): 
                        #print("-------------------USED MAIN BREAK----------------")
                        break
            
            except Exception as e:
                print(f"Error processing page {i}: {str(e)}")
                continue

            #if o > 15: break
        # Handle any remaining incomplete chunk at the end
        if len(incomplete_chunk["text"]) > 0:
            final_text = " ".join(incomplete_chunk["text"].tolist()).lower()
            textwise_page[final_text] = incomplete_chunk["pages"]
        
        chunks = np.array(list(textwise_page.keys()))
        return chunks, textwise_page
        
        #def extract_text_and_chunks(self):
        # extract text
        """with fitz.open(self.path) as pdf:
            for page in pdf:
                self.text_p.append(page.get_text())
                self.text += page.get_text()
        return self.text_p, self.text"""

        """# pypdf2 works well with some other forms of pdfs as well
        with open(self.path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            pages = reader.pages
            total_pages = len(pages)

            for i in range(total_pages):
                textn = pages[i].extract_text()
                self.text_p.append(textn)
                self.text += textn
        """
        """ 
        pdf = pdfplumber.open(self.path)
        textwise_page = {}
        for i, page in enumerate(pdf.pages):
            text = page.extract_text().split()
            for j in range(0, len(text), self.chunk_size):
                new_chunk = " ".join(text[j : j + self.chunk_size]).lower()
                textwise_page[new_chunk] = i

        text_chunks =  np.array(list(textwise_page.keys()))
        return text_chunks, textwise_page"""

        """
        def create_chunks(self):
        # create chunks
        for i in range(0, len(self.text), self.chunk_size):
            new_chunk = self.text[i : i + self.chunk_size].lower()
            self.chunks.append(new_chunk)
        return self.chunks
        """

        """# Find description text below image
        descr = ""
        for block in text_blocks:
            if block[1] > image_bbox[3] and block[1] < image_bbox[3] + margin:
                descr += " " + block[4]

        if check_caption(descr):
            pass

        else:
            descr = ""
            for block in text_blocks:
                if block[1] < image_bbox[3] and block[1] > image_bbox[1]:
                    descr += " " + block[4]

            if check_caption(descr):
                pass
            else:

                descr = ""
                for block in text_blocks:
                    if block[1] < image_bbox[1] and block[1] > image_bbox[1] + margin:
                        descr += " " + block[4]"""




class Embedd:
    def __init__(self):
        self.client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        self.model_name = "voyage-3-lite"    # recommended for RAG
        print("==> Voyage embedding model ready")

    def generate_embeddings(self, chunks):
        """
        chunks: list of text passages
        returns: numpy array of embeddings (768-dim)
        """
        print("==> generating embeddings with Voyage...")

        # DO NOT TOUCH THE API
        response = self.client.embed(
            model=self.model_name,
            texts=chunks
        )

        emb = np.array(response.embeddings)   # shape (N, 512)
        N, D = emb.shape                      # D = 512

        # Duplicate first 256 dims
        prefix = emb[:, :256]                # shape (N, 256)

        # Final = prefix(256) + original(512) = 768
        final_emb = np.concatenate([prefix, emb], axis=1)

        print("Padded embedding from 512 -> 768 (using first 256 dims)")

        return final_emb


    @staticmethod
    def cosine_similarity_numpy(v1, v2):
        dot_product = np.dot(v1, v2.T)
        norm_v1 = np.linalg.norm(v1, axis=1)
        norm_v2 = np.linalg.norm(v2, axis=1)
        return dot_product / (norm_v1[:, np.newaxis] * norm_v2)

    def search(self, query, embeddings, chunks, chunk_page):
        # Embed the query
        query_embedding = self.generate_embeddings([query])[0]

        # Compute similarity
        similarities = self.cosine_similarity_numpy(
            np.array([query_embedding]), embeddings
        )

        best_match_index = similarities.argsort()[0][::-1]

        ref = [chunks[i] for i in best_match_index[:5]]

        page_nums = []
        for text in ref:
            pages = chunk_page[text]
            if isinstance(pages, int):
                page_nums.append(pages)
            else:
                page_nums.extend(pages)

        return "\n".join(ref), page_nums

class LLM:
  def __init__(self):
    print("==> LLM initiating")
    self.api_key = os.getenv("GOOGLE_API_KEY")
    self.client = genai.Client(api_key=self.api_key)
    self.query = None
    self.rag_response = None
    print("==> LLM initiated")

  def generate_rag(self, query):
    self.query = query
    prompt = f"""
                            You are an AI assistant capable of processing large documents and providing detailed, structured responses.
                            Your task is to analyze the user query and guide a retrieval system to fetch relevant information from a knowledge base or document repository.

                            Here's the workflow:
                            1. I will provide you with a query or a goal.
                            2. Analyze the query and list the key information, topics, or concepts that should be retrieved to answer it.

                            ### Input Query:
                            {self.query}

                            ### Your Output:
                            1. Identify key information or topics relevant to the query.
                            2. Suggest search terms or filters to retrieve the most relevant content.
                            keep it approx 100 words at max and keep it pointed towards original query
                            """

    response = self.client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.95,
            "max_output_tokens": 1024,
        }
    )
    
    # Extract text from response - handle Google GenAI response structure
    result = None
    try:
        # Method 1: Direct text attribute
        if hasattr(response, 'text') and response.text is not None:
            result = str(response.text).strip()
        # Method 2: From candidates structure
        elif hasattr(response, 'candidates') and response.candidates:
            if len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content'):
                    content = candidate.content
                    if hasattr(content, 'parts') and content.parts and len(content.parts) > 0:
                        part = content.parts[0]
                        if hasattr(part, 'text'):
                            result = str(part.text).strip()
                        elif isinstance(part, str):
                            result = str(part).strip()
    except Exception as e:
        print(f"Error extracting text in LLM.generate_rag: {e}")
        import traceback
        traceback.print_exc()
    
    # Fallback
    if not result:
        result = "Error: Could not extract text from response"
    
    print("==> enhanced query generated")
    return result

  def generate_final_response(self, rag_response):
    prompt = f"""You are a chatbot geared with RAG. you are provided reference information from RAG mechanism
                Here is the retrieved information. Refine your response by integrating this data to provide a complete answer to the query.

                ### Original Query:
                {self.query}

                ### Retrieved Information:
                {rag_response}

                ### Your Task:
                1. Synthesize the retrieved data into a coherent, detailed response.
                2. Present the final response in a user-friendly format, highlighting key points and providing structured details if required.
                Return answer as if you are interacting with user. Keep it formal and in less than 100 words.
              """
    print("==> sending enhanced query")
    
    response = self.client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.95,
            "max_output_tokens": 1024,
        }
    )
    
    # Extract text from response - handle Google GenAI response structure
    result = None
    try:
        # Method 1: Direct text attribute
        if hasattr(response, 'text') and response.text is not None:
            result = str(response.text).strip()
        # Method 2: From candidates structure
        elif hasattr(response, 'candidates') and response.candidates:
            if len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content'):
                    content = candidate.content
                    if hasattr(content, 'parts') and content.parts and len(content.parts) > 0:
                        part = content.parts[0]
                        if hasattr(part, 'text'):
                            result = str(part.text).strip()
                        elif isinstance(part, str):
                            result = str(part).strip()
    except Exception as e:
        print(f"Error extracting text in LLM.generate_final_response: {e}")
        import traceback
        traceback.print_exc()
    
    # Fallback
    if not result:
        result = "Error: Could not extract text from response"
    
    print("==> final response received")
    return result

"""class rag_process:
    def __init__(self, path):
        #
            #print("==> Processing of pdf started")
            self.path = path
            obj = pdf_processing(self.path, 256, 50)
            #print("chunking...")
            self.chunks, self.chunk_page = obj.chunk_pdf_with_pages()
            #print("extracting images...")
            self.cap_img = obj.extract_captions_and_images()
            self.embeddings = self.em.generate_embeddings(self.chunks)
            self.cap_embeddings = self.em.generate_embeddings(list(self.cap_img.keys()))
            print("==> Processing the pdf finished")
        #
        print("rag process initiated")
        self.em = embedd()
        self.client = QdrantClient(host="localhost", port=6333)
        print("local host ready")

    def search(self, book, query_em):
        REFERENCE_BOOKS = {
                                'deep learning with python': "data_dl_collection",
                                'python data science handbook': "data_ds_collection"#,
                                #'book3': 'path/to/book3.pdf',
                            }

        results = self.client.search(
                                    collection_name = REFERENCE_BOOKS[book],
                                    query_vector = query_em,
                                    with_payload = True,  # Retrieve text payload
                                    limit = 5  # Adjust number of results as needed
                                )

        print("loaded respected embeddings")
        #top_embeddings = [res.id for res in results]
        top_chunks = "\n__new_chunk__\n"([res.payload['text'] for res in results])
        return top_chunks

    def execute(self, query):
        #path = "/content/Deep Learning with Python.pdf"
        """"""
        # this query comes from backend in which it comes from frontend
        query = query.lower()
        llm = LLM()
        enhanced_query = llm.generate_rag(query)
        query_em = self.em.generate_embeddings(enhanced_query)

        rag_response = self.search()
        #final_response = llm.generate_final_response(rag_response)
        bin_img = self.em.fetch_image(query, self.cap_embeddings, self.cap_img)
        #print("the page nums :", page_nums)
        #return final_response, bin_img, page_nums

class rag_process:
    def __init__(self, book_name):

        #Initialize the RAG process for the selected book.

        print("RAG process initiated")
        self.em = embedd()
        self.client = QdrantClient(host = "localhost", port = 6333)
        self.book_name = book_name
        print(f"RAG process ready for book: {self.book_name}")

    def search(self, query_em):

        #Search for relevant chunks in the selected book.

        REFERENCE_BOOKS = {
            'deep learning with python': "data_dl_collection",
            'python data science handbook': "data_ds_collection"
        }

        results = self.client.search(
            collection_name = REFERENCE_BOOKS[self.book_name],
            query_vector = query_em,
            with_payload = True,  # Retrieve text payload
            limit = 5  # Adjust number of results as needed
        )

        print("Loaded respected embeddings")
        top_chunks = "\n__new_chunk__\n".join([res.payload['text'] for res in results])
        return top_chunks

    def execute(self, query):

       # Execute the RAG process for the given query.

        query = query.lower()
        llm = LLM()
        enhanced_query = llm.generate_rag(query)
        query_em = self.em.generate_embeddings([enhanced_query])[0]

        rag_response = self.search(query_em)
        final_response = llm.generate_final_response(rag_response)
        return final_response
        


class rag_process:
    def __init__(self, book_name):
        #Initialize ONLY with picklable data.

        self.client = QdrantClient(host="localhost", port=6333)
        print("connection successfull with client")
        self.em = embedd()
        self.book_name = book_name  # Only store the book name (a string)
        print(f"RAG process initialized for book: {self.book_name}")

    def search(self, query_em):

        #Search for relevant chunks in the selected book.

        REFERENCE_BOOKS = {
            'deep learning with python': "data_dl_collection",
            'python data science handbook': "data_ds_collection"
        }

        # Initialize client here
        results = self.client.search(
            collection_name=REFERENCE_BOOKS[self.book_name],
            query_vector = query_em,
            with_payload=True,
            limit=5
        )

        print("Loaded respected embeddings")
        top_chunks = "\n__new_chunk__\n".join([res.payload['text'] for res in results])
        return top_chunks

    def execute(self, query):

        #Execute the RAG process for the given query.

        llm = LLM()
        query = query.lower()
        enhanced_query = llm.generate_rag(query)
        query_em = self.em.generate_embeddings([enhanced_query])[0]

        rag_response = self.search(query_em)
        final_response = llm.generate_final_response(rag_response)
        return final_response
"""

class load_db:
    def __init__(self, book_name):
        self.client = QdrantClient(url = os.getenv("QDRANT_URL"), api_key = os.getenv("QDRANT_API_KEY"))
        
        # Define reference books and their Qdrant collection names
        self.REFERENCE_BOOKS = {
            'deep learning with python': "data_dl_collection",
            'python data science handbook': "data_ds_collection"
        }

        # Use the scroll API to fetch all records
        self.records, _ = self.client.scroll(
            collection_name = self.REFERENCE_BOOKS[book_name],
            limit = 100, #adjust the limit as needed
            with_payload = True,  # Include payload in the response
            with_vectors = False  # Exclude vectors to save bandwidth
        )
        self.records = [rec.payload['text'] for rec in self.records]

    def info(self):
        # Extract the text payloads from the recordssele
        return self.client, self.records

class rag_process:
    def __init__(self, book_name):
        """
        Initialize the RAG process with Qdrant and direct Google GenAI calls.

        Args:
            book_name (str): Name of the book to search in.
        """
        # Initialize Qdrant client
        db = load_db(book_name)
        self.client, documents = db.info()
        print("Connection successful with Qdrant client")

        # Initialize embedding model
        self.em = Embedd()

        # Store book name
        self.book_name = book_name
        print(f"RAG process initialized for book: {self.book_name}")

        # Define reference books and their Qdrant collection names
        self.REFERENCE_BOOKS = {
            'deep learning with python': "data_dl_collection",
            'python data science handbook': "data_ds_collection"
        }

        # Initialize Google GenAI client
        self.genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def _call_llm(self, prompt: str) -> str:
        """Call Google GenAI API directly and extract text response."""
        try:
            response = self.genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "temperature": 0.7,
                    "top_k": 40,
                    "top_p": 0.95,
                    "max_output_tokens": 1024,
                }
            )

            # Extract text from response
            if hasattr(response, 'text') and response.text is not None:
                return str(response.text).strip()
            elif hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    parts = candidate.content.parts
                    if parts and len(parts) > 0 and hasattr(parts[0], 'text'):
                        return str(parts[0].text).strip()

            return "Error: Could not extract text from response"
        except Exception as e:
            print(f"Error calling Google GenAI: {e}")
            return f"Error: {str(e)}"

    def _expand_query(self, query: str) -> str:
        """Expand user query with relevant terms using f-string prompt."""
        prompt = f"""Expand this search query with 3-5 relevant terms and synonyms.
Maintain the core meaning but add technical variations.
Original: {query}
Enhanced:"""
        return self._call_llm(prompt)

    def _retrieve_context(self, query_em, query: str) -> str:
        """Perform retrieval using Qdrant."""
        qdrant_results = self.client.query_points(
            collection_name=self.REFERENCE_BOOKS[self.book_name],
            query=query_em,
            with_payload=True,
            limit=5
        ).points

        qdrant_chunks = [res.payload['text'] for res in qdrant_results]
        top_chunks = list(set(qdrant_chunks))  # Remove duplicates
        return "\n__new_chunk__\n".join(top_chunks)

    def _generate_response(self, query: str, context: str) -> str:
        """Generate final response using f-string prompt."""
        prompt = f"""Use this context to answer the question:
Context: {context}
Question: {query}
Provide a comprehensive answer (~400 tokens):"""
        return self._call_llm(prompt)

    def execute(self, query):
        """
        Execute the RAG process for the given query.

        Args:
            query (str): User query.

        Returns:
            str: Final response generated by the LLM.
        """
        # Step 1: Query expansion
        enhanced_query = self._expand_query(query)

        # Step 2: Generate embeddings for the enhanced query
        query_em = self.em.generate_embeddings([enhanced_query])[0]

        # Step 3: Retrieve relevant context
        rag_response = self._retrieve_context(query_em, query)

        # Step 4: Generate final response
        final_response = self._generate_response(query, rag_response)

        return final_response