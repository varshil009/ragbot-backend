import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image
import io

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

from langchain_core.prompts import ChatPromptTemplate
#from langchain_community.retrievers import BM25Retriever
from langchain_core.language_models.llms import LLM as LangChainLLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from typing import Any, List, Optional
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
    
    def check_caption(self, cap):
        k = cap.lower()
        return any(keyword in k for keyword in ["figure", "fig", "chart", "table", "graph"])

    def search_image_caption(self, margin, text_blocks, image_bbox):
        text_positions = {
                        'above': [],
                        'below': [],
                        'inside': []
                        }
    
        # Scan text blocks once and categorize them
        for block in text_blocks:
            y_pos = block[1]
            if y_pos > image_bbox[3] and y_pos < image_bbox[3] + margin:
                text_positions['below'].append(block[4])
            elif y_pos < image_bbox[3] and y_pos > image_bbox[1]:
                text_positions['inside'].append(block[4])
            elif y_pos < image_bbox[1] and y_pos > image_bbox[1] + margin:
                text_positions['above'].append(block[4])
        
        # Check each position for captions
        for position in ['below', 'inside', 'above']:
            text = ' '.join(text_positions[position])
            if self.check_caption(text):
                return text
                
        return ""

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

    def extract_captions_and_images(self):
        images = {}

        with fitz.open(self.path) as fitz_pdf:
            for page_num, page in enumerate(fitz_pdf):
                #print(f"Processing Page {page_num + 1}")

                # First get the raw image list
                image_list = page.get_images()
                if image_list:
                    #print(f"Page {page_num + 1} contains {len(image_list)} image(s).")

                    # Get locations of images on page
                    img_info_list = page.get_image_info()

                    # Extract text blocks with their bounding boxes
                    text_blocks = page.get_text("words")

                    # Process each image
                    for img_index, img in enumerate(image_list):
                        # Get the image location from img_info_list
                        img_info = img_info_list[img_index]
                        image_bbox = img_info['bbox']
                        #print(f"Image bounding box: {image_bbox}")

                        # Extract the raw image data
                        base_image = fitz_pdf.extract_image(img[0])
                        image_bytes = base_image["image"]

                        # getthe image caption
                        descr = self.search_image_caption(20, text_blocks, image_bbox)

                        # Convert to PIL Image while maintaining original quality
                        images[descr] = [Image.open(io.BytesIO(image_bytes)), image_bbox]

        return images




class Embedd:
    def __init__(self):
        self.client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        self.model_name = "voyage-3-lite"    # recommended for RAG
        print("==> Voyage embedding model ready")

    def generate_embeddings(self, chunks):
        """
        chunks: list of text passages
        returns: numpy array of embeddings
        """
        print("==> generating embeddings with Voyage...")
        response = self.client.embed(
            model=self.model_name,
            texts=chunks
        )
        # Voyage returns list of lists (float vectors)
        return np.array(response.embeddings)

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


    def fetch_image(self, query, cap_embeddings, cap_img):
        print("==> fetching image")
        query_embedding = self.generate_embeddings([query])[0]
        similarities = self.cosine_similarity_numpy(np.array([query_embedding]), cap_embeddings)
        similarities = np.where(similarities >= 0.8, similarities, -1)
        best_match_index = similarities.argsort()[0, ::-1][0]
        img = cap_img[list(cap_img.keys())[best_match_index]][0]
        print("==> image fetched successfully")
        return img

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

class CustomGoogleGenAI(LangChainLLM):
    """Custom LangChain LLM wrapper for Google GenAI client."""
    
    client: Any
    model: str = "gemini-2.5-flash"
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.95
    max_output_tokens: int = 1024
    
    @property
    def _llm_type(self) -> str:
        return "google_genai"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Google GenAI API."""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": self.temperature,
                    "top_k": self.top_k,
                    "top_p": self.top_p,
                    "max_output_tokens": self.max_output_tokens,
                }
            )
            
            # Extract text from response - ensure we always return a valid string
            text_result = None
            
            # Try different ways to extract text based on Google GenAI response structure
            try:
                # Method 1: Check if response has text attribute directly (Google GenAI SDK property)
                # This is often the easiest way - the SDK may provide this
                if hasattr(response, 'text'):
                    try:
                        text_attr = response.text
                        if text_attr is not None:
                            text_result = str(text_attr).strip()
                            if text_result:
                                return text_result
                    except Exception as e:
                        print(f"Error accessing response.text: {e}")
                
                # Method 2: Extract from candidates structure (most common)
                elif hasattr(response, 'candidates') and response.candidates:
                    if len(response.candidates) > 0:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'content'):
                            content = candidate.content
                            # Check for parts array
                            if hasattr(content, 'parts'):
                                parts = content.parts
                                if parts and len(parts) > 0:
                                    for part in parts:
                                        # Try to get text from part
                                        if hasattr(part, 'text') and part.text is not None:
                                            potential_text = str(part.text).strip()
                                            if potential_text:
                                                text_result = potential_text
                                                break
                                        elif isinstance(part, str) and part.strip():
                                            text_result = str(part).strip()
                                            break
                                        # Try accessing part attributes
                                        elif hasattr(part, '__dict__'):
                                            for attr in dir(part):
                                                if not attr.startswith('_') and attr not in ['text', 'content']:
                                                    try:
                                                        val = getattr(part, attr)
                                                        if isinstance(val, str) and val.strip():
                                                            text_result = val.strip()
                                                            break
                                                    except:
                                                        pass
                            # Check if content has text directly
                            if not text_result and hasattr(content, 'text') and content.text is not None:
                                text_result = str(content.text).strip()
                
                # Method 3: Try to get text from response object directly
                if not text_result and hasattr(response, 'text'):
                    text_result = str(response.text).strip()
                    
            except Exception as extract_error:
                print(f"Error extracting text from response: {extract_error}")
                import traceback
                traceback.print_exc()
            
            # Fallback: Try response.text property (Google GenAI SDK may provide this)
            if not text_result:
                try:
                    if hasattr(response, 'text'):
                        text_prop = response.text
                        if text_prop:
                            text_result = str(text_prop).strip()
                except Exception as e:
                    print(f"Error accessing response.text: {e}")
            
            # Last resort: Debug and provide error
            if not text_result:
                print(f"WARNING: Could not extract text. Response type: {type(response)}")
                if hasattr(response, 'candidates') and response.candidates:
                    print(f"Candidates count: {len(response.candidates)}")
                    if len(response.candidates) > 0:
                        cand = response.candidates[0]
                        print(f"Candidate type: {type(cand)}")
                        if hasattr(cand, 'content'):
                            print(f"Content type: {type(cand.content)}")
                            print(f"Content attributes: {[a for a in dir(cand.content) if not a.startswith('_')]}")
                text_result = "Error: Could not extract text from API response. Check console for details."
            
            # Ensure we never return None or empty - LangChain Pydantic requires a non-empty string
            if not text_result or len(text_result) == 0:
                text_result = "No response generated from the model."
            
            # Final validation - must be a string
            if not isinstance(text_result, str):
                text_result = str(text_result)
            
            return text_result
            
        except Exception as e:
            print(f"Error calling Google GenAI: {str(e)}")
            print(f"Response type: {type(response) if 'response' in locals() else 'N/A'}")
            # Return error message as string to prevent Pydantic validation error
            return f"Error: {str(e)}"

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
        Initialize the RAG process with Qdrant, BM25, and LangChain chains.
        
        Args:
            book_name (str): Name of the book to search in.
            documents (list): List of text chunks for BM25 retrieval.
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
        
        # Initialize BM25 retriever with the provided documents
        #self.bm25_retriever = BM25Retriever.from_texts(documents)
        #self.bm25_retriever.k = 5  # Number of top chunks to retrieve
        
        # Define reference books and their Qdrant collection names
        self.REFERENCE_BOOKS = {
            'deep learning with python': "data_dl_collection",
            'python data science handbook': "data_ds_collection"
        }
        # Initialize Google GenAI client for LangChain integration
        self.genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        # Create custom LLM wrapper for LangChain
        self.llm = CustomGoogleGenAI(client=self.genai_client, model="gemini-2.5-flash")

        # Define LangChain chains using LCEL (LangChain Expression Language)
        self.query_expansion_chain = self._create_query_expansion_chain()
        self.hybrid_retrieval_chain = self._create_hybrid_retrieval_chain()
        self.response_generation_chain = self._create_response_generation_chain()

        
    def _create_query_expansion_chain(self):
        """Chain to enhance and expand user queries using LCEL."""
        prompt = ChatPromptTemplate.from_template(
            """Expand this search query with 3-5 relevant terms and synonyms.
            Maintain the core meaning but add technical variations.
            Original: {query}
            Enhanced:"""
        )
        # Use LCEL with pipe operator instead of LLMChain
        return prompt | self.llm

    def _create_hybrid_retrieval_chain(self):
        """Chain to perform hybrid retrieval using Qdrant and BM25."""
        def hybrid_retrieval(query_em, query):
            # Semantic search with Qdrant
            qdrant_results = self.client.search(
                collection_name=self.REFERENCE_BOOKS[self.book_name],
                query_vector=query_em,
                with_payload=True,
                limit=5
            )
            qdrant_chunks = [res.payload['text'] for res in qdrant_results]
            
            # Keyword search with BM25
            #bm25_results = self.bm25_retriever.invoke(query)
            #bm25_chunks = [doc.page_content for doc in bm25_results]
            
            # Combine results from both retrievers
            top_chunks = list(set(qdrant_chunks))  # Remove duplicates
            return "\n__new_chunk__\n".join(top_chunks)
        
        return hybrid_retrieval

    def _create_response_generation_chain(self):
        """Chain to generate the final response using the LLM with LCEL."""
        prompt = ChatPromptTemplate.from_template(
            """Use this context to answer the question:
            Context: {context}
            Question: {query}
            Provide a comprehensive answer (~400 tokens):"""
        )
        # Use LCEL with pipe operator instead of LLMChain
        return prompt | self.llm

    def execute(self, query):
        """
        Execute the RAG process for the given query using LangChain chains.
        
        Args:
            query (str): User query.
        
        Returns:
            str: Final response generated by the LLM.
        """
        # Step 1: Query expansion using LCEL invoke
        enhanced_query_response = self.query_expansion_chain.invoke({"query": query})
        # Extract text from Pydantic AIMessage object (LangChain returns Pydantic models)
        enhanced_query = self._extract_text_from_response(enhanced_query_response)
        
        # Step 2: Generate embeddings for the enhanced query
        query_em = self.em.generate_embeddings([enhanced_query])[0]
        
        # Step 3: Perform hybrid retrieval
        rag_response = self.hybrid_retrieval_chain(query_em, query)
        
        # Step 4: Generate final response using LCEL invoke
        final_response_obj = self.response_generation_chain.invoke(
            {
                "query": query,
                "context": rag_response
            }
        )
        # Extract text from Pydantic AIMessage object (LangChain returns Pydantic models)
        final_response = self._extract_text_from_response(final_response_obj)
        
        return final_response
    
    def _extract_text_from_response(self, response_obj: Any) -> str:
        """
        Extract text from LangChain response objects (which are Pydantic models).
        
        LangChain chains return AIMessage objects (Pydantic BaseMessage models),
        which need to be properly extracted before JSON serialization for the frontend.
        
        Args:
            response_obj: Response from LangChain chain (could be AIMessage, str, or other)
        
        Returns:
            str: Extracted text content
        """
        # Handle Pydantic AIMessage objects from LangChain
        if isinstance(response_obj, (AIMessage, BaseMessage)):
            return response_obj.content
        
        # Handle objects with 'content' attribute (Pydantic models often have this)
        if hasattr(response_obj, 'content'):
            content = response_obj.content
            # If content is a list (like in some message formats), extract text
            if isinstance(content, list) and len(content) > 0:
                if hasattr(content[0], 'text'):
                    return content[0].text
                elif isinstance(content[0], str):
                    return content[0]
            elif isinstance(content, str):
                return content
        
        # Handle string responses
        if isinstance(response_obj, str):
            return response_obj
        
        # Fallback: convert to string (this handles Pydantic model serialization)
        # Pydantic models have __str__ and __repr__ methods that work well
        return str(response_obj)