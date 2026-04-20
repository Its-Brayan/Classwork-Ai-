import os
import chromadb
from langchain_classic import text_splitter
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
import torch
load_dotenv()

class VectorDB:
   """
    A simple vector database wrapped using chromaDB with HuggingFace embedding
    """
   def __init__(self,collection_name:str = None,embedding_model:str = None):
       """
            Initialize the vector database.

            Args:
                collection_name: Name of the ChromaDB collection
                embedding_model: HuggingFace model name for embeddings
            """
       self.collection_name = collection_name or os.getenv("COLLECTION_NAME")
       self.embedding_model_name = embedding_model or os.getenv("EMBEDDING_MODEL")
       #initialize chromadb client
       self.client = chromadb.PersistentClient("./chroma_db")
       #load embedding model
       print(f"Loading embedding model...${self.embedding_model_name}")
       self.embedding_model = SentenceTransformer(self.embedding_model_name)
       self.collection = self.client.get_or_create_collection(
           name=self.collection_name,
           metadata={"description":"Class work collection"}
       )
       print(f"Vector database initialized with {self.collection_name}")
      
   def chunk_text(self,publication:str,chunk_size:int=10000,chunk_overlap:int=500) -> List[str]:
       text_splitter = RecursiveCharacterTextSplitter(
           chunk_size = chunk_size,
           chunk_overlap = chunk_overlap
       )
       return text_splitter.split_text(publication)
   
   def insert_documents(self,documents:List) -> None:
       """
       Insert documents into the vector database.
       Args:
       documents: List of documents to insert
       """
       devices = (
           "cuda"
           if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu"
    )
       print(f'Using device:{devices}')
       model = HuggingFaceEmbeddings(
           model_name = os.getenv("EMBEDDING_MODEL"),
           model_kwargs={"device":devices}
       )
       #gets how many docouments are already in the collection
       existing_docs = self.collection.count()
       for publication in documents:
           content = getattr(publication,"page_content",str(publication))
           if not content.strip():
               continue
           chunked_publication = self.chunk_text(content)
           print(f"Chunked publications {len(chunked_publication)}")
           embeddings = model.embed_documents(chunked_publication)
           print("Embeddings created")
           ids = list(range(existing_docs,existing_docs+len(chunked_publication)))
           ids = [f"document{id}" for id in ids]
           self.collection.add(
               ids = ids,
               documents = chunked_publication,
               embeddings = embeddings
           )
           existing_docs += len(chunked_publication)
    
   def search_db(self,query:str,top_k:int=5):
       """
       Search the vector database for relevant documents based on a query
       """
       query_vector = self.embedding_model.encode(query).tolist()#convert query to vector
       result = self.collection.query(
           query_embeddings=[query_vector],
           n_results=top_k,
           include=["documents","metadatas","distances"]
       )
       #format results
       formatted_results = []
       for i,doc in enumerate(result["documents"][0]):
           formatted_results.append({
               "content":doc,
                "title":result["metadatas"][0][i].get("title","No Title") if result["metadatas"][0][i] else "No Title",
               "similarity": 1 - result["distances"][0][i]
           })
           return formatted_results


     
      