import os
from dotenv import load_dotenv
from config_loader import load_config,load_prompt
from vectordb import VectorDB
from langchain_community.document_loaders import PyMuPDFLoader,DirectoryLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List
from config_loader import load_prompt,load_config
import gradio as gr
load_dotenv()
config = load_config()
prompt_config = load_prompt()
def load_documents() -> List[str]:
    """
    Load documents from the specified directory
    """
    try:
        documents = []
        loader = DirectoryLoader("/home/brayan/Aiprojects/ClassworkAi/data",loader_cls=PyMuPDFLoader)
        loaded_docs = loader.load()
        print(f"Successfully loaded {loaded_docs}")
        documents.extend(loaded_docs)
    except Exception as e:
        print(f"Error loading documents:{str(e)}")
    publications = []
    for doc in documents:
        publications.append(doc.page_content)
    return publications

class ClassWorkAi:
    def __init__(self):
        self.llm = self.initialize_llm()
        if not self.llm:
            raise ValueError("LLm initialization failed. Please set a valid Google api Key")
        #initialize vector database
        self.vector_db = VectorDB()
        self.qa_prompt = prompt_config['qa_prompt']
        self.prompt_template = ChatPromptTemplate.from_template(
            template ="""
       {qa_prompt}

       Context:
       {context}

       Question:
       {question}
"""
        )
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def initialize_llm(self):
        if os.getenv("GOOGLE_API_KEY"):
            model_name = config['llm']
            print(f"Using Ai model: {model_name}")
            return ChatGoogleGenerativeAI(
                api_key = os.getenv('GOOGLE_API_KEY'),
                model = model_name,
                temperature = 0.0
            )
    
    def add_documents(self,documents:List) -> None:
        self.vector_db.insert_documents(documents)
    
    def invoke(self,input:str,top_k:int = 3) -> str:
          """
        Query the RAG assistant.

        Args:
            input: User's input
            n_results: Number of relevant chunks to retrieve

        Returns:
            Dictionary containing the answer and retrieved context
        """
          relevant_docs = self.vector_db.search_db(input,top_k)
          context = "\n\n".join(docs['content']for docs in relevant_docs)
          response = self.chain.invoke({
              "context":context,
              "question":input,
              "qa_prompt":self.qa_prompt
          })
          llm_answer = response.content if hasattr(response,"content") else str(response)
          return llm_answer
    
def main():
    """
    Main function to demonstrate Classwork AI
    """
    # Initialize Classwork AI
    classwork_ai = ClassWorkAi()
    
    # Load documents 
    sample_docs = load_documents()
    print(f"Loaded {len(sample_docs)} documents")

    classwork_ai.add_documents(sample_docs)
    
    def chat(question):
        return classwork_ai.invoke(question)
    
    # Launch Gradio interface
    interface = gr.Interface(
        fn=chat,
        inputs=gr.Textbox(lines=2, placeholder="Ask a question about your class notes"),
        outputs="text",
        title="Classwork Ai Assistant"
    )
    interface.launch(share=True)

if __name__ == "__main__":
    main()
    
    