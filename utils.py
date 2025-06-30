import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import torch

def load_pdf(file):
    """Load PDF file and return text content"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name
    
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    return documents

def create_vectorstore(documents):
    """Create vector store from documents"""
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    # Use Hugging Face embeddings (free)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create vector store
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

def create_llm():
    """Create Hugging Face LLM pipeline"""
    # Use a smaller, faster model that works well for Q&A
    model_name = "microsoft/DialoGPT-medium"
    
    # Create text generation pipeline
    text_pipeline = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=model_name,
        max_length=512,
        temperature=0.7,
        do_sample=True,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Create LangChain wrapper
    llm = HuggingFacePipeline(pipeline=text_pipeline)
    return llm

def get_answer(question, retriever):
    """Get answer from the retrieval QA chain"""
    try:
        # Create LLM
        llm = create_llm()
        
        # Create QA chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        # Get answer
        result = qa({"query": question})
        return result["result"]
    
    except Exception as e:
        return f"Error processing question: {str(e)}"