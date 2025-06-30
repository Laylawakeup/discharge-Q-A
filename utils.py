import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings 

def process_pdf(uploaded_file):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

 
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()


    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)

 
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

   
    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever()
    return retriever

def get_answer(question, retriever):
    llm = ChatOpenAI(temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa.run(question)
    return result
