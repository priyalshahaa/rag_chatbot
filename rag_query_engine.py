import os
import pymupdf 
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

llm = OpenAI(openai_api_key=openai_api_key)

def load_and_chunk_pdf(file_path):
    doc = pymupdf.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

chunks = load_and_chunk_pdf('file_path.pdf')

def create_faiss_index(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

vector_store = create_faiss_index(chunks)

def chatbot_response(query, vector_store):
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    response = qa_chain.invoke({"query": query})
    return response["result"]

query = "What is the main topic of the document?"
response = chatbot_response(query, vector_store)
print(response)