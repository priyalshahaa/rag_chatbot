import os
import warnings
import pymupdf 
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

def get_pdf_file():
    while True:
        file_path = input("Enter PDF file path: ").strip()
        if not file_path:
            continue
        if not os.path.exists(file_path):
            continue
        if not file_path.lower().endswith('.pdf'):
            continue
        return file_path

def load_and_chunk_pdf(file_path):
    try:
        doc = pymupdf.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        return text_splitter.split_text(text)
    except Exception:
        return None

def create_faiss_index(chunks):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return FAISS.from_texts(chunks, embeddings)
    except Exception:
        return None

def chatbot_response(query, vector_store):
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.3)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            verbose=False
        )
        return qa_chain.invoke({"query": query})["result"]
    except Exception:
        return "Error generating response"

def main():
    pdf_path = get_pdf_file()
    chunks = load_and_chunk_pdf(pdf_path)
    if not chunks:
        return
    
    vector_store = create_faiss_index(chunks)
    if not vector_store:
        return
    
    while True:
        query = input("You: ").strip()

        if query.lower() in ['exit', 'quit']:
            print("Chatbot: Goodbye!")
            break
            
        if not query:
            continue
            
        answer = chatbot_response(query, vector_store)
        print(f"Chatbot: {answer}")

if __name__ == "__main__":
    main()