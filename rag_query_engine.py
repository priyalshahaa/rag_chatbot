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

def get_pdf_file():
    """Prompt user to input PDF file path and validate it"""
    while True:
        file_path = input("Enter the path to your PDF file: ").strip()
        if not file_path:
            print("Please provide a file path.")
            continue
        
        if not os.path.exists(file_path):
            print(f"File not found at: {file_path}")
            continue
            
        if not file_path.lower().endswith('.pdf'):
            print("Please provide a PDF file")
            continue
            
        return file_path

def load_and_chunk_pdf(file_path):
    """Load and chunk PDF file content"""
    try:
        doc = pymupdf.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None

def create_faiss_index(chunks):
    """Create FAISS vector store from text chunks"""
    try:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(chunks, embeddings)
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

def chatbot_response(query, vector_store):
    """Generate response using the vector store"""
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        retriever = vector_store.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        response = qa_chain.invoke({"query": query})
        return response["result"]
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    pdf_path = get_pdf_file()
    
    print("\nProcessing PDF...")
    chunks = load_and_chunk_pdf(pdf_path)
    if not chunks:
        print("Failed to process PDF. Please check the file and try again.")
        return
    
    print("Creating search index...")
    vector_store = create_faiss_index(chunks)
    if not vector_store:
        print("Failed to create search index.")
        return
    
    print(f"\nChatbot ready to discuss {os.path.basename(pdf_path)}!")
    print("Type 'exit' to quit.\n")
    
    while True:
        query = input("Your question: ").strip()
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        if not query:
            print("Please enter a question.")
            continue
            
        response = chatbot_response(query, vector_store)
        print("\nAnswer:", response, "\n")

if __name__ == "__main__":
    main()