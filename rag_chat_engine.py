import os
import warnings
import pymupdf
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

warnings.filterwarnings("ignore")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")

def get_pdf_file():
    """Prompt user for PDF file path and validate it"""
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
    """Load and chunk PDF file content"""
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
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None

def create_faiss_index(chunks):
    """Create vector store from PDF chunks"""
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return FAISS.from_texts(chunks, embeddings)
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

def initialize_rag_chain(vector_store):
    """Initialize the RAG conversation chain"""
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.3)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the conversation, generate a search query to look up")
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, retriever_prompt
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions about the provided document. 
         If the answer is not found in the documents, you can use your own knowledge to answer, but state that you are using your own knowledge.
         Answer based only on the following context:\n{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

def main():
    pdf_path = get_pdf_file()
    
    print("\nProcessing document...")
    chunks = load_and_chunk_pdf(pdf_path)
    if not chunks:
        print("Failed to process PDF. Please check the file and try again.")
        return
    
    vector_store = create_faiss_index(chunks)
    if not vector_store:
        print("Failed to create search index.")
        return
    
    rag_chain = initialize_rag_chain(vector_store)
    chat_history = []

    while True:
        try:
            query = input("You: ").strip()
            if query.lower() in ["exit", "quit"]:
                print("Chatbot: Goodbye!")
                break
                
            if not query:
                continue
                
            result = rag_chain.invoke({
                "input": query,
                "chat_history": chat_history
            })
            
            print(f"Chatbot: {result['answer']}")
            
            chat_history.extend([
                HumanMessage(content=query),
                AIMessage(content=result["answer"])
            ])
            
        except KeyboardInterrupt:
            print("\nChatbot: Goodbye!")
            break
        except Exception as e:
            print(f"Chatbot: Sorry, I encountered an error. Please try again.")

if __name__ == "__main__":
    main()