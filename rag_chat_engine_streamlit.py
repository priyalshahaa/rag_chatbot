from dotenv import load_dotenv
import os
import warnings
import pymupdf
import tempfile
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage

warnings.filterwarnings("ignore")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")

def load_and_chunk_pdf(file_path):
    """Load and chunk PDF file into manageable pieces"""
    try:
        doc = pymupdf.open(file_path)
        text = "".join(page.get_text() for page in doc)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_text(text)
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return None

def initialize_rag_chain(pdf_path):
    """Initialize vector store and RAG chain for a specific PDF"""
    chunks = load_and_chunk_pdf(pdf_path)
    if not chunks:
        return None
    
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.from_texts(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.3)
        
        retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the conversation, generate a search query to look up")
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, retriever_prompt)
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions about the provided document. 
             If the answer is not found in the documents, you can use your own knowledge to answer, but state that you are using your own knowledge.
             Keep answers concise. Answer based only on the following context:\n{context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        return create_retrieval_chain(history_aware_retriever, question_answer_chain)
    except Exception as e:
        st.error(f"Error initializing RAG chain: {e}")
        return None

def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
    st.title("📄 PDF Chatbot")
    st.caption("Ask questions about your PDF document")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_pdf" not in st.session_state:
        st.session_state.current_pdf = None
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

    with st.sidebar:
        st.header("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            if st.session_state.current_pdf != uploaded_file.name:
                with st.spinner("Processing PDF..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.getbuffer())
                            tmp_path = tmp.name
                        
                        st.session_state.rag_chain = initialize_rag_chain(tmp_path)
                        if st.session_state.rag_chain:
                            st.session_state.current_pdf = uploaded_file.name
                            st.session_state.messages = []
                            st.success(f"Ready to chat about: {uploaded_file.name}")
                        else:
                            st.error("Failed to process PDF")

                        os.unlink(tmp_path)
                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")
                        st.session_state.current_pdf = None
                        st.session_state.rag_chain = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the PDF..."):
        if st.session_state.rag_chain is None:
            st.warning("Please upload a PDF file first")
            st.stop()
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    lc_chat_history = []
                    for msg in st.session_state.messages[:-1]:
                        if msg["role"] == "user":
                            lc_chat_history.append(HumanMessage(content=msg["content"]))
                        else:
                            lc_chat_history.append(AIMessage(content=msg["content"]))
                    
                    result = st.session_state.rag_chain.invoke({
                        "input": prompt,
                        "chat_history": lc_chat_history
                    })
                    response = result["answer"]
                except Exception as e:
                    response = f"Sorry, I encountered an error: {str(e)}"
                
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


if __name__ == "__main__":
    main()