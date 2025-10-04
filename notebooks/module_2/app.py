import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from typing import List
import nest_asyncio

# Updated configuration for Mistral AI
MODEL_NAME = "mistral-small-latest"
MODEL_PROVIDER = "mistral"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_RETRIEVAL_DOCS = 5
TEMPERATURE = 0.1

# Updated system prompt for better responses
RAG_SYSTEM_PROMPT = """You are a helpful assistant specializing in LangSmith documentation and tools. 
Use the provided context to answer questions accurately and concisely.

Guidelines:
- Answer based on the retrieved context
- If information is missing from context, say so clearly  
- Keep responses practical and to the point
- Use 3-4 sentences maximum for clarity

Context will be provided below, followed by the user's question."""

# Initialize Mistral client
mistral_client = ChatMistralAI(
    model=MODEL_NAME,
    temperature=TEMPERATURE
)

def get_vector_db_retriever():
    """
    Initialize vector store with HuggingFace embeddings for LangSmith documentation.
    """
    persist_path = os.path.join(tempfile.gettempdir(), "langsmith_docs.parquet")
    
    # Initialize HuggingFace embeddings
    embd = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Load existing vector store if available
    if os.path.exists(persist_path):
        print(f"Loading existing vector store from {persist_path}")
        try:
            vectorstore = SKLearnVectorStore(
                embedding=embd,
                persist_path=persist_path,
                serializer="parquet"
            )
            return vectorstore.as_retriever(search_kwargs={"k": MAX_RETRIEVAL_DOCS})
        except Exception as e:
            print(f"Error loading vector store: {e}")

    # Create new vector store from LangSmith documentation
    print("Creating vector store from LangSmith documentation...")
    try:
        # Load documents from LangSmith sitemap
        ls_docs_sitemap_loader = SitemapLoader(
            web_path="https://docs.smith.langchain.com/sitemap.xml", 
            continue_on_failure=True
        )
        ls_docs = ls_docs_sitemap_loader.load()
        print(f"Loaded {len(ls_docs)} documents from LangSmith documentation")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(ls_docs)
        print(f"Split documents into {len(doc_splits)} chunks")

        # Create and persist vector store
        vectorstore = SKLearnVectorStore.from_documents(
            documents=doc_splits,
            embedding=embd,
            persist_path=persist_path,
            serializer="parquet"
        )
        vectorstore.persist()
        print(f"Vector store created and saved to {persist_path}")
        
        return vectorstore.as_retriever(search_kwargs={"k": MAX_RETRIEVAL_DOCS})
        
    except Exception as e:
        print(f"Error creating vector store: {e}")
        # Fallback retriever with sample documents
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.documents import Document
        
        class DummyRetriever(BaseRetriever):
            def _get_relevant_documents(self, query: str):
                return [
                    Document(
                        page_content="LangSmith is a platform for debugging, testing, evaluating, and monitoring LLM applications. It provides tracing, evaluation tools, and monitoring capabilities.",
                        metadata={"source": "fallback"}
                    ),
                    Document(
                        page_content="LangSmith tracing helps you understand your LLM application's execution by tracking inputs, outputs, and intermediate steps.",
                        metadata={"source": "fallback"}
                    )
                ]
        
        print("Using fallback retriever with sample content")
        return DummyRetriever()

nest_asyncio.apply()
retriever = get_vector_db_retriever()

@traceable(run_type="retriever")
def retrieve_documents(question: str):
    """Retrieve relevant documents from the vector store."""
    return retriever.invoke(question)

@traceable(run_type="chain")
def generate_response(question: str, documents):
    """Generate response using Mistral AI based on retrieved documents."""
    # Format the documents into context
    formatted_docs = "\n\n".join(doc.page_content for doc in documents)
    
    messages = [
        {
            "role": "system",
            "content": RAG_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"Context: {formatted_docs}\n\nQuestion: {question}"
        }
    ]
    
    return call_mistral(messages)

@traceable(run_type="llm")
def call_mistral(messages: List[dict]) -> str:
    """Call Mistral AI with the provided messages."""
    # Convert dict messages to LangChain message objects
    langchain_messages = []
    for msg in messages:
        if msg["role"] == "system":
            langchain_messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
    
    response = mistral_client.invoke(langchain_messages)
    return response

@traceable(run_type="chain")
def langsmith_rag(question: str):
    """Main RAG pipeline using Mistral AI and HuggingFace embeddings."""
    documents = retrieve_documents(question)
    response = generate_response(question, documents)
    return response.content

# Simple test function
def test_rag_system():
    """Test the RAG system with a few example questions."""
    print("=" * 60)
    print("LANGSMITH RAG SYSTEM - MISTRAL AI VERSION")
    print("=" * 60)
    print(f"LLM: {MODEL_PROVIDER} ({MODEL_NAME})")
    print(f"Embeddings: {EMBEDDING_MODEL}")
    print("-" * 60)
    
    # Test questions
    test_questions = [
        "What is LangSmith and how does it help with LLM applications?",
        "How do I set up tracing in LangSmith?",
        "What evaluation features does LangSmith provide?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nTest {i}: {question}")
        print("-" * 40)
        try:
            response = langsmith_rag(question)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 40)
    
    print("\nKey Changes Made:")
    print("- Switched from OpenAI to Mistral AI (mistral-small-latest)")
    print("- Updated to use HuggingFace embeddings")
    print("- Simplified system prompt for better responses")
    print("- Maintained original RAG pipeline structure")
    print("- Added proper message formatting for Mistral AI")
    print("=" * 60)

if __name__ == "__main__":
    # Run the test
    test_rag_system()
