import getpass
import os

import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdfminer.high_level import extract_text
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass(
        "Enter your OpenAI API Key: ")  # Get key from user console if not set

# Initialize Qdrant client
client = QdrantClient(":memory:")
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# Initialize LangChain VectorStore for Qdrant
vector_store = QdrantVectorStore(
    client=client,
    collection_name="documents",
    embedding=embeddings,
)

def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    return extract_text(file)

def extract_text_from_html(url):
    """Extracts text from an HTML page given its URL."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()

def handle_document_upload(uploaded_file=None, html_url=None):
    """Handles document upload and indexing in the vector store.
    
    Args:
        uploaded_file: Uploaded PDF file.
        html_url: URL of an HTML page.
    
    Returns:
        First document chunk if successful, otherwise None.
    """
    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)
    elif html_url:
        text = extract_text_from_html(html_url)
    else:
        st.error("No valid document or URL provided.")
        return None

    documents = [Document(page_content=text)]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=300) # Split text into chunks
    splits = text_splitter.split_documents(documents)
    vector_store.add_documents(splits)
    st.success("Document uploaded and indexed successfully!")

    return splits[0]

def perform_search(query, search_type="semantic"):
    """Performs document search using either semantic or keyword-based search.
    
    Args:
        query: The search query string.
        search_type: Type of search 'semantic' or 'keyword'.
    
    Returns:
        List of matching Document objects.
    """
    if search_type == "semantic":
        return vector_store.similarity_search(query, k=10)
    else:
        documents = []
        scroll_result, offset = vector_store.client.scroll(
            collection_name=vector_store.collection_name)   # Fetch all chunks by page
        documents.extend([doc.payload['page_content']
                         for doc in scroll_result])
        while offset:
            scroll_result, offset = vector_store.client.scroll(
                collection_name=vector_store.collection_name, offset=offset)    # Next page
            documents.extend([doc.payload['page_content']
                             for doc in scroll_result])

        fuzzy_scores = []
        for idx, doc in enumerate(documents):
            fuzzy_score = fuzz.partial_ratio(query.lower(), doc.lower())
            fuzzy_scores.append((fuzzy_score, idx))
        fuzzy_scores.sort(reverse=True, key=lambda x: x[0])
        top_indices = [idx for _, idx in fuzzy_scores[:20]] # Top 20 results
        results = [documents[idx] for idx in top_indices]

        ret = [Document(page_content=doc) for doc in results]
        return ret

def get_response(user_query, doc_head, context):
    """Generates a streaming response to a user query using an LLM.
    
    Args:
        user_query: The user's query.
        doc_head: First chunk of the document being queried.
        context: Retrieved context information.
    
    Returns:
        Stream of responses from the language model.
    """
    template = """
            The start of the document being queried is: {doc_head}\n\n
            Use the following context to answer the User Query: {context}\n\n
            User Query: {user_query}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model='gpt-4o-mini',
                     max_tokens=1000)
    chain = prompt | llm | StrOutputParser()
    return chain.stream(
        {"doc_head": doc_head, "context": context, "user_query": user_query})

def process_document(doc_head):
    """Processes a document and retrieves responses based on user queries.
    
    Args:
        doc_head: The first chunk of the document being processed.
    """
    st.header("Prompt")
    search_type = st.selectbox(
        "Select document search type", [
            "semantic", "keyword"])
    user_query = st.text_input("User query about the document")
    st.subheader("LLM Response")

    if user_query:
        retrieved_docs = perform_search(user_query, search_type)
        context = "\n\n".join(str(doc.page_content) for doc in retrieved_docs)

        with st.chat_message("AI"):
            st.write_stream(get_response(user_query,
                                         doc_head.page_content,
                                         context))

        st.subheader("Search results used as context")
        for i, result in enumerate(retrieved_docs):
            st.subheader(f"Document chunk {i+1}")
            st.write(result.page_content)

        st.subheader("Document head")
        st.write(doc_head.page_content)

# Streamlit flow follows
st.title("RAG Application")

st.header("Upload Document or Enter HTML URL")
option = st.radio("Choose an input method:", ("Upload PDF", "Enter HTML URL"))

doc_head = None

# PDF or HTML mode selector
if option == "Upload PDF":
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_file is not None:
        doc_head = handle_document_upload(uploaded_file=uploaded_file)
elif option == "Enter HTML URL":
    html_url = st.text_input("Enter the URL of an HTML page")
    if html_url:
        doc_head = handle_document_upload(html_url=html_url)

if doc_head:
    process_document(doc_head)
