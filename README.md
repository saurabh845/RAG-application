# RAG Application with LangChain, Qdrant, and OpenAI

**Deployment link**: https://saurabh845-rag-application-rag-app-dccqso.streamlit.app

A **Retrieval-Augmented Generation (RAG)** application using **LangChain**, **Qdrant**, and **OpenAI models**, with **Streamlit** for the web interface.

## Libraries used
- **LangChain**: Framework for building LLM-powered applications.
- **Qdrant Vector Store**: Used for **semantic search** (dense vector-based retrieval).
- **OpenAI (GPT-4o-mini)**: Generates responses based on retrieved context.
- **FuzzyWuzzy**: Used for **keyword search** (approximate text matching).
- **Streamlit**: Provides an interactive **web-based UI**.

## How It Works
1. **Upload a PDF** or **Enter an HTML URL** to extract and index text.
2. Choose between semantic search (vector-based) or keyword search (fuzzy matching).
3. Enter your prompt query.
4. App retrieves relevant document chunks based on user prompt to generate response.
5. The response is **streamed** and displayed along with the search results used as context.

## Local Installation & Setup
In case the above deployment link is unreponsive.
```sh
# Clone the repository
git clone https://github.com/saurabh845/RAG-application
cd RAG-application

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API Key in .env file:
# OPENAI_API_KEY=your_openai_api_key

# Run the Streamlit app
streamlit run app.py
```
