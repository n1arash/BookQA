import os
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.tools import ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
import torch
from llama_index.core.agent import ReActAgent
from dotenv import load_dotenv
from llama_index.core.tools import QueryEngineTool
from llama_parse import LlamaParse

# Load environment variables
load_dotenv()

# Set up Streamlit page configuration
st.set_page_config(page_title="Document Q&A", page_icon="📚", layout="wide")
st.title("Document Q&A System")

# Check for CUDA availability
if torch.cuda.is_available():
    st.success("Using CUDA GPU")
else:
    st.warning("Using CPU")

# Set up environment variables
os.environ["GROQ_API_KEY"] = "ANY" # Groq API
os.environ["LLAMA_CLOUD_API_KEY"] = "ANY" # Llama Parser

# Initialize Groq LLM
@st.cache_resource
def get_llm():
    return Groq(model="llama3-70b-8192", api_key=os.environ["GROQ_API_KEY"])

llm = get_llm()

# Initialize HuggingFace Embedding
@st.cache_resource
def get_embedding():
    return HuggingFaceEmbedding(
        model_name="nomic-ai/nomic-embed-text-v1.5", 
        trust_remote_code=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_folder=".cache",
    )

embedding = get_embedding()

# Set up llama-index settings
Settings.embed_model = embedding
Settings.llm = llm

# Load and index documents
@st.cache_resource
def load_and_index_documents():
    documents = LlamaParse(result_type="markdown", num_workers=8).load_data(
        "./data/art.pdf"
    )
    documents = [doc for doc in documents if doc.text.strip()]  # Filter out empty documents
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    return index

index = load_and_index_documents()

# Create query engine and tool
query_engine = index.as_query_engine(llm=llm)
seduction_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        description="A RAG Engine with expertise in seduction of people based on Art Of Seduction book",
        name="seduction_tool"
    )
)

# Initialize ReActAgent
@st.cache_resource
def get_agent():
    return ReActAgent.from_tools([seduction_tool], llm=llm, verbose=True, max_iterations=-1)

agent = get_agent()

# Streamlit UI for chat
st.subheader("Chat with the Document")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your question?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = agent.chat(prompt)
        message_placeholder.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

st.sidebar.header("About")
st.sidebar.info("This is a Document Q&A system using llama-index and Streamlit. It allows you to chat with the content of the loaded document in a Q&A pattern.")
