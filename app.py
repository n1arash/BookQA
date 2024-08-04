import os
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.tools import ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
import torch
from llama_index.core.agent import ReActAgent
# from dotenv import load_dotenv
from llama_index.core.tools import QueryEngineTool
from llama_index.vector_stores.elasticsearch import ElasticsearchStore

# Load environment variables
# load_dotenv()

# Set up Streamlit page configuration
st.set_page_config(page_title="Document Q&A", page_icon="📚", layout="wide")
st.title("Document Q&A System")

# Check for CUDA availability
if torch.cuda.is_available():
    st.success("Using CUDA GPU")
else:
    st.warning("Using CPU")

# Set up environment variables
os.environ["GROQ_API_KEY"] = "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" # Groq API
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-QjpOoj9QxyW3SfC9R3SKbuJZNAuwp9u8fY5467aDsk3OOH1e" # Llama Parser

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
    documents = SimpleDirectoryReader(input_dir="./data").load_data()
    documents = [doc for doc in documents if doc.text.strip()]  # Filter out empty documents
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    return index

index = load_and_index_documents()

# Create query engine and tool
query_engine = index.as_query_engine(llm=llm)
networking_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="networking_tool",
        description="A comprehensive Retrieval-Augmented Generation (RAG) Engine specialized in computer networks, protocols, architectures, and cybersecurity. It provides in-depth knowledge on topics such as TCP/IP, OSI model, routing, switching, wireless networks, network security, and emerging technologies like SDN and 5G. Ideal for answering technical questions, explaining concepts, and providing practical insights on network design and troubleshooting."
    )
)

# Initialize ReActAgent
@st.cache_resource
def get_agent():
    return ReActAgent.from_tools([networking_tool], llm=llm, verbose=True, max_iterations=-1)

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
