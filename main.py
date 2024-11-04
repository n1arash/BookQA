import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.tools import ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from pprint import pprint
import torch
from llama_index.core.agent import ReActAgent
from dotenv import load_dotenv
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool
from llama_parse import LlamaParse

load_dotenv()

if torch.cuda.is_available():
    print("USING CUDA GPU")
else:
    print("USING CPU.")

os.environ["GROQ_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "pr-frosty-talent-74"
os.environ["LLAMA_CLOUD_API_KEY"] = ""

llm = Groq(model="llama3-70b-8192", api_key=os.environ["GROQ_API_KEY"])


embedding = HuggingFaceEmbedding(
    model_name="nomic-ai/nomic-embed-text-v1.5", 
    trust_remote_code=True,
    device="cuda",
    cache_folder=".cache",
)

Settings.embed_model = embedding
Settings.llm = llm

documents = LlamaParse(result_type="markdown", num_workers=8).load_data(
    "./data/art.pdf"
)

documents = [doc for doc in documents if doc.text.strip()]  # Filter out empty documents

index = VectorStoreIndex.from_documents(documents, show_progress=True)

query_engine = index.as_query_engine(llm=llm)
seduction_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        description="A RAG Engine with expertise in seduction of people based on Art Of Seduction book",
        name="seduction_tool"
    )
)

agent = ReActAgent.from_tools([seduction_tool], llm=llm, verbose=True, max_iterations=20)

response = agent.chat(input("Question: "))
print(response)
