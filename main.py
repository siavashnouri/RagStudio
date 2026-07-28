from src.data_loading import load_data, URLS
from src.vector_store.indexing import MultiRepresentation
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_mongodb.docstores import MongoDBDocStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_nvidia import NVIDIAEmbeddings

from openai import OpenAI
vector_client = QdrantClient()
# vector_client.create_collection(
#     collection_name="demo_collection",
#     vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
# )
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
llm= ChatOllama(model="qwen3.5:4b")
embedding_model = NVIDIAEmbeddings(model="nvidia/nv-embedcode-7b-v1", api_key="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")
mongo_db_store = MongoDBDocStore.from_connection_string(connection_string="mongodb://admin:admin@localhost:27017/",namespace="RAG.docstore")
# documents = load_data()
vector_store = QdrantVectorStore(client=vector_client, collection_name="demo_collection", embedding=embedding_model)
multi_representation = MultiRepresentation(
    llm=llm,
    vector_store=vector_store,
    byte_store=mongo_db_store

)
# multi_representation.index(documents=documents)
user_input = "give me a example of code with Deep Agent with Gemini"
documents = multi_representation.retriever.invoke(user_input)
retrieved_context = [x.page_content for x in documents]
