from src.data_loading import load_data, URLS
from src.vector_store.indexing import MultiRepresentation, RAPTOR,RAPTORRetrieverQdrant
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_mongodb.docstores import MongoDBDocStore
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range
from qdrant_client.http.models import Distance, VectorParams
from langchain_nvidia import NVIDIAEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from src.query_translation import MultiQuery, RAGFusion, Decomposition, StepBack
from dotenv import load_dotenv

load_dotenv(".env")
vector_client = QdrantClient()

llm= ChatOllama(model="qwen3.5:4b")
embedding_model = NVIDIAEmbeddings(model="nvidia/nv-embedcode-7b-v1", api_key="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")
mongo_db_store = MongoDBDocStore.from_connection_string(connection_string="mongodb://admin:admin@localhost:27017/",namespace="RAG.docstore")
# documents = load_data()
vector_store = QdrantVectorStore(client=vector_client, collection_name="test_collection", embedding=embedding_model)
raptor = RAPTOR(llm=llm, vector_store=vector_store, byte_store=mongo_db_store)


template = "please answer **ONLY** base on this contexts:\n\n {context}\n\n question is: {question}"
chain = StepBack(llm=llm, retriever=raptor.retriever)
print(chain.invoke("what is the backend in Deep Agent and what is for"))
