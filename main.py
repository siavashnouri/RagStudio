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
from src.query_translation import MultiQuery, RAGFusion
vector_client = QdrantClient()
# vector_client.create_collection(
#     collection_name="test_collection",
#     vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
# )
llm= ChatOllama(model="qwen3.5:4b")
embedding_model = NVIDIAEmbeddings(model="nvidia/nv-embedcode-7b-v1", api_key="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")
mongo_db_store = MongoDBDocStore.from_connection_string(connection_string="mongodb://admin:admin@localhost:27017/",namespace="RAG.docstore")
# documents = load_data()
vector_store = QdrantVectorStore(client=vector_client, collection_name="test_collection", embedding=embedding_model)
raptor = RAPTOR(llm=llm, vector_store=vector_store, byte_store=mongo_db_store)
# multi_representation = MultiRepresentation(
#     llm=llm,
#     vector_store=vector_store,
#     byte_store=mongo_db_store

# )
# raptor.index(documents=documents)
# # multi_representation.index(documents=documents)
# user_input = "give me a example of code with Deep Agent with Gemini"
# documents = multi_representation.retriever.invoke(user_input)
# retrieved_context = [x.page_content for x in documents]
# x = vector_store.add_texts(["hi how are you", "how are you"], metadatas=[{"parent": 1}, {"parent": 2}])

# x = vector_store.similarity_search_with_score(query="how can i implement a deep agent code", filter=Filter(must=[FieldCondition(key="metadata.level", range=Range(gte=2))]))
# print (x[1])
# from sklearn.cluster import DBSCAN

# x = raptor.retriever.invoke("how can i implement a deep agent")
# print(x)
# # # clustering = DBSCAN()
# # # clustering.fit([c.vector for c in x])
# # # print(clustering.labels_)
def context_merge(contexts:list[Document]):
    contexts = [c for context in contexts for c in context]
    contexts = [d.page_content for d in contexts]
    contexts = list(set(contexts))
    context = "\n".join(contexts)
    return context
template = "please answer **ONLY** base on this contexts:\n\n {context}\n\n question is: {question}"
# retriever_chain = {"question": MultiQuery(llm=llm)} | raptor.retriever.map() | context_merge

chain = {"context": RAGFusion(llm=llm, retriever=raptor.retriever), "question": RunnablePassthrough()} | ChatPromptTemplate.from_template(template=template) | llm | StrOutputParser()
print(chain.invoke("what is the backend in Deep Agent and what is for"))
