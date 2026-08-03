from . import Indexing
from langchain_core.documents import Document
from pydantic import BaseModel
from typing import Literal
from sklearn.cluster import DBSCAN, KMeans
from langchain_core.vectorstores import VectorStore
from langchain_core.stores import BaseStore
from uuid import uuid4
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from qdrant_client.models import Filter, FieldCondition, Range
from langchain_qdrant import QdrantVectorStore
class Node(BaseModel):
    id:str
    text:str
    level:int
    children:list[str]|None=None
    embedding:list[float] |None = None
    metadata:dict = {}
    cluster:int|None= None


class RAPTOR(Indexing):

    def __init__(self,llm, vector_store:VectorStore,byte_store:BaseStore, levels:int=3, cluster_method:Literal["DBSCAN", "KMeans"]="DBSCAN"):
        match cluster_method:
            case "DBSCAN":
                self.__clustering = DBSCAN(min_samples=2)
            case "KMeans":
                self.__clustering = KMeans()
        self.__levels = levels
        self.__llm = llm
        self.__vector_store = vector_store
        self.__byte_store = byte_store
        if isinstance(self.__vector_store, QdrantVectorStore):
            self.__retriever = RAPTORRetrieverQdrant(vector_store=self.__vector_store, byte_store=self.__byte_store, top_k=3)
        else:
            raise Exception(f"retriever {self.__retriever} (instance of {type(self.__retriever)}) is not supported")

        

    def index(self,documents:list[Document]):

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_text = "\n".join([doc.page_content for doc in documents])
        splitted = splitter.split_text(all_text)

        nodes = {
            node.id: node for node in [Node(id=str(uuid4()), text=t, level=1) for t in splitted]
        }


        #save chunks in database
        self.__byte_store.mset([(node.id, Document(page_content=node.text)) for node in nodes.values()])


        ids = self.__vector_store.add_texts([node.text for node in nodes.values()], metadatas=[{"level": node.level, "children": node.children,"node_id":node.id} for node in nodes.values()])
        new_nodes = nodes.copy()
        for level in range(2, self.__levels+1):
            # self.__vector_store.embeddings.embed_documents()
            embeddings = self.__vector_store.client.retrieve(collection_name="test_collection", ids= ids, with_vectors=True)
            self.__clustering.fit([e.vector for e in embeddings])

            # set label for each node
            for doc, label in zip(embeddings, self.__clustering.labels_):
                node = new_nodes[doc.payload["metadata"]["node_id"]]
                node.cluster = label



            new_nodes = self.__build_level__(new_nodes.values(), level=level)
            ids = self.__vector_store.add_texts([node.text for node in new_nodes.values()], metadatas=[{"level": node.level, "children": node.children, "node_id": node.id} for node in new_nodes.values()])

    @property
    def retriever(self):
        return self.__retriever

        
        
        
        
        


    def __build_level__(self, nodes:list[Node], level:int) -> dict[str,Node]:
        summary_template = "give a summary of this blow document: \n\n{doc}"
        summary_template = ChatPromptTemplate.from_template(summary_template)
        new_nodes = []
        texts = {node.cluster: {"text": "", "children": []} for node in nodes}
        for node in nodes:
            texts[node.cluster]["text"] += node.text
            if node.children:
                texts[node.cluster]["children"] = node.children
            else:
                texts[node.cluster]["children"].append(node.id)
        chain = RunnableSequence({"doc": RunnablePassthrough()} | summary_template | self.__llm | StrOutputParser())
        new_nodes = []
        for cluster, value in texts.items():
            summary = chain.invoke(input=value.get('text'))
            # summary = chain.batch(inputs=value.get("text"),config={"max_concurrency": 4})
            new_nodes.append(Node(text=summary, children=value.get("children"), level=level, id=str(uuid4()), cluster=int(cluster)))
        # new_nodes = [Node(id=uuid4, text=summary,parent=node.id, level=level) for summary,node in zip(summary, nodes)]
        new_nodes = { node.id: node for node in new_nodes}
        return new_nodes

            
        pass

    def __embed_documents__(self, nodes: list[Node]):
        pass

    def __summarize_documents__(self, nodes:list[Node]):
        pass


    def __cluster_documents__(self, nodes:list[Node]):
        pass

    def evaluate(self):
        return "coming soon"





class RAPTORRetrieverQdrant (BaseRetriever):
    vector_store:VectorStore
    byte_store:BaseStore
    top_k:int=3
    def _get_relevant_documents(self, query, *, run_manager):
        print(query)
        documents = self.vector_store.similarity_search(query=query, filter=Filter(must=[FieldCondition(key="metadata.level", range=Range(gte=2))]), k=self.top_k)
        ids = []
        for doc in documents:
            ids.extend(doc.metadata['children'])

        ids = list(set(ids))
        documents = self.byte_store.mget(ids)
        return documents

    async def _aget_relevant_documents(self, query, *, run_manager):
        documents = await self.vector_store.asimilarity_search(query=query, filter=Filter(must=[FieldCondition(key="metadata.level", range=Range(gte=2))]), k=self.top_k)
        ids = []
        for doc in documents:
            ids.extend(doc.metadata['children'])

        ids = list(set(ids))
        documents = await self.byte_store.amget(ids)
        return documents

        

        