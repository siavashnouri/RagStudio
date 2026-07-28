
from . import Indexing
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableConfig
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from uuid import uuid4


class MultiRepresentation(Indexing):
    
    def __init__(self, llm, vector_store, byte_store):
        self.__llm = llm
        self.__vector_store = vector_store
        self.__byte_store = byte_store
        self.__retriever = MultiVectorRetriever(
            docstore=byte_store,
            vectorstore=vector_store,
            id_key="doc_id"

        )




    def index(self, documents:list[Document]):
        summary_template = "give a summary of this blow document: \n\n{doc}"
        chain: RunnableSequence = (
            {"doc": lambda d: d.page_content}
            | ChatPromptTemplate.from_template(template=summary_template)
            | self.__llm
            | StrOutputParser()
        )
        summaries = chain.batch(inputs= documents, config={"max_concurrency": 4})
        # id_key="doc_id"
        # retriever = MultiVectorRetriever(
        #     docstore=self.__byte_store,
        #     vectorstore=self.__vector_store,
        #     id_key=id_key
        # )
        doc_ids = [str(uuid4()) for _ in documents]
        
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(summaries)
            ]
    
        self.__retriever.vectorstore.add_documents(summary_docs)
        self.__retriever.docstore.mset(list(zip(doc_ids, documents)))
        # return retriever

    
    def evaluate(self):
        return "coming soon"




    @property
    def llm(self):
        return self.__llm

    @property
    def vector_store(self):
        return self.__vector_store
    
    @property
    def byte_store(self):
        return self.__byte_store

    @property
    def retriever(self):
        return self.__retriever


if __name__=="__main__":
    pass