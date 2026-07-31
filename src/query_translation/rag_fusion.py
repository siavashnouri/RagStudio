from .multi_query import MultiQuery
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
class RAGFusion(MultiQuery):

    def __init__(self, llm, retriever, message_count = 5, document_count=5):
        self._document_count = document_count
        super().__init__(llm, retriever, message_count)
    def _reciprocal_rank_fusion(self, results:list[list[Document]]):

        doc_score = {}
        for documents in results:
            for rank, document in enumerate(documents, start=1):
                doc_content = document.page_content
                doc_score[doc_content] = doc_score.get(doc_content, 0) + 1.0 /(60 +rank)


        ranked_doc = sorted(doc_score.items(), key=lambda x: x[1], reverse=True)
        ranked_doc = [x[0] for x in ranked_doc[:self._document_count]]
        return "\n\n---\n\n".join(ranked_doc)

        
    def _build_chain(self):
        """Constructs the static LCEL execution pipeline."""
        template = """You are an AI language model assistant. Your task is to generate 5
                different versions of the given user question to retrieve relevant documents from a vector 
                database. By generating multiple perspectives on the user question, your goal is to help
                the user overcome some of the limitations of the distance-based similarity search.

                Provide these alternative questions separated by newlines.

                Original question: {question}"""
        template = ChatPromptTemplate.from_template(template=template)
        retriever_chain = {"question": RunnablePassthrough()} | template | self._llm | StrOutputParser() | self._clean_and_split | self._retriever.map() | self._reciprocal_rank_fusion
        chain = {"context": retriever_chain, "question": RunnablePassthrough()} | ChatPromptTemplate.from_template(template=self.TEMPLATE) | self._llm | StrOutputParser()
        return chain