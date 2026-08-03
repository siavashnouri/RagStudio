from .query_translation_abc import QueryTranslation
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class StepBack(QueryTranslation):
    PROMPT_TEMPLATE = "please answer **ONLY** base on this contexts:\n\n {original_context}\n\n {step_back_context}\n\n question is: {question}"

    def __init__(self, llm, retriever:Runnable):
        self._llm=llm
        self._retriever = retriever



    def _build_chain(self):
        template = """
            You are an expert at query reformulation for Retrieval-Augmented Generation (RAG).

            Your task is to generate ONE step-back question.

            Examples:

            Original:
            How do I optimize HNSW parameters in Qdrant for 10 million vectors?

            Step-back:
            What factors affect the performance and accuracy of approximate nearest neighbor indexes?

            ---

            Original Question:
            {question}

            Step-back Question:
            """

        template = ChatPromptTemplate.from_template(template=template)

        step_back_retriever_chain = {"question": RunnablePassthrough()} | template | self._llm | StrOutputParser() | self._retriever
        # original_retriever_chain = {"question": RunnablePassthrough()} | | self._retriever
        chain = {"question": RunnablePassthrough(), 'step_back_context': step_back_retriever_chain, "original_context": self._retriever} | ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE) | self._llm | StrOutputParser()
        return chain