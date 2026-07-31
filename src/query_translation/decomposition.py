
from .query_translation_abc import QueryTranslation
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
class Decomposition(QueryTranslation):

    def __init__(self, llm, retriever: Runnable, steps: int=3):
        self._llm=llm
        self._retriever = retriever
        self._step=steps
        super().__init__()


    def _build_chain(self):
        chain = self._decompose_query_chain() | RunnableLambda(self._answer_subquestions)
        return chain


    def _decompose_query_chain(self) -> Runnable:

        template = """You are a helpful assistant that generates multiple sub-questions related to an input question.\n
        The goal is break down the input to a set of sub-problem / sub-question that can be answers in isolation. \n
        Separate each sub-question by newline (\n)
        Generate multiple search queries related to : {question}
        Output ({step})
        """

        template = ChatPromptTemplate.from_template(template=template)

        chain = {"question": RunnablePassthrough(), "step": lambda x: self._step} | template | self._llm | StrOutputParser() | (lambda x : x.split("\n"))
        return chain

    def _answer_subquestions(self,sub_questions:list[str]) -> str:
        template = """Answer the question based on `context` and question-answer pair.\n
                    main question is : {question}\n\n
        
                    Here is any available background question-answer pair\n
                    \n {q_a_pair}\n
        
                    \n
                    here is the additional context:\n
                    {context}
                """
        q_a_pair_string = ""
        for question in sub_questions:
            chain = {"question": itemgetter("question"), "q_a_pair": itemgetter("q_a_pair"), "context": itemgetter("question")|self._retriever} | ChatPromptTemplate.from_template(template)|self._llm |StrOutputParser()
            result = chain.invoke({"question": question, "q_a_pair": q_a_pair_string})
            q_a_pair_string += self._format_qa_pair(question=question, answer=result)

        return result


    def _format_qa_pair(self, question:str, answer:str) -> str:

        return f"Question:{question}\nAnswer:{answer}"

