from . import QueryTranslation
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class MultiQuery(QueryTranslation):
    def __init__(self,llm,  message_count:int=5):
        self.__message_count = message_count
        self.__llm = llm
        super().__init__()



    def invoke(self, input, config = None, **kwargs):
        template = """You are an AI language model assistant. Your task is to generate 5 
                different versions of the given user question to retrieve relevant documents from a vector 
                database. By generating multiple perspectives on the user question, your goal is to help
                the user overcome some of the limitations of the distance-based similarity search.

                Provide these alternative questions separated by newlines.

                Original question: {question}"""
        template = ChatPromptTemplate.from_template(template=template)
        chain = {"question": RunnablePassthrough()} | template |self.__llm | StrOutputParser() | (lambda x: x.split("\n"))
        return chain.invoke(input)
