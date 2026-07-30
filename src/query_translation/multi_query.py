from . import QueryTranslation
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.documents import Document

class MultiQuery(QueryTranslation):
    def __init__(self,llm, retriever:Runnable,  message_count:int=5):
        self._message_count = message_count
        self._retriever = retriever
        self._llm = llm
        super().__init__()

    def _clean_and_split(self, text: str) -> list[str]:
        """Cleans LLM output, removes empty lines or numbered prefixes."""
        lines = text.strip().split("\n")
        questions = []
        for line in lines:
            cleaned = line.strip()
            if cleaned and cleaned[0].isdigit() and "." in cleaned[:3]:
                cleaned = cleaned.split(".", 1)[-1].strip()
            if cleaned:
                questions.append(cleaned)
        return questions[:self._message_count]

    def _context_merge(self, contexts:list[Document]):
        """Flattens list of document lists, deduplicates, and formats to string."""
        contexts = [c for context in contexts for c in context]
        contexts = [d.page_content for d in contexts]
        contexts = list(set(contexts))
        context = "\n".join(contexts)
        return context

    def _build_chain(self):
        """Constructs the static LCEL execution pipeline."""
        template = """You are an AI language model assistant. Your task is to generate {count}
                different versions of the given user question to retrieve relevant documents from a vector 
                database. By generating multiple perspectives on the user question, your goal is to help
                the user overcome some of the limitations of the distance-based similarity search.

                Provide these alternative questions separated by newlines.

                Original question: {question}"""
        template = template.format(count=self._message_count)
        template = ChatPromptTemplate.from_template(template=template)
        chain = {"question": RunnablePassthrough()} | template |self._llm | StrOutputParser() | self._clean_and_split | self._retriever.map() | self._context_merge
        return chain