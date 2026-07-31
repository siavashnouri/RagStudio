from abc import ABC, abstractmethod
from langchain_core.runnables import Runnable

class QueryTranslation(Runnable):
    PROMPT_TEMPLATE = "please answer **ONLY** base on this contexts:\n\n {context}\n\n question is: {question}"


    @abstractmethod
    def _build_chain(self) -> Runnable:
        pass




    
    def invoke(self, input, config = None, **kwargs):
        return self._build_chain().invoke(input, config=config, kwargs=kwargs)

    async def ainvoke(self, input, config = None, **kwargs):
        return await self._build_chain().ainvoke(input, config=config, kwargs=kwargs)