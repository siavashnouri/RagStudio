from abc import ABC, abstractmethod
from langchain_core.runnables import Runnable

class QueryTranslation(Runnable):



    @abstractmethod
    def _build_chain(self) -> Runnable:
        pass




    
    def invoke(self, input, config = None, **kwargs):
        return self._build_chain().invoke(input, config=config, kwargs=kwargs)

    async def ainvoke(self, input, config = None, **kwargs):
        return await self._build_chain().ainvoke(input, config=config, kwargs=kwargs)