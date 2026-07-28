from abc import ABC, abstractmethod
from langchain_core.documents import Document
class Indexing(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def index(self, documents:list[Document]):
        pass
    
    @abstractmethod
    def evaluate(self):
        pass
