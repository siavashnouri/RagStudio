from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_core.documents import Document
URLS = (
    "https://docs.langchain.com/oss/python/deepagents/overview",
    "https://docs.langchain.com/oss/python/deepagents/quickstart",
    "https://docs.langchain.com/oss/python/deepagents/customization",
    "https://docs.langchain.com/oss/python/deepagents/models",
    "https://docs.langchain.com/oss/python/deepagents/comparison",
    "https://docs.langchain.com/oss/python/deepagents/tools",
    "https://docs.langchain.com/oss/python/deepagents/backends",
    "https://docs.langchain.com/oss/python/deepagents/permissions",
    

)


def load_data(urls:list[str]= URLS) -> list[Document]:
    loader = WebBaseLoader(
        web_path=urls,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                id=("content", "header") #Scrap only the element with id of `content` and `header`
            )
        ),

    )
    
    docs = loader.load()
    return docs




if __name__=="__main__":
    docs = load_data.load()
    print(docs[0].page_content)