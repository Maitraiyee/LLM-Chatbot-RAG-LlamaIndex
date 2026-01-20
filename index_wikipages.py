from llama_index.readers.wikipedia import WikipediaReader # read wiki pages
from llama_index.core.indices.vector_store import VectorStoreIndex # in memory vector store
from llama_index.core.node_parser import SentenceSplitter # split text during preprocessing 
from llama_index.program.openai import OpenAIPydanticProgram # get structured op from openai api call
from pydantic import BaseModel # data_validation 
import openai 
from utils import get_apikey
# define the data model in pydantic
class WikiPageList(BaseModel):
    "Data model for WikiPageList"
    pages: list


def wikipage_list(query):
    openai.api_key = get_apikey()
    
    prompt_template_str = """
    Given the input {query}, extract the Wikipedia pages mentioned after "please index:" and return them as a list.
    If only one page is mentioned, return a single element list.
    """

    # we dont need to in general use openAi pydantic, we can just extract the list ourselves as well 
    # we just use openAiPydantic for structured extraction when the data is supermessy, so that we can extract with help of llm
    program = OpenAIPydanticProgram.from_defaults(
        output_cls = WikiPageList,
        prompt_template_str= prompt_template_str,
        verbose = True
        )

    wikipage_requests = program(query=query)
    return wikipage_requests


def create_wikidocs(wikipage_requests):
    reader = WikipediaReader()
    documents = reader.load_data(pages=wikipage_requests.pages) # takes list[str] as input and we used openaiapi earlier to convert the query to list 
    return documents


def create_index(query):
    global index
    wikipage_requests = wikipage_list(query)
    documents = create_wikidocs(wikipage_requests)
    
    # in order to split the sentences we need obj of SenteceSplitter class, define chunk size and chunk overlap 
    text_splits = SentenceSplitter(chunk_size = 150, chunk_overlap = 45)

    # we need doc parsing into nodes, we are pasing the chunks into nodes, define nodes using the sentencesplitter obj  
    nodes = text_splits.get_nodes_from_documents(documents)

    # after parsing we need to index the parsed docs 
    index = VectorStoreIndex(nodes)
    return index

if __name__ == "__main__":
    query = "/get wikipages: paris, lagos, lao"
    # wikipage_requests = wikipage_list(query)
    # documents = create_wikidocs(wikipage_requests)
    index = create_index(query)
    print("INDEX CREATED", index)
