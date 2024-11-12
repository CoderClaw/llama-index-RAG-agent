from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv


load_dotenv()

llm = Ollama(model="llama3.2:1b", request_timeout=300.0)

parser = LlamaParse(result_type="markdown")

file_extractor = {".pdf": parser}

documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

text_splitter = SentenceSplitter(chunk_size=500,chunk_overlap=50)
Settings.text_splitter = text_splitter

embed_model = resolve_embed_model("local:BAAI/bge-m3")

vector_index = VectorStoreIndex.from_documents(documents, transformations=[text_splitter], embed_model=embed_model, show_progress=True)

query_engine = vector_index.as_query_engine(llm=llm)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    retries = 0

    while retries < 3:
        try:
            response = query_engine.query(prompt)
            print(response)
            break
        except Exception as e:
            retries += 1
            print(f"Error occured, retry #{retries}:", e)

    if retries >= 3:
        print("Unable to process request, try again...")
        continue


