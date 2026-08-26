from dotenv import load_dotenv
load_dotenv()

from ingest import load_documents, build_index, build_vector_index
from rag_helper import RAGBase
from tools import RecallTools
from agent import RecallAgent

docs = load_documents()
index = build_index(docs)
vector_index = build_vector_index(docs)

rag = RAGBase(index=index, vector_index=vector_index, retrieval_mode="hybrid", use_reranking=True)
tools = RecallTools(rag=rag, documents=docs)
agent = RecallAgent(recall_tools=tools)

result = agent.ask("What models of cribs have been recalled?", verbose=True)

print("\nANSWER:")
print(result["answer"])
print("\nTOOL CALLS MADE:")
print(result["tool_calls"])