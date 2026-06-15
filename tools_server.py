# tools_server.py
from mcp.server.fastmcp import FastMCP
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from serpapi import GoogleSearch
from dotenv import load_dotenv
import chromadb, os

load_dotenv()
mcp = FastMCP("MyTools")

CHROMA_PATH = "./chroma_db"


def get_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return Chroma(
        client=client,
        collection_name="documents",
        embedding_function=embeddings
    )


@mcp.tool()
def rag_search(query: str) -> str:
    """
    Search through user-uploaded documents to answer questions.
    Use this when the user asks about documents they have uploaded.
    """
    try:
        vectorstore = get_vectorstore()
        results = vectorstore.similarity_search(query, k=4)

        if not results:
            return "No relevant content found in uploaded documents."

        output = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "Unknown")
            output.append(f"[{i}] Source: {source}\n{doc.page_content}")

        return "\n\n".join(output)

    except Exception as e:
        return f"RAG search error: {str(e)}"


@mcp.tool()
def google_search(query: str) -> str:
    """Search Google using SerpAPI."""
    params = {"q": query, "api_key": os.getenv("SERP_API_KEY")}
    search = GoogleSearch(params)
    results = search.get_dict()

    if "organic_results" not in results:
        return "No results found"

    output = []
    for item in results["organic_results"][:5]:
        output.append(
            f"Title: {item.get('title', '')}\n"
            f"Snippet: {item.get('snippet', '')}\n"
            f"Link: {item.get('link', '')}\n"
        )
    return "\n\n".join(output)


@mcp.tool()
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_stock_price(symbol: str) -> str:
    """
    Fetch latest stock price for a given NEPSE stock symbol (e.g. 'NABIL', 'NICA', 'NLIC').
    """
    params = {
        "q": f"{symbol.upper()} NEPSE stock price today site:sharesansar.com OR site:hamroshare.com.np OR site:merolagani.com",
        "api_key": os.getenv("SERP_API_KEY")
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    if "organic_results" not in results:
        return f"No stock data found for symbol '{symbol}'"

    output = []
    for item in results["organic_results"][:3]:
        output.append(
            f"Title: {item.get('title', '')}\n"
            f"Snippet: {item.get('snippet', '')}\n"
            f"Link: {item.get('link', '')}\n"
        )
    return "\n\n".join(output)


if __name__ == "__main__":
    mcp.run()