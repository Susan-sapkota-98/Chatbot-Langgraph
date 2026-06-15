# rag_backend.py
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
from queue import Queue
import asyncio, threading, sqlite3, aiosqlite, sys, os, tempfile, chromadb

load_dotenv()

CHROMA_PATH = "./chroma_db"


# ── Background event loop ─────────────────────────────────────────────────────
_loop = asyncio.new_event_loop()
_thread = threading.Thread(target=_loop.run_forever, daemon=True)
_thread.start()

def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


# ── MCP client ────────────────────────────────────────────────────────────────
_mcp_client = MultiServerMCPClient({
    "my_tools": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [os.path.join(os.path.dirname(__file__), "tools_server.py")],
        "env": {**os.environ},
    }
})

# ── Sync SQLite for thread_names only ────────────────────────────────────────
sync_conn = sqlite3.connect("chatbot.db", check_same_thread=False)
sync_conn.execute('''
    CREATE TABLE IF NOT EXISTS thread_names (
        thread_id TEXT PRIMARY KEY,
        name TEXT DEFAULT 'New Chat'
    )
''')
sync_conn.commit()

# ── Async init: MCP tools + AsyncSqliteSaver ─────────────────────────────────
async def _init_async():
    tools = await _mcp_client.get_tools()
    aio_conn = await aiosqlite.connect("chatbot.db")
    checkpointer = AsyncSqliteSaver(aio_conn)
    await checkpointer.setup()
    return tools, checkpointer

tools, checkpointer = run_async(_init_async())


# ── LLM + graph ───────────────────────────────────────────────────────────────
llm = ChatOllama(model="qwen2.5:7b")
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

tool_node = ToolNode(tools)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition, {"tools": "tools", END: END})
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


# ── RAG ingestion ─────────────────────────────────────────────────────────────
def ingest_documents(uploaded_files: list) -> dict:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    vectorstore = Chroma(
        client=client,
        collection_name="documents",
        embedding_function=embeddings
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    summary = {"total_chunks": 0, "files": []}

    for uploaded_file in uploaded_files:
        is_pdf = uploaded_file.type == "application/pdf"
        suffix = ".pdf" if is_pdf else ".txt"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path) if is_pdf else TextLoader(tmp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
            chunks = splitter.split_documents(docs)
            vectorstore.add_documents(chunks)
            summary["files"].append({"name": uploaded_file.name, "chunks": len(chunks)})
            summary["total_chunks"] += len(chunks)
        finally:
            os.unlink(tmp_path)

    return summary


def list_ingested_sources() -> list[str]:
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection("documents")
        results = collection.get(include=["metadatas"])
        sources = sorted({m.get("source", "Unknown") for m in results["metadatas"]})
        return sources
    except Exception:
        return []


# ── Thread helpers ─────────────────────────────────────────────────────────────
async def _retrieve_all_threads_async():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])

    result = []
    for thread_id in all_threads:
        row = sync_conn.execute(
            "SELECT name FROM thread_names WHERE thread_id=?", (thread_id,)
        ).fetchone()

        if row and row[0] and row[0] != "New Chat":
            name = row[0]
        else:
            state = await chatbot.aget_state(
                config={"configurable": {"thread_id": thread_id}}
            )
            messages = state.values.get("messages", [])
            first_human = next(
                (m.content for m in messages if isinstance(m, HumanMessage)), None
            )
            if first_human:
                name = first_human[:40] + ("..." if len(first_human) > 40 else "")
                sync_conn.execute(
                    "INSERT OR REPLACE INTO thread_names (thread_id, name) VALUES (?, ?)",
                    (thread_id, name)
                )
                sync_conn.commit()
            else:
                name = "New Chat"

        result.append((thread_id, name))
    return result


def retrieve_all_threads():
    return run_async(_retrieve_all_threads_async())


def get_conversation_state(thread_id):
    return run_async(
        chatbot.aget_state(config={"configurable": {"thread_id": thread_id}})
    )


def save_thread_name(thread_id, name):
    sync_conn.execute(
        "INSERT OR REPLACE INTO thread_names (thread_id, name) VALUES (?, ?)",
        (str(thread_id), name)
    )
    sync_conn.commit()


# ── Async stream bridge ───────────────────────────────────────────────────────
def stream_graph(inputs: dict, config: dict):
    q = Queue()

    async def _run():
        try:
            async for chunk in chatbot.astream(inputs, config, stream_mode="messages"):
                q.put(chunk)
        except Exception as e:
            q.put(e)
        finally:
            q.put(None)

    asyncio.run_coroutine_threadsafe(_run(), _loop)

    while True:
        item = q.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item