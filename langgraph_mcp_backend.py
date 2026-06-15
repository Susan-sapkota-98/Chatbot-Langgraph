# langgraph_mcp_backend.py
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient  # ← NEW
from dotenv import load_dotenv
import asyncio, threading, sqlite3, sys, os

load_dotenv()


# ── Background event loop (lets async MCP work inside sync Streamlit) ─────────
_loop = asyncio.new_event_loop()
_thread = threading.Thread(target=_loop.run_forever, daemon=True)
_thread.start()

def run_async(coro):
    """Run an async coroutine safely from sync code."""
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


# ── Start MCP client and load tools ──────────────────────────────────────────
_mcp_client = MultiServerMCPClient({
    "my_tools": {
        "transport": "stdio",
        "command": sys.executable,                          # same Python interpreter
        "args": [os.path.join(os.path.dirname(__file__), "tools_server.py")],
        "env": {**os.environ},                             # passes SERP_API_KEY etc.
    }
})

async def _start_mcp():
    return await _mcp_client.get_tools()

tools = run_async(_start_mcp())          # blocks until MCP server is ready


# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatOllama(model="qwen2.5:7b")
llm_with_tools = llm.bind_tools(tools)


# ── Graph state ───────────────────────────────────────────────────────────────
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ── SQLite checkpointer ────────────────────────────────────────────────────────
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
conn.execute('''
    CREATE TABLE IF NOT EXISTS thread_names (
        thread_id TEXT PRIMARY KEY,
        name TEXT DEFAULT 'New Chat'
    )
''')
conn.commit()
checkpointer = SqliteSaver(conn=conn)


# ── Build graph ───────────────────────────────────────────────────────────────
tool_node = ToolNode(tools)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges(
    "chat_node",
    tools_condition,
    {"tools": "tools", END: END}
)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


# ── Thread helpers (unchanged) ────────────────────────────────────────────────
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])

    result = []
    for thread_id in all_threads:
        row = conn.execute(
            "SELECT name FROM thread_names WHERE thread_id=?", (thread_id,)
        ).fetchone()

        if row and row[0] and row[0] != "New Chat":
            name = row[0]
        else:
            state = chatbot.get_state(
                config={"configurable": {"thread_id": thread_id}}
            )
            messages = state.values.get("messages", [])
            first_human = next(
                (m.content for m in messages if isinstance(m, HumanMessage)), None
            )
            if first_human:
                name = first_human[:40] + ("..." if len(first_human) > 40 else "")
                conn.execute(
                    "INSERT OR REPLACE INTO thread_names (thread_id, name) VALUES (?, ?)",
                    (thread_id, name)
                )
                conn.commit()
            else:
                name = "New Chat"

        result.append((thread_id, name))
    return result


def save_thread_name(thread_id, name):
    conn.execute(
        "INSERT OR REPLACE INTO thread_names (thread_id, name) VALUES (?, ?)",
        (str(thread_id), name)
    )
    conn.commit()