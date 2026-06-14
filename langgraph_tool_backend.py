from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from serpapi import GoogleSearch
from langchain_core.tools import tool
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import requests
import os


load_dotenv()
llm = ChatOllama(model="qwen2.5:7b")
@tool
def google_search(query: str) -> str:
    """
    Search Google using SerpAPI.
    """
    params = {
        "q": query,
        "api_key":os.getenv("SERP_API_KEY")
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    if "organic_results" not in results:
        return "No results found"

    output = []

    for item in results["organic_results"][:5]:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        link = item.get("link", "")

        output.append(
            f"Title: {title}\n"
            f"Snippet: {snippet}\n"
            f"Link: {link}\n"
        )

    return "\n\n".join(output)

@tool
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
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}
    
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given NEPSE symbol (e.g. 'NABIL', 'NICA', 'NLIC').
    Uses the official Nepal Stock Exchange API.
    """
    try:
        # Search for the security by symbol
        search_url = f"https://nepalstock.com.np/api/nots/security/{symbol.upper()}"
        headers = {"Accept": "application/json"}
        r = requests.get(search_url, headers=headers, timeout=10)
        data = r.json()

        if not data:
            return {"error": f"No data found for symbol '{symbol}'"}

        return {
            "symbol": data.get("symbol"),
            "name": data.get("securityName"),
            "last_traded_price": data.get("lastTradedPrice"),
            "percent_change": data.get("percentageChange"),
            "open": data.get("openPrice"),
            "high": data.get("highPrice"),
            "low": data.get("lowPrice"),
            "close": data.get("closePrice"),
            "volume": data.get("totalTradeQuantity"),
        }

    except Exception as e:
        return {"error": str(e)}



tools = [google_search, get_stock_price, calculator]
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)

conn.execute('''
    CREATE TABLE IF NOT EXISTS thread_names (
        thread_id TEXT PRIMARY KEY,
        name TEXT DEFAULT 'New Chat'
    )
''')
conn.commit()

checkpointer = SqliteSaver(conn=conn)
tool_node = ToolNode(tools)


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges(
    "chat_node",
    tools_condition
)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

def retrive_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    result = []
    for thread_id in all_threads:
        row = conn.execute(
            "SELECT name FROM thread_names WHERE thread_id=?", (thread_id,)
        ).fetchone()

        if row and row[0] and row[0] != 'New Chat':
            name = row[0]
        else:
            state = chatbot.get_state(
                config={'configurable': {'thread_id': thread_id}}
            )
            messages = state.values.get('messages', [])
            first_human = next(
                (m.content for m in messages if isinstance(m, HumanMessage)), None
            )
            if first_human:
                name = first_human[:40] + ('...' if len(first_human) > 40 else '')
                conn.execute(
                    "INSERT OR REPLACE INTO thread_names (thread_id, name) VALUES (?, ?)",
                    (thread_id, name)
                )
                conn.commit()
            else:
                name = 'New Chat'

        result.append((thread_id, name))
    return result

def save_thread_name(thread_id, name):
    conn.execute(
        "INSERT OR REPLACE INTO thread_names (thread_id, name) VALUES (?, ?)",
        (str(thread_id), name)
    )
    conn.commit()