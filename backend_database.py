from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3

llm = ChatOllama(model="llama3")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
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

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

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