import streamlit as st
from rag_backend import (
    chatbot, retrieve_all_threads, save_thread_name,
    ingest_documents, list_ingested_sources,
    stream_graph, get_conversation_state
)
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
import uuid


def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    st.session_state['current_chat_named'] = False
    add_thread(thread_id, 'New Chat')
    st.session_state['message_history'] = []


def add_thread(thread_id, name='New Chat'):
    thread_id_str = str(thread_id)
    existing_ids = [t[0] for t in st.session_state['chat_threads']]
    if thread_id_str not in existing_ids:
        st.session_state['chat_threads'].insert(0, (thread_id_str, name))


def load_conversation(thread_id):
    state = get_conversation_state(thread_id)   # ← replaces chatbot.get_state()
    return state.values.get('messages', [])


def update_chat_name(thread_id, name):
    thread_id_str = str(thread_id)
    for i, (tid, _) in enumerate(st.session_state['chat_threads']):
        if tid == thread_id_str:
            st.session_state['chat_threads'][i] = (tid, name)
            break
    save_thread_name(thread_id_str, name)


def stream_response(user_input, config):
    full_response = ""
    status_container = None

    tool_labels = {
        "google_search":   ("🔍", "Searching the web..."),
        "get_stock_price": ("📈", "Fetching stock price..."),
        "calculator":      ("🧮", "Calculating..."),
        "rag_search":      ("📄", "Searching documents..."),  # ← NEW
    }

    for message_chunk, metadata in stream_graph(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    ):
        node = metadata.get("langgraph_node", "")

        if (
            node == "chat_node"
            and isinstance(message_chunk, AIMessage)
            and message_chunk.tool_calls
        ):
            tool_name = message_chunk.tool_calls[0].get("name", "")
            icon, label = tool_labels.get(tool_name, ("⚙️", f"Running {tool_name}..."))
            args = message_chunk.tool_calls[0].get("args", {})
            detail = ", ".join(f"{k}: {v}" for k, v in args.items())

            status_container = st.status(f"{icon} {label}", expanded=True)
            status_container.write(f"**Tool:** `{tool_name}`")
            if detail:
                status_container.write(f"**Input:** {detail}")

        if isinstance(message_chunk, ToolMessage) and status_container:
            status_container.update(label="✅ Done", state="complete", expanded=False)
            status_container = None

        if (
            node == "chat_node"
            and message_chunk.content
            and not isinstance(message_chunk, ToolMessage)
            and not (isinstance(message_chunk, AIMessage) and message_chunk.tool_calls)
        ):
            full_response += message_chunk.content
            yield message_chunk.content

    if status_container:
        status_container.update(label="✅ Done", state="complete", expanded=False)


# ── Session setup ─────────────────────────────────────────────────────────────
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    raw = retrieve_all_threads()
    normalized = []
    for item in raw:
        if isinstance(item, tuple):
            normalized.append(item)
        else:
            normalized.append((str(item), 'New Chat'))
    st.session_state['chat_threads'] = normalized

if 'current_chat_named' not in st.session_state:
    st.session_state['current_chat_named'] = False

add_thread(st.session_state['thread_id'], 'New Chat')

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title('Langgraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()
    st.rerun()

# ── RAG Upload Section ────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.header('📄 Document Upload')

uploaded_files = st.sidebar.file_uploader(
    'Upload PDF or TXT files',
    type=['pdf', 'txt'],
    accept_multiple_files=True
)

if uploaded_files:
    if st.sidebar.button('⬆️ Ingest Documents'):
        with st.sidebar:
            with st.spinner('Embedding and storing...'):
                summary = ingest_documents(uploaded_files)
            st.success(
                f"✅ Ingested **{summary['total_chunks']}** chunks "
                f"from **{len(summary['files'])}** file(s)"
            )
            for f in summary['files']:
                st.caption(f"📎 {f['name']} → {f['chunks']} chunks")

# Show already-ingested documents
sources = list_ingested_sources()
if sources:
    with st.sidebar.expander(f'📚 Ingested Documents ({len(sources)})'):
        for s in sources:
            st.caption(f'• {s}')

# ── Conversations ─────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.header('My Conversations')

for thread_id, name in st.session_state['chat_threads']:
    display_name = name if name and name != 'New Chat' else 'New Chat'
    if st.sidebar.button(display_name, key=thread_id):
        st.session_state['thread_id'] = thread_id
        st.session_state['current_chat_named'] = True
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages
        st.rerun()

# ── Render conversation ───────────────────────────────────────────────────────
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input('Type here')

if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    CONFIG = {
        'configurable': {'thread_id': st.session_state['thread_id']},
        'metadata': {'thread_id': st.session_state['thread_id']},
        'run_name': 'chat_turn'
    }

    with st.chat_message('assistant'):
        ai_message = st.write_stream(stream_response(user_input, CONFIG))

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

    if not st.session_state['current_chat_named']:
        chat_name = user_input[:40] + ('...' if len(user_input) > 40 else '')
        update_chat_name(st.session_state['thread_id'], chat_name)
        st.session_state['current_chat_named'] = True
        st.rerun()