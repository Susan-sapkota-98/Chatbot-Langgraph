import streamlit as st
from langgraph_mcp_backend import chatbot, retrieve_all_threads, save_thread_name
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
    state = chatbot.get_state(
        config={'configurable': {'thread_id': thread_id}}
    )
    return state.values.get('messages', [])


def update_chat_name(thread_id, name):
    thread_id_str = str(thread_id)
    for i, (tid, _) in enumerate(st.session_state['chat_threads']):
        if tid == thread_id_str:
            st.session_state['chat_threads'][i] = (tid, name)
            break
    save_thread_name(thread_id_str, name)


def stream_response(user_input, config):
    """
    Streams the assistant response, showing a status container
    whenever a tool (web search, stock lookup, calculator) is in use.
    Returns the full assistant text response.
    """
    full_response = ""
    status_container = None

    # Map tool names to friendly display labels
    tool_labels = {
        "google_search":   ("🔍", "Searching the web..."),
        "get_stock_price": ("📈", "Fetching stock price..."),
        "calculator":      ("🧮", "Calculating..."),
    }

    for message_chunk, metadata in chatbot.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="messages"
    ):
        node = metadata.get("langgraph_node", "")

        # ── Tool call detected: open a status container ──────────────────
        if (
            node == "chat_node"
            and isinstance(message_chunk, AIMessage)
            and message_chunk.tool_calls
        ):
            tool_name = message_chunk.tool_calls[0].get("name", "")
            icon, label = tool_labels.get(tool_name, ("⚙️", f"Running {tool_name}..."))

            # Show tool arguments as extra detail inside the status box
            args = message_chunk.tool_calls[0].get("args", {})
            detail = ", ".join(f"{k}: {v}" for k, v in args.items())

            status_container = st.status(f"{icon} {label}", expanded=True)
            status_container.write(f"**Tool:** `{tool_name}`")
            if detail:
                status_container.write(f"**Input:** {detail}")

        # ── Tool result received: close the status container ─────────────
        if isinstance(message_chunk, ToolMessage) and status_container:
            status_container.update(label="✅ Done", state="complete", expanded=False)
            status_container = None

        # ── Stream final assistant text ───────────────────────────────────
        if (
            node == "chat_node"
            and message_chunk.content
            and not isinstance(message_chunk, ToolMessage)
            and not (isinstance(message_chunk, AIMessage) and message_chunk.tool_calls)
        ):
            full_response += message_chunk.content
            yield message_chunk.content

    # Close status if still open (edge case)
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

# ── Render conversation history ───────────────────────────────────────────────
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