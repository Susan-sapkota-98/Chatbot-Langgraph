import streamlit as st
from langgraph_tool_backend import chatbot, retrive_all_threads, save_thread_name
from langchain_core.messages import HumanMessage
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

# session setup
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    raw = retrive_all_threads()
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

# sidebar UI
st.sidebar.title('Langgraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()
    st.rerun()

st.sidebar.header('My Conversations')

# unpacks (thread_id, name) and shows name
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

# render conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']},
              'metadata':{
                  'thread_id':st.session_state['thread_id']
              },
              'run_name':'chat_turn'
              }

    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content
            for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages'
            )
            if message_chunk.content
        )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

    # Name the chat after the first message
    if not st.session_state['current_chat_named']:
        chat_name = user_input[:40] + ('...' if len(user_input) > 40 else '')
        update_chat_name(st.session_state['thread_id'], chat_name)
        st.session_state['current_chat_named'] = True
        st.rerun()