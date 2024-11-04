import time
import os
import joblib
import streamlit as st
import pandas as pd
from pathlib import Path
import openai
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import datetime

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY
new_chat_id = f'{time.time()}'
MODEL_ROLE = 'ai'
AI_AVATAR_ICON = '🧑🏽‍💻'

# Create a data/ folder if it doesn't already exist
os.makedirs('data/', exist_ok=True)

image_link=[]

try:
    past_chats = joblib.load('data/past_chats_list')
    knowledge_graph = joblib.load('data/graph_data')
except:
    past_chats = {}
    knowledge_graph = {}
    
    

def get_conversational_chain():
    prompt_template = """
    You are a customer Professional Programming Professor. Use the uploaded files to give detailed explanations about programming queries without unnecessary elaboration. 
    If the user asks a question unrelated to data structures or programming, respond as a basic chatbot.
    WARNING: 
    - DO NOT EVER OUTPUT ANY KIND OF CODE, YOU CAN ONLY OUTPUT PSEUDOCODE.
    - YOU MUST ALWAYS SEARCH FROM THE UPLOADED FILES TO ENRICH YOUR RESPONSE.
    
    Given a user input, categorize it into one of the following topics: Problem Definition, Concept Explanation, Step-by-step Guidance, or Pseudo Code Visualization, and provide a response based on the categorization. The chatbot should output only one topic at a time and cannot output more than one topic simultaneously.

    For Problem Definition:
    - If the user input is related to understanding the problem statement, what the question is asking for, or the context of the problem, provide a detailed explanation in bullet points.
    
    For Concept Explanation:
    - If the user input is related to understanding a specific concept or algorithm, explain it in detail and provide examples if necessary.
    
    For Step-by-step Guidance:
    - If the user input is asking for a step-by-step guide or algorithm for solving a problem, provide detailed instructions.
    
    For Pseudo Code Visualization:
    - If the user input is asking for a pseudo code representation of an algorithm, provide a clear pseudo code explanation.
    
    Context:\n {context}\n
    Question:\n {question}\n
    Knowledge:\n{knowledge_graph}\n

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question","knowledge_graph"])
    return prompt

def user_input(user_question, context, knowledge_graph):
    prompt = get_conversational_chain()
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt.format(context=context, question=user_question,knowledge_graph =knowledge_graph)},
            {"role": "user", "content": user_question},
        ],
    )
    return response.choices[0].message['content']

@st.dialog("Clear chat history?")
def modal():
    button_cols = st.columns([1, 1])  # Equal column width for compact fit
    
    # Add custom CSS for button styling
    st.markdown(
        """
        <style>
        .stButton button {
            width: 100%;
            padding: 10px;
            font-size: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if button_cols[0].button("Yes"):
        clear_chat_history()
        st.rerun()
    elif button_cols[1].button("No"):
        st.rerun()
        
def clear_chat_history():
    st.session_state.pop('chat_id', None)
    st.session_state.pop('messages', None)
    
    for file in Path('data/').glob('*'):
        file.unlink()

# Sidebar allows a list of past chats
with st.sidebar:
    st.write('# Sidebar Menu')
    
    if st.session_state.get('chat_id') is None:
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id] + list(past_chats.keys()),
            format_func=lambda x: past_chats.get(x, 'New Chat'),
            placeholder='_',
        )
    else:
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id, st.session_state.chat_id] + list(past_chats.keys()),
            index=1,
            format_func=lambda x: past_chats.get(x, 'New Chat' if x != st.session_state.chat_id else st.session_state.chat_title),
            placeholder='_',
        )
    if st.button("Clear Chat History", key="clear_chat_button"):
        modal()
    
    st.session_state.chat_title = f'OPENAI-{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}'
    
st.title('🧑🏽‍💻 Chat with OpenAI GPT')

# Chat history (allows to ask multiple questions)
try:
    st.session_state.messages = joblib.load(f'data/{st.session_state.chat_id}-st_messages')
except:
    st.session_state.messages = []

# Check if 'messages' is not in st.session_state and initialize with a default message
if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "🧑🏽‍💻",  # or any valid emoji
        "content": "Hey there, I'm your OpenAI chatbot. Feel free to ask any questions regarding Data Structures to me."
    })

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(
        name=message.get('role', 'user'),
        avatar=message.get('avatar', None),
    ):
        st.markdown(message['content'])

if prompt := st.chat_input('Your message here...'):
    # Display user message in chat message container
    if st.session_state.chat_id not in past_chats.keys():
        past_chats[st.session_state.chat_id] = st.session_state.chat_title
        joblib.dump(past_chats, 'data/past_chats_list')
    
    with st.chat_message('user'):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append(
        dict(
            role='user',
            content=prompt,
        )
    )

    with st.spinner("Waiting for AI response..."):
        response = user_input(prompt, "Add context from uploaded files if available.")  # Update context handling if necessary

    # Display assistant response in chat message container
    with st.chat_message(
        name=MODEL_ROLE,
        avatar=AI_AVATAR_ICON,
    ):
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append(
        dict(
            role=MODEL_ROLE,
            content=response,
            avatar=AI_AVATAR_ICON,
        )
    )
    
    # Save to file
    joblib.dump(st.session_state.messages, f'data/{st.session_state.chat_id}-st_messages')
