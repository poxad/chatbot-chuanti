import time
import os
import joblib
import streamlit as st
import pandas as pd
from pathlib import Path
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import datetime
from PIL import Image
# import image_ocr
import easyocr
import numpy as np
import pdf2image
from io import BytesIO
import glob
import time

load_dotenv()

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
api_key = GOOGLE_API_KEY
new_chat_id = f'{time.time()}'
MODEL_ROLE = 'ai'
AI_AVATAR_ICON = '✨'

# Create a data/ folder if it doesn't already exist
os.makedirs('data/', exist_ok=True)

image_link=[]

try:
    past_chats = joblib.load('data/past_chats_list')
except:
    past_chats = {}

def get_conversational_chain():
    prompt_template = """
    You are a customer Professional Programming Professor. Use the uploaded files to give detail exlanation to everything about programming queries without yapping. 
    If the user asks a question that is not related to data structures or programming, answer like a normal basic chatbot (ChatGPT).
    WARNING: 
    - DO NOT EVER OUTPUT ANY KIND OF CODE, YOU CAN ONLY OUTPUT PSEUDOCODE.
    - YOU MUST ALWAYS SEARCH FROM THE UPLOADED FILES TO ENRICH YOUR RESPONSE.
            
    Given a user input, categorize it into one of the following topics: Problem Definition, Concept Explanation, Step-by-step Guidance, or Pseudo Code Visualization, and provide a response based on the categorization. The chatbot should output only one topic at a time and cannot output more than one topic simultaneously.

    For Problem Definition:
    - If the user input is related to understanding the problem statement, what the question is asking for, or the context of the problem, provide a detailed explanation in bullet points.
    - Answer in the following format:
        1. Objective of the question
        2. Inputs
        3. Constraints (if any)
        4. Output
        5. Different types of approaches for the problem starting from the best to worst according to complexity.
        6. Explain briefly the pros and cons for each approach.
        7.  Explain possible edge cases and warnings of what might go wrong so that the user can have an idea.
    - Format the output neatly with a heading for each point and subheading for child points. Do not provide code or implementation ideas.
    
    For Concept Explanation:
    - If the user input is related to understanding a specific concept or algorithm, explain it in detail and provide examples if necessary.
    - Answer in the following format:
        1. What is the concept
        2. Different Types of this approach and identify which approach to use in this question
        3. Time Complexity

    - Format the output neatly with a heading for each point and subheading for child points. Do not provide code or implementation ideas.
    
    For Step-by-step Guidance:
    - If the user input is asking for a step-by-step guide or algorithm for solving a problem, provide detailed instructions.
    - Answer in the following format:
        1. Step by step implementation
    - Format the output neatly with a heading for each point and subheading for child points. Do not provide code or implementation ideas.
    
    For Pseudo Code Visualization:
    - If the user input is asking for a pseudo code representation of an algorithm, provide a clear pseudo code explanation.
    - Answer in the following format:
        1. Code Visualization
    - Format the output neatly with a heading for each point and subheading for child points. Do not provide code or implementation ideas.
    - Please input some comments  in the pseudocode for better readability
        REMEMBER THAT Pseudocode ia simply an implementation of an algorithm in the form of annotations and informative TEXT WRITTEN IN PLAIN ENGLISH. It has NO syntax like any of the programming language.
        IMPORTANT: YOU CAN ONLY GIVE CODE PSEUDOCODE FORMAT.

    VERY IMPORTANT: Along with the response, suggest a set of close related questions that users can pick formatted with numbers so that it can be easier to read. 

    
    Context:\n {context}\n
    Question:\n {question}\n

    Answer:
    """
    # "The question you asked is not available in the context, did you mean ... ?" or you could
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local(r"faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]


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
    st.session_state.pop('gemini_history', None)
    
    for file in Path('data/').glob('*'):
        file.unlink()

# Sidebar allows a list of past chats
with st.sidebar:
    st.write('# Sidebar Menu')
    
    if st.session_state.get('chat_id') is None:
        # st.write(past_chats.keys())
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id] + list(past_chats.keys()),
            format_func=lambda x: past_chats.get(x, 'New Chat'),
            placeholder='_',
        )
        # st.write(st.session_state)
    else:
        # This will happen the first time AI response comes in
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id, st.session_state.chat_id] + list(past_chats.keys()),
            index=1,
            format_func=lambda x: past_chats.get(x, 'New Chat' if x != st.session_state.chat_id else st.session_state.chat_title),
            placeholder='_',
        )
    if st.button("Clear Chat History", key="clear_chat_button"):
        # st.write(st.session_state)
        modal()
    
    
    st.session_state.chat_title = f'PDF-{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}'

st.title('🤖 Chat with Gemini')


# Chat history (allows to ask multiple questions)
try:
    st.session_state.messages = joblib.load(f'data/{st.session_state.chat_id}-st_messages')
    st.session_state.gemini_history = joblib.load(f'data/{st.session_state.chat_id}-gemini_messages')
except:
    st.session_state.messages = []
    st.session_state.gemini_history = []

st.session_state.model = genai.GenerativeModel('gemini-pro')
st.session_state.chat = st.session_state.model.start_chat(history=st.session_state.gemini_history)

# Check if 'messages' is not in st.session_state and initialize with a default message
if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "✨",  # or any valid emoji
        "content": "Hey there, I'm your Text Extraction chatbot. Please upload the necessary files in the sidebar to add more context to this conversation."
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
        response = user_input(prompt, api_key)

            
        
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
    st.session_state.gemini_history = st.session_state.chat.history
    # Save to file
    joblib.dump(st.session_state.messages, f'data/{st.session_state.chat_id}-st_messages')
    joblib.dump(st.session_state.gemini_history, f'data/{st.session_state.chat_id}-gemini_messages')
