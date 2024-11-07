import time
import os
import joblib
import streamlit as st
import pandas as pd
from pathlib import Path
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import datetime
import easyocr
from PyPDF2 import PdfReader
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)
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

# st.write(knowledge_graph)
def get_conversational_chain():
    prompt_template = """
    You are a Programming Professor specializing in Data Structures, tasked with providing detailed, contextually aware responses based on both the knowledge graph (past chats) and uploaded files.

    Guidelines:
    - **Primary Information Source**: First, consult the knowledge graph (past chats) for any relevant information on the user’s current query. If the topic has been previously discussed, use that context to provide continuity and avoid redundancy.
    - **Supplementary Information**: Use the uploaded files for additional context or new details only if the knowledge graph does not fully address the query.
    - **Code Restriction**: DO NOT output actual code; provide only PSEUDOCODE when necessary.
    - **Topic Categorization**: Based on the user’s question, categorize it into one of the following: Problem Definition, Concept Explanation, Step-by-step Guidance, or Pseudo Code Visualization. Respond based on the selected category only.

    **Categories & Response Structure**:
    1. **Problem Definition**:
        - If the query relates to understanding a problem, list out the main requirements in bullet points.
        - First, check the knowledge graph for any relevant prior discussions on this problem. Reference those points to provide consistency or elaborate on what was previously discussed.

    2. **Concept Explanation**:
        - For questions about specific concepts or algorithms, provide a detailed explanation, using examples if helpful.
        - Check the knowledge graph first for any related topics (e.g., if the user previously asked about the "Shortest Path Problem," include insights from that discussion). Supplement with details from uploaded files if necessary.

    3. **Step-by-step Guidance**:
        - If the user is asking for a solution approach, outline a detailed step-by-step process.
        - Use steps or strategies discussed in prior conversations from the knowledge graph first, building upon them with any additional information from the uploaded files as needed.

    4. **Pseudo Code Visualization**:
        - When pseudocode is requested, generate a clear version.
        - First, check for any pseudocode or related discussions in past chats and reference them to maintain continuity. Use new file information only if no relevant past context is found.

    **Response Priorities**:
    - **Prioritize Past Chats**: Always consult the knowledge graph (past chats) as the primary source. This maintains continuity, builds on previous interactions, and prevents redundant explanations.
    - Supplement responses with uploaded files only if additional information is needed.
    - Limit responses to programming and data structure-related queries. For unrelated topics, respond briefly as a basic chatbot.

    ---

    **Check Knowledge Graph for Related Query**:
    Before answering the question, ask the user:
    - "Do you recall implementing this [topic] previously? For example, have you worked on a hashtable before? We may have already discussed it."Then Explain it.
    
    If the user confirms, respond based on the context of the knowledge graph.

    Context from Knowledge Graph (Past Chats):\n{knowledge_graph}\n
    Additional Context from Uploaded Files:\n{context}\n
    Question:\n{question}\n

    Lastly 3 reccomended questions that is very closely related to this current user's query in order for user to learn deeper about the current topic.

    Answer:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["knowledge_graph", "context", "question"])
    return prompt


def user_input(user_question, knowledge_graph):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    new_db = FAISS.load_local(r"faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    prompt = get_conversational_chain()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt.format(context=docs, question=user_question, knowledge_graph=knowledge_graph)},
            {"role": "user", "content": user_question},
        ],
    )
    return response.choices[0].message.content

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
    
st.title('🧑🏽‍💻 Chat with AI Tutor')

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
        response = user_input(prompt, knowledge_graph)  # Update context handling if necessary

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