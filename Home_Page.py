import streamlit as st
from streamlit_option_menu import option_menu
import yaml
import streamlit as st
from yaml.loader import SafeLoader
from st_pages import add_page_title, get_nav_from_toml


st.markdown("""
    <style>
    .stButton button {
        width: 100%;
        height: 100px;
        font-size: 50px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)
# make_sidebar()
st.markdown("### 🤖 Data Structures - Chen Yi Shin")
# st.info("Repository on [Github](https://github.com/poxad/all-in-one-chatbot.git)")
# st.image("assets/1a.jpg")
st.markdown("---")
# Example prompts
example_prompts = [
    "👨‍💻 Chatbot",
    "❓ Quiz",
]

button_cols = st.columns(2)

if button_cols[0].button(example_prompts[0]):
    st.switch_page("Chatbot.py")
if button_cols[1].button(example_prompts[1]):
    st.switch_page("Quiz.py")
# Add created by text
# st.markdown('''
#     <p style="font-size: 20px;">
#     Created by <a href="https://www.jasonjonarto.com" style="text-decoration: underline; color: gray;">Jason Jonarto</a>
#     </p>
# ''', unsafe_allow_html=True)
