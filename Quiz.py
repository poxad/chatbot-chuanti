import time
import os
import streamlit as st
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd
# quiz_questions = [
#     [
#         'Which image represents Reinforcement Learning?',
#         None,  # No text options
#         ['assets/1a.jpeg', 'assets/1b.jpeg', 'assets/1c.jpeg', 'assets/1d.jpeg'],  # Image paths for options
#         'Option A',  # Correct answer (image path)
#     ],
#     [
#         'What type of learning uses historical data to make predictions?',
#         ['Supervised', 'Unsupervised', 'Reinforcement', 'None of the above'],  # Text options (4 options)
#         None,  
#         'Supervised'  # Correct answer (text)
#     ],
# ]

# Initialize session state for answers and submission status
# if 'answers' not in st.session_state:
#     st.session_state.answers = [None] * len(quiz_questions)
if 'submitted' not in st.session_state:
    st.session_state.submitted = False

if 'disabled' not in st.session_state:
    st.session_state.disabled = False

if 'sampled_df' not in st.session_state:
    # Streamlit app
    df = pd.read_csv("assets/question_datasets.csv", encoding='ISO-8859-1')
    df = df.iloc[:-1]  # Select all rows except the last one
    st.session_state.sampled_df = df.sample(n=5)  # Store sampled questions in session state

st.title("❓ Data Structures Quiz")
st.divider()
# test = st.multiselect(
#             "Select your answers:",
#             ["Line 51: if (curr -> next -> data ==val){", "Line 52: Node* temp = curr-> next"],
#             label_visibility="collapsed",
#         )
# df = pd.read_csv("assets/question_datasets.csv", encoding='ISO-8859-1')
# st.write(test)
# for i in test:
#     st.write(i)
# st.write(df["answer"][5].split(","))
# st.write(test == df["answer"][5].split(","))

sampled_df = st.session_state.sampled_df
question, question_image, a, b, c, d, e, f, g, h, answer, answer_alpha = sampled_df.T.values
correct_counter = 0
row_count = sampled_df.shape[0]
for i in range(row_count):
    if pd.notna(question_image[i]):
        image_path = "assets/" + question_image[i]

    if pd.notna(question[i]) and pd.isna(question_image[i]):
        st.write(question[i])

    elif pd.notna(question_image[i]) and pd.isna(question[i]):
        st.image(image_path)

    elif pd.notna(question[i]) and pd.notna(question_image[i]):
        st.write(question[i])
        st.image(image_path)

    options = [a[i], b[i], c[i], d[i], e[i], f[i], g[i], h[i]]
    # Filter out None values from the options
    filtered_options = [opt for opt in options if pd.notna(opt)]

    # User input handling
    if len(answer_alpha[i]) == 1:
        user_choice_radio = st.radio(
            "Select your answer:",
            filtered_options,
            index=None,
            label_visibility="collapsed",
            disabled=st.session_state.disabled,
            key=f"radio_{i}"
        )
    else:
        user_choice_multiselect = st.multiselect(
            "Select your answers:",
            filtered_options,
            label_visibility="collapsed",
            disabled=st.session_state.disabled,
            key=f"multiselect_{i}"
        )

    # Show feedback if submitted
    if st.session_state.submitted:
        st.write(answer[i])
        if len(answer_alpha[i].split(',')) == 1:
            if user_choice_radio == answer[i]:
                st.success("Correct", icon="✅")
                correct_counter += 1
            else:
                st.error("Incorrect", icon="❌")
        else:
            if user_choice_multiselect == answer[i].split(","): 
                st.success("Correct", icon="✅")
                correct_counter += 1
            else:
                st.error("Incorrect", icon="❌")
    st.write(f"Answer key: {answer[i]}")
    st.divider()

# Submit button logic
if st.button("Submit", type="primary", disabled=st.session_state.disabled) and not st.session_state.submitted:
    st.session_state.disabled = True
    st.session_state.submitted = True  # Set the flag to True upon submission
    st.rerun()  # Rerun to reflect the submission immediately

# Evaluate the answers (results are already shown under each question)
if st.session_state.submitted:
    st.header(f"Quiz completed! You got {correct_counter} out of {row_count} questions correct.")
    if st.button("Restart Quiz"):
        st.session_state.disabled = False
        st.session_state.submitted = False 
        st.session_state.sampled_df = pd.read_csv("assets/question_datasets.csv", encoding='ISO-8859-1').sample(n=5) # Clear sampled questions to trigger resampling
        st.rerun()  # Restart the quiz and allow new sampling



# for index, (question, options, images, correct_answer) in enumerate(quiz_questions):
#     st.subheader(f"**{question}**", divider="gray")

#     if images is not None:
#         cols = st.columns(4)  # Create 4 columns for image options
#         selected_option = st.radio(
#             "Select your answer:",
#             ['Option A', 'Option B', 'Option C', 'Option D',],
#             key=f"answer_{index}",
#             index=None,
#             label_visibility="collapsed",
#             disabled=st.session_state.disabled
#         )

#         for i, image in enumerate(images):
#             with cols[i]:
#                 st.image(image, caption=f"Option {chr(65 + i)}", use_column_width=True)  # Use A, B, C, D as option labels
#     else:
#         selected_option = st.radio(
#             "Select your answer:",
#             options,
#             key=f"answer_{index}",
#             index=None,
#             label_visibility="collapsed",
#             disabled=st.session_state.disabled
#         )

#     # Store the selected answer in the session state
#     st.session_state.answers[index] = selected_option
#     # Show the result only after submission
#     if st.session_state.submitted:
#         selected_answer = st.session_state.answers[index]
#         if selected_answer == correct_answer:
#             st.success(f"Correct", icon="✅")
#         else:
#             st.error(f"Incorrect", icon="❌")

# # Submission button logic
# st.divider()
# if st.button("Submit", type="primary",disabled=st.session_state.disabled) and not st.session_state.submitted:
#     st.session_state.disabled=True
#     st.session_state.submitted = True  # Set the flag to True upon submission
#     st.rerun()  # Rerun to reflect the submission immediately

# # Evaluate the answers (results are already shown under each question)
# if st.session_state.submitted:
#     correct_count = sum(
#         1 for index, (question, options, images, correct_answer) in enumerate(quiz_questions)
#         if st.session_state.answers[index] == correct_answer
#     )

#     st.header(f"Quiz completed! You got {correct_count} out of {len(quiz_questions)} questions correct.")
#     if st.button("Restart Quiz"):
#         st.session_state.disabled=False
#         st.session_state.submitted = False 
#         st.rerun() 
