import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv


# Load environment variables from the .env file
load_dotenv()

# Get the OpenAI API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI client
client = OpenAI(api_key=openai_api_key)

# Function to preprocess the problem specification with GPT-4
def preprocess_problem_spec(problem_spec):
    # Prompt with specific instructions on how to preprocess the problem specification
    messages = [
        {"role": "system", "content": "You are a helpful assistant that preprocesses coding problem specifications to make them more intuitive."},
        {
            "role": "user",
            "content": f"""
            I want you to preprocess a problem specification and make it more intuitive and straightforward, similar to the following format:

            ### Title:
            Give a clear and concise title that captures the problem's main focus.

            ### Description:
            Provide a short description of the objective of the problem, focusing on what the user needs to achieve. Avoid excessive details, keeping it concise and clear.

            ### Input and Output:
            - Clearly define the input format, breaking it down by each operation or command.
            - Clearly define the expected output format, specifying any edge cases or conditions where no output might be required.

            ### Examples:
            Provide sample inputs and outputs that illustrate typical cases, using simple text blocks. Include explanations if necessary, but keep them short and to the point.

            ### Constraints:
            List any important constraints on input size, value ranges, or other rules the solution must follow.

            ---
            Apply this structure to the following problem specification:

            {problem_spec}
            """
        }
    ]
    
    # Make the API call to OpenAI with GPT-4 model
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )


    # Extract the generated message from the API response
    preprocessed_spec = completion.choices[0].message.content
    return preprocessed_spec


# Function to preprocess the problem specification with GPT-4
def preprocess_problem_spec(problem_spec):
    # Prompt for problem specification preprocessing
    messages = [
        {"role": "system", "content": "You are a helpful assistant that preprocesses problem specifications to make them more intuitive."},
        {
            "role": "user",
            "content": f"""
            I want you to preprocess a problem specification and make it more intuitive and straightforward, similar to the following format:

            ### Title:
            Give a clear and concise title that captures the problem's main focus.

            ### Description:
            Provide a short description of the objective of the problem, focusing on what the user needs to achieve. Avoid excessive details, keeping it concise and clear.

            ### Input and Output:
            - Clearly define the input format, breaking it down by each operation or command.
            - Clearly define the expected output format, specifying any edge cases or conditions where no output might be required.

            ### Examples:
            Provide sample inputs and outputs that illustrate typical cases, using simple text blocks. Include explanations if necessary, but keep them short and to the point.

            ### Constraints:
            List any important constraints on input size, value ranges, or other rules the solution must follow.

            ---
            Apply this structure to the following problem specification:

            {problem_spec}
            """
        }
    ]
    
    # Make the API call to OpenAI with GPT-4 model
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )

    # Extract the generated message from the API response
    preprocessed_spec = completion.choices[0].message.content
    return preprocessed_spec

# Function to analyze the code for potential logical errors
def analyze_code_errors(problem_spec, user_code):
    prompt_ver1 = f"""
    You are given a problem specification and a piece of code that attempts to solve it. Your task is to analyze the code and point out where the user might have made logical errors based on the problem requirements without yapping. Focus on common issues like:

    - **Wrong choice of data structures**: 
        - Are the data structures used (e.g., lists, dictionaries, queues, etc.) appropriate for the problem?
        - Is the user correctly utilizing the chosen data structure's properties?

    - **Incorrect implementation of the data structure**:
        - Are operations like insertions, deletions, or lookups implemented correctly according to the problem requirements?

    - **Faulty logic in handling edge cases**:
        - Are there any missing conditions or incorrect handling of cases like empty lists, specific input sizes, or the final output format?

    - **Failure to meet problem constraints**:
        - Is the code violating any constraints such as time/space complexity, order of operations, or the problem’s specific rules?
    """
    prompt_ver2 = f"""
    You are given a problem specification and a piece of code that attempts to solve it. Your task is to analyze the code and point out where the user might have made logical errors based on the problem requirements without yapping.
    """


    # Prompt for analyzing the user's code based on the problem specification
    messages = [
        {"role": "system", "content": "You are an expert programmer who analyzes code for logical errors based on a given problem specification."},
        {
            "role": "user",
            "content": f"""
            You are given a problem specification and a piece of code that attempts to solve it. Your task is to analyze the code and point out where the user might have made logical errors based on the problem requirements. Focus on common issues like:

              1. **Wrong choice of data structures**: 
                  - Are the data structures used (e.g., lists, dictionaries, queues, etc.) appropriate for the problem?
                  - Is the user correctly utilizing the chosen data structure's properties?

              2. **Incorrect implementation of the data structure**:
                  - Are operations like insertions, deletions, or lookups implemented correctly according to the problem requirements?

              3. **Faulty logic in handling edge cases**:
                  - Are there any missing conditions or incorrect handling of cases like empty lists, specific input sizes, or the final output format?

              4. **Failure to meet problem constraints**:
                  - Is the code violating any constraints such as time/space complexity, order of operations, or the problem’s specific rules?
            ---
            Problem Specification:
            {problem_spec}

            Code:
            ```python
            {user_code}
            ```

            IMPORTANT: DO NOT PRINT CODE, YOU SHOULD ONLY GIVE HINTS PARAGRAPH(S)
            """
        }
    ]
    
    # Make the API call to OpenAI with GPT-4 model
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )

    # Extract the generated message from the API response
    analysis = completion.choices[0].message.content
    return analysis


# Creating two columns: one for the Problem Specification and one for the Code
col1, col2 = st.columns(2)



# Title and description of the app
st.title("Problem-Code Matcher")
# Creating two columns: one for the Problem Specification and one for the Code
col1, col2 = st.columns(2)

# Column for Problem Specification input
with col1:
  st.subheader("Problem Specification")
  problem_spec = st.text_area("Paste the problem specification here", height=400)

# Column for Code input
with col2:
  st.subheader("User Code")
  user_code = st.text_area("Paste the user code here", height=400)

# Simple button to start the comparison
if st.button("Match"):
    if problem_spec and user_code:
        # Call the preprocessing function for the problem specification
        with st.spinner("Processing the problem specification..."):
            preprocessed_spec = preprocess_problem_spec(problem_spec)
        
        # Display the preprocessed problem specification
        # st.subheader("Preprocessed Problem Specification")
        # st.write(preprocessed_spec)

        # Call the function to analyze the code for errors
        with st.spinner("Analyzing the user code..."):
            code_analysis = analyze_code_errors(preprocessed_spec, user_code)

        # Display the code analysis
        st.subheader("Code Analysis")
        st.write(code_analysis)
    else:
        st.error("Please enter both a problem specification and user code.")