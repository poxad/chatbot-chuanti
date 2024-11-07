import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
import time
import datetime
# from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from neo4j_connection import Neo4jConnection  # Import the connection class
import joblib


# Load environment variables from the .env file
load_dotenv()

# Get the OpenAI API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI client
client = OpenAI(api_key=openai_api_key)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Initialize Neo4jGraph
conn = Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
# st.success("Connection to Neo4j successful!")

try:
    st.session_state.messages = joblib.load(f'data/{time.time()}-problemspec')
except:
    st.session_state.messages = []



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
def analyze_code_errors(problem_spec, user_code, knowledge_graph):
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

    prompt_ver3 = f"""
    Based on the problem specification and user code below, identify the main type of logical error present in the code with one sentence. The possible types of errors include:

    - **Wrong data structure choice**: The code uses an inappropriate data structure for the requirements.
    - **Incorrect implementation**: An error in how the data structure or algorithm is implemented.
    - **Edge case handling error**: Failure to handle an important edge case (like empty inputs or boundary values).
    - **Constraint violation**: The code violates constraints like time complexity, space complexity, or other problem-specific limits.

    """


    # Prompt for analyzing the user's code based on the problem specification
    messages = [
        {"role": "system", "content": "You are an expert programmer who analyzes code for logical errors based on a given problem specification, utilizing insights from a related knowledge graph for more context."},
        {
            "role": "user",
            "content": f"""
            Before proceeding with the analysis of your code, I need to check with you first:
            
            - Do you recall working on a similar problem or having discussed a similar issue (concept, algorithm, or error) related to this problem before? 
            - For example, have you worked on similar problems involving [example problem], [specific error], or [related concept]? We may have discussed this in our past conversations (knowledge graph).

            If turns out there is the similar problems, concepts or error in the knowledge graph, explain it with the insights of the knowledge graph. If not, I will proceed with a fresh analysis using the knowledge graph provided.

            ---
            Problem Specification:
            {problem_spec}

            Code:
            ```
            {user_code}
            ```

            Knowledge Graph Insights:
            ```
            {knowledge_graph}
            ```

            IMPORTANT:
            - Focus on using the knowledge graph information to enhance your analysis. If specific knowledge graph insights align with known issues, mention them in a way that prompts the user to consider these aspects.
            - Treat Knowledge Graph as like records of past chats histories of the user.
            - Do not print or repeat the code.
            - Provide a brief, single-paragraph response with hints or guiding questions to encourage critical thinking around potential issues.
            - Focus primarily on logical errors, misunderstandings of problem requirements, and incorrect use of data structures rather than syntax issues.

            Example: "Have you considered how [condition] might affect the result? Could updating your approach to [aspect] lead to more reliable results?"
            """
        }
    ]



    
    # Make the API call to OpenAI with GPT-4 model
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    # Extract the generated message from the API response
    analysis = completion.choices[0].message.content
    return analysis

def analyze_code_error_kg(problem_spec, user_code):
    messages = [
        {"role": "system", "content": "You are an expert programmer who analyzes code for logical errors based on a given problem specification."},
        {
            "role": "user",
            "content": f"""
            Based on the problem specification and the user code below, first generate a **concise title for the problem, maximum 5 words** in just a few words. Then, identify and briefly describe each logical error in the code. 

            Format each error label as a very short noun phrase without verbs, capturing the essence of the issue (e.g., "wrong reverse integer" instead of "wrongly reverse integer").

            Format the response as follows:
            - "Brief Title of the Problem"; error1; error2; error3; ...

            --- 
            Problem Specification:
            {problem_spec}

            Code:
            ```python
            {user_code}
            ```
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
            st.session_state.messages.append(
        dict(
            role='problem',
            content=preprocessed_spec,
        )
    )
        
        with st.spinner("Analyzing the user code..."):
              try:
                  kg_result = joblib.load('data/graph_data')
              except:
                  kg_result = {}
              final_analysis = analyze_code_errors(preprocessed_spec, user_code, kg_result)
              st.session_state.messages.append(
                  dict(
                      role="result",
                      content=final_analysis,
                  )
                )
        # Display the code analysis
        st.subheader("Code Analysis")
        st.write(final_analysis)
        chat_id = f'{time.time()}'
        st.session_state.chat_id = chat_id
        st.session_state.chat_title = f'PROBLEMSPEC-{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}'      
        past_chats = joblib.load('data/past_chats_list')
        if st.session_state.chat_id not in past_chats.keys():
            past_chats[st.session_state.chat_id] = st.session_state.chat_title
            joblib.dump(past_chats, 'data/past_chats_list')
        joblib.dump(st.session_state.messages, f'data/{chat_id}-problemspec')
    else:
        st.error("Please enter both a problem specification and user code.")


