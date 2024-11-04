import os
import pprint
import joblib
import json

from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# If error here, need to start database first : https://console.neo4j.io/?product=aura-db&tenant=2f27695c-4050-43d8-99fa-40642e452dea#databases
graph = Neo4jGraph()

client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

# Load the past chats from the joblib file
try:
    past_chats = joblib.load('data/past_chats_list')
except:
    past_chats = {}

chat_histories = json.loads(json.dumps(past_chats, indent=4))

# load all past_chat and create knowledge graph. Currently set to manual update, if you want, can use cron. 
for key, value in chat_histories.items():
    # print(key)
    try :
        past_chat = joblib.load(f"data/{key}-st_messages")
        chat_history = json.dumps(past_chat, indent=4)
    except:
        past_chat = {}
    
    try :
        past_chat = joblib.load(f"data/{key}-problemspec")
        chat_history = json.dumps(past_chat, indent=4)
    except:
        past_chat = {}
        break

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": """
                Summarize this chat history between user and a chatbot. I want you to only summarize what the user wants and change it into a knowledge tree with these allowed nodes and allowed relationships :
                Allowed Nodes:
                Person: Represents users, developers, or instructors interacting with the assistant.
                Problem: Represents coding issues or challenges that a person is trying to solve.
                Algorithm:Represents specific algorithms that can be used to solve coding problems.
                Concept: Represents key programming concepts, such as object-oriented programming, recursion, or concurrency.
                Programming Language: Represents languages such as Python, C++, Java, etc.
                Framework: Represents frameworks like Django, Flask, or React.
                Library: Represents libraries used in coding, such as NumPy, Pandas, or TensorFlow.
                Snippet: Represents a piece of code related to solving a problem or using a concept.
                Tool: Represents tools used in development, such as VS Code, Git, or Docker.
                Project: Represents a coding project that the user is working on.
                Function: Represents specific functions or methods used in coding.
                Error: Represents errors or bugs encountered while writing code.
                Test: Represents test cases or unit tests written to validate code.
                Solution: Represents a proposed or correct solution to a given problem.
                Slide: Represents a slide from a teaching material or presentation.

                Allowed Relationships:
                Person-WorksOn→Project: A person is working on a specific coding project.
                Person-TryingToSolve→Problem: A person is trying to solve a particular coding problem.
                Person-Prefers→Programming Language: A person has a preference for a particular programming language.
                Person-Asks→Question: A person asks a specific question related to coding.
                Problem-CanBeSolvedBy→Algorithm: A problem can be solved using a particular algorithm.
                Problem-Uses→Concept: A problem involves using a specific concept (e.g., recursion).
                Algorithm-IsImplementedIn→Programming Language: An algorithm is implemented in a particular programming language.
                Concept-IsExplainedOn→Slide: A concept is explained on a particular slide.
                Programming Language-Has→Framework: A programming language has specific frameworks associated with it.
                Framework-UsedIn→Project: A framework is used in a specific project.
                Project-DependsOn→Library: A project depends on a specific library.
                Library-Includes→Function: A library includes particular functions or methods.
                Function-IsPartOf→Snippet:
                A function is part of a code snippet.
                Snippet-Solves→Problem:
                A code snippet solves a particular problem.
                Problem-CausedBy→Error:
                A problem is caused by a specific error or bug.
                Error-ResolvedBy→Solution:
                An error is resolved by a particular solution.
                Solution-Resolves→Problem:
                A solution resolves a specific problem.
                Project-Has→Test:
                A project has related test cases.
                Test-Validates→Snippet:
                A test validates the correctness of a code snippet.

                Define the nodes and the relationships without any additional information

                """+chat_history,
            }
        ],
        model="gpt-4-turbo",
    )

    summary = chat_completion.choices[0].message.content
    print("Summary:", summary)

    llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo")

    llm_transformer_filtered = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=["Person", "Algorithm", "Problem", "Concept", "Programming Language", "Framework", "Library", "Tool", "Project", "Function", "Error", "Testcase", "Solution", "Slide", "Snippet"],
        allowed_relationships=["WorksOn", "TryingToSolve", "Prefers", "Asks", "CanBeSolvedBy", "Uses", "IsImplementedIn", "IsExplainedOn", "Has", "UsedIn", "DependsOn", "Includes", "IsPartOf", "Solves", "ResolvedBy", "Resolves"],
    )
    documents = [Document(page_content=summary)]
    graph_documents_filtered = llm_transformer_filtered.convert_to_graph_documents(
        documents
    )
    print(f"Nodes:")
    pprint.pp(graph_documents_filtered[0].nodes)
    print(f"Relationships:")
    pprint.pp(graph_documents_filtered[0].relationships)

    graph.add_graph_documents(graph_documents_filtered)
    print(graph.schema)

    # storing graph data
    file_path = 'data/graph_data'
    joblib.dump({'nodes': graph_documents_filtered[0].nodes, 'relationships': graph_documents_filtered[0].relationships}, file_path)