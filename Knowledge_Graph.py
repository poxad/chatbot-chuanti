import os
import pprint
import joblib
import json

from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from neo4j_connection import Neo4jConnection
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
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

# Load all past_chat and create knowledge graph. Currently set to manual update, if you want, can use cron. 
for key, value in chat_histories.items():
    print(key)
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

    # Generate chat completion summary using OpenAI API
    chat_completion = client.chat.completions.create(
        messages=[{
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
        }],
        model="gpt-4-turbo",
    )

    summary = chat_completion.choices[0].message.content

    llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo")

    # Define the restricted ids set
    restricted_ids = {"Person", "Algorithm", "Problem", "Concept", "Programming Language", 
                      "Framework", "Library", "Tool", "Project", "Function", 
                      "Error", "Testcase", "Solution", "Slide", "Snippet"}

    # Initialize LLMGraphTransformer
    llm_transformer_filtered = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=["Person", "Algorithm", "Problem", "Concept", "Programming Language", "Framework", "Library", "Tool", "Project", "Function", "Error", "Testcase", "Solution", "Slide", "Snippet"],
        allowed_relationships=["WorksOn", "TryingToSolve", "Prefers", "Asks", "CanBeSolvedBy", "Uses", "IsImplementedIn", "IsExplainedOn", "Has", "UsedIn", "DependsOn", "Includes", "IsPartOf", "Solves", "ResolvedBy", "Resolves"],
    )

    # Convert documents to graph documents
    documents = [Document(page_content=summary)]
    graph_documents_filtered = llm_transformer_filtered.convert_to_graph_documents(documents)

    # Filter out nodes with restricted ids before adding to the graph
    for graph_doc in graph_documents_filtered:
        graph_doc.nodes = [node for node in graph_doc.nodes if node.id not in restricted_ids]

    # Add the filtered graph documents to the graph
    graph.add_graph_documents(graph_documents_filtered)

# Define the directory and file path to store graph data
file_dir = 'data'
file_path = os.path.join(file_dir, 'graph_data')

# Ensure the directory exists (even if it's the first time running the script)
if not os.path.exists(file_dir):
    os.makedirs(file_dir)

# Function to retrieve and store graph data
def store_graph(conn):
    def get_nodes():
        # Execute the Cypher query to get all nodes
        all_nodes = conn.query("MATCH (n) RETURN n")

        extracted_nodes = []
        # Process each node record
        for nodes in all_nodes:
            node = nodes['n']  # Extract the node from the record
            labels = list(node.labels)  # Convert frozen set of labels to list
            node_id = node['id']  # Access the 'id' property
            extracted_nodes.append({'labels': labels, 'id': node_id})

        return extracted_nodes

    def get_relationships():
        query = """
        MATCH (start)-[r]->(end)
        RETURN start.id AS start_node, type(r) AS relationship, end.id AS end_node
        """
        result = conn.query(query)
        relationships = []
        # Process query result
        for record in result:
            start_node = record["start_node"]
            relationship = record["relationship"]
            end_node = record["end_node"]
            relationships.append(f"{start_node} -> {relationship} -> {end_node}")

        return relationships

    # Get nodes and relationships
    nodes = get_nodes()
    relationships = get_relationships()

    return nodes, relationships

# Example of usage
conn = Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
nodes, relationships = store_graph(conn)

# Print nodes and relationships for verification
pprint.pprint(nodes)
pprint.pprint(relationships)

# Store the graph data in a joblib file
joblib.dump({'nodes': nodes, 'relationships': relationships}, file_path)
