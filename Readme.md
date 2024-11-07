# Personalized AI Tutor For Programming

This project is a Streamlit-based application featuring several pages, including an AI Tutor chatbot and a problem-code matching tool. The AI Tutor provides answers to programming-related questions using a knowledge graph and RAG. The Problem-Code Matching tool allows users to input a coding problem and incorrect code for analysis.

## Table of Contents
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)

---

## Features

### AI Tutor 🧑🏽‍💻
This page includes a conversational AI chatbot designed to help users with Data Structures-related questions. It uses:
- **Past Chat Context**: Draws on previous interactions for context-aware answers.
- **Knowledge Graph**: A stored knowledge base that aids in continuity and understanding user progress.
- **Response Categories**: Problem Definition, Concept Explanation, Step-by-step Guidance, and Pseudo Code Visualization.

### Problem-Code Matching ✨
This page allows users to input a coding problem and solution. The AI analyzes the solution for logical errors based on the problem specification, with attention to:
- **Data Structure Appropriateness**
- **Implementation Correctness**
- **Edge Case Handling**
- **Problem Constraints**

### DS Quiz ❓
This page includes a Data Structures quiz with questions to test users' understanding of key concepts.

### Admin Panel 🛠️
The Admin Panel provides administrative controls, including chat history management and data storage.

---

## Setup

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install Dependencies**
   Ensure you have Python installed (3.7+ recommended). Install required packages using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

3. **API Key Setup**
   - Set up an `.env` file in the root directory with your API keys. The app uses OpenAI’s API and Neo4j for the knowledge graph.
   - Add the following variables:
     ```plaintext
     OPENAI_API_KEY=your_openai_api_key
     NEO4J_URI=your_neo4j_uri
     NEO4J_USERNAME=your_neo4j_username
     NEO4J_PASSWORD=your_neo4j_password
     ```
4. **Chatbot Context**
   - Put any PDF files to be your context

4. **Run the Application**
   Start the Streamlit app by running:
   ```bash
   streamlit run app.py
   ```

---

## Usage

- **AI Tutor**: Type questions into the chat input box. The AI will use past conversations and knowledge graphs to answer in a context-aware manner.
- **Problem-Code Matching**: Enter a problem description and corresponding code to receive feedback on the code’s logical feedback.
- **DS Quiz**: Navigate to the quiz page to answer questions on Data Structures and check your understanding.
- **Admin Panel**: Access this page to manage chat history and stored data files.

---

## Notes

- The AI Tutor depends on OpenAI’s API, so ensure you have sufficient API quota.
- Neo4j database connection is used for the knowledge graph. Ensure the connection details are correct and that Neo4j is running if using a local server.

---
