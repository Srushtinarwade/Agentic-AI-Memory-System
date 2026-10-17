# Agentic AI Memory System

## Overview

An AI-powered conversational memory system that allows an agent to remember information across conversations, retrieve relevant memories, and forget information when requested.

## Features

- **Short-term memory:** Stores recent conversation history using Redis.
- **Long-term memory:** Stores persistent, user-specific information.
- **Memory management:** Create, retrieve, list, update, and delete memories.
- **Agentic workflow:** Uses LangGraph to coordinate retrieval, response generation, memory extraction, and forgetting.
- **LLM integration:** Uses Google Gemini.
- **REST API:** Built with FastAPI.
- **Automated tests:** Includes tests for short-term and long-term memory operations.

## Tech Stack

Python · FastAPI · LangGraph · Google Gemini · Redis · Streamlit

## Project Structure

```text
Agentic_AI_Assignment/
├── app/
│   ├── memory/
│   └── router/
├── tests/
│   ├── test_short_term.py
│   └── test_long_term.py
├── main.py
├── frontend.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd Agentic_AI_Assignment
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file containing the environment variables required by the application. Add your Gemini API key and never publish actual credentials.

### 5. Start Redis

Start your configured Redis service or Docker container.

### 6. Run the backend

```bash
python -m uvicorn main:app --reload
```

API documentation: http://127.0.0.1:8000/docs

### 7. Run the frontend

Open a second terminal, activate the virtual environment, and run:

```bash
streamlit run frontend.py
```

### 8. Run the tests

```bash
python -m pytest -v
```

The current test suite contains 10 passing tests covering short-term and long-term memory functionality.
