# PaperCopilot

Help students understand research papers quickly.

## Overview

PaperCopilot is a single-user local application that leverages AI to make research papers more accessible and easier to understand. It uses semantic search and Claude's capabilities to provide contextual insights.

## Tech Stack

- **Python** - Core programming language
- **Streamlit** - Web application framework
- **FAISS** - Fast similarity search and clustering of dense vectors
- **Sentence Transformers** - Creating embeddings for semantic search
- **Claude API** - Advanced language model for paper analysis

## Project Structure

```
paper_copilot/
├── app.py                     # Main Streamlit application
├── requirements.txt           # Project dependencies
├── README.md                  # This file
├── .gitignore                 # Git ignore rules
│
├── src/
│   ├── parser.py             # PDF extraction and metadata extraction
│   ├── embeddings.py         # Embedding generation
│   ├── vectorstore.py        # FAISS operations
│   ├── claude_client.py      # Claude API interactions
│   ├── prompts.py            # All system prompts
│   ├── summarizer.py         # Executive and student summaries
│   ├── mentor_mode.py        # Explanations and methodology analysis
│   ├── roadmap_generator.py  # Prerequisite concepts and reading paths
│   └── paper_analyzer.py     # Contributions, limitations, datasets, equations
│
├── papers/                    # Uploaded research papers directory
├── indexes/                   # FAISS vector store indexes
├── outputs/                   # Generated analysis and summaries
└── data/                      # Additional data storage
```

## Core Features

**Paper Understanding:**
- PDF parsing and metadata extraction
- Semantic embedding generation with Sentence Transformers

**Student Assistance:**
- Executive summaries (comprehensive overview)
- Student-friendly summaries (simplified explanations)
- Mentor mode (explain like a CSE student, identify assumptions, explain methodology)
- Reading roadmap generator (prerequisites and concept progression)
- Paper analyzer (contributions, limitations, future work, datasets, metrics, equations)

**Search & Retrieval:**
- FAISS vector store for fast semantic search
- Context-aware paper analysis via Claude API

## Requirements

- Python 3.8+
- Anthropic API key for Claude access

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

## Notes

- Single-user local application
- No enterprise architecture
- No Docker containerization
