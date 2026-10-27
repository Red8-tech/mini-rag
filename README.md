# Mini RAG — Company Policy Question Answering

A lightweight Retrieval-Augmented Generation (RAG) application built with Python, TF-IDF, scikit-learn, and Groq. It answers questions using a small collection of company policy documents and displays the source files used for each answer.

## Features

- Load text documents from a local directory.
- Split documents into overlapping text chunks.
- Retrieve relevant chunks using TF-IDF and cosine similarity.
- Filter weak matches using a similarity threshold and topic-specific checks.
- Generate context-based answers using a Groq-hosted language model.
- Display source filenames for supported answers.
- Handle unsupported questions without calling the language model when retrieval finds no results.
- Validate user input.
- Test retrieval and answer generation using pytest and mocked API responses.

## Technology Stack

- Python 3.10+
- scikit-learn
- Groq API
- python-dotenv
- pytest

## Project Structure

```text
mini-rag/
├── app/
│   ├── answerer.py
│   ├── chunker.py
│   ├── document_loader.py
│   └── retriever.py
├── data/
│   └── documents/
│       ├── expense_policy.txt
│       ├── leave_policy.txt
│       └── remote_work.txt
├── tests/
│   ├── test_answerer.py
│   └── test_retriever.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## How It Works

1. **Document loading:** Read text files from `data/documents/`.
2. **Chunking:** Divide each document into smaller, overlapping text segments while preserving source filenames.
3. **Retrieval:** Convert chunks and the user's question into TF-IDF vectors and rank them using cosine similarity.
4. **Filtering:** Apply a minimum similarity threshold and additional topic checks to reject certain unsupported questions.
5. **Answer generation:** Send the question and retrieved context to the Groq API.
6. **Source attribution:** Display the source filenames associated with retrieved context, unless the answer is classified as not found.

## Prerequisites

- Python 3.10 or later
- pip
- A Groq API key for interactive answer generation

## Installation on Windows

Open PowerShell in the project directory.

### 1. Create a virtual environment

```powershell
python -m venv venv
```

### 2. Activate it

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the virtual environment's Python executable directly instead.

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file in the project root with:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Replace the placeholder with your own API key. Never commit your real API key to a public repository.

## Running the Application

From the project root, run:

```powershell
python -m app.answerer
```

Ask questions interactively. Type `exit` or `quit` to stop the application.

## Example Questions

- How many annual leave days do employees receive?
- How far in advance should I request leave?
- Can I work remotely two days per week?
- What is the deadline for expense claims?
- Does the company reimburse personal purchases?
- What is the company's maternity leave policy?

If the available documents do not provide relevant context, the application should report that the information could not be found.

## Running Tests

Run the complete test suite:

```powershell
python -m pytest -v
```

The current suite contains nine tests covering:

- Retrieval of relevant policy chunks
- Rejection of the tested unsupported maternity-leave question
- Empty-question validation
- Invalid `top_k` validation
- Invalid similarity-threshold validation
- Answer generation with a mocked model response
- Avoiding an API call when retrieval returns no results
- Clearing sources when the model explicitly reports missing information

The tests use mocked model responses, so they do not need real Groq API calls.

## Configuration

The retriever currently uses:

- TF-IDF vectorization
- Unigrams and bigrams
- English stop-word removal
- Cosine similarity
- A default minimum similarity score of `0.15`
- A default maximum of three retrieved chunks

These values are initial settings and may need adjustment as the document collection grows.

## Current Limitations

- TF-IDF is based on word overlap and may not understand semantic similarity or paraphrases.
- The current maternity and parental leave safeguards are topic-specific rules rather than a general semantic support detector.
- Similarity scores alone do not guarantee that a chunk supports the generated answer.
- Source attribution is based on retrieved chunks and the model's response wording; it does not independently verify every factual claim.
- Documents are loaded from local text files only.
- The application does not currently provide a web interface or persistent vector database.

## Future Improvements

- Replace topic-specific checks with a more general relevance and evidence-support strategy.
- Add tests for more unsupported questions and paraphrased queries.
- Improve chunk boundaries to avoid splitting sentences.
- Add structured logging and clearer API error handling.
- Explore embedding-based retrieval and a vector database.
- Add a simple web interface.
- Add CI checks for automated testing and static analysis.

## Learning Outcomes

This project demonstrates practical experience with:

- Building a basic RAG pipeline
- Python modular design and type hints
- Text processing and information retrieval
- Integrating an external LLM API
- Environment-variable configuration
- Mocking external dependencies
- Automated testing with pytest
- Handling unsupported questions and retrieval limitations
