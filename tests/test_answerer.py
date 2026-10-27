from types import SimpleNamespace
from unittest.mock import Mock

from app.answerer import generate_answer
from app.chunker import chunk_documents


def make_chunks():
    documents = [
        {
            "source": "leave_policy.txt",
            "content": (
                "Employees receive 18 paid annual leave days. "
                "Requests must be submitted five working days "
                "in advance. Sick leave is separate."
            ),
        },
        {
            "source": "expense_policy.txt",
            "content": (
                "Approved business travel expenses are reimbursed. "
                "Claims require receipts and must be submitted "
                "within 30 days. Personal purchases are not eligible."
            ),
        },
    ]

    return chunk_documents(documents, chunk_size=200, overlap=40)


def make_client(answer: str) -> Mock:
    """Create a fake Groq client with a predefined answer."""

    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=answer)
            )
        ]
    )

    return client




def test_returns_not_found_without_calling_model():
    chunks = make_chunks()
    client = make_client("This should never be returned.")

    result = generate_answer(
        "What is the company's maternity leave policy?",
        chunks,
        client,
    )

    assert result["answer"] == (
        "I could not find relevant information in the documents."
    )
    assert result["sources"] == []
    client.chat.completions.create.assert_not_called()


def test_returns_answer_and_relevant_source():
    chunks = make_chunks()
    client = make_client(
        "Employees receive 18 days of paid annual leave."
    )

    result = generate_answer(
        "How many annual leave days do employees receive?",
        chunks,
        client,
    )

    assert result["answer"] == (
        "Employees receive 18 days of paid annual leave."
    )
    assert result["sources"] == ["leave_policy.txt"]
    client.chat.completions.create.assert_called_once()


def test_clears_sources_when_model_says_information_not_found():
    chunks = make_chunks()
    client = make_client(
        "The information was not found in the provided documents."
    )

    result = generate_answer(
        "How many annual leave days do employees receive?",
        chunks,
        client,
    )

    assert result["sources"] == []
