import pytest

from app.chunker import chunk_documents
from app.retriever import retrieve


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
            "source": "remote_work.txt",
            "content": (
                "Employees may work remotely up to two days "
                "per week with manager approval."
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




def test_retrieves_relevant_leave_policy():
    chunks = make_chunks()

    results = retrieve(
        "How many annual leave days do employees receive?",
        chunks,
    )

    assert results
    assert results[0]["source"] == "leave_policy.txt"


def test_retrieves_relevant_expense_policy():
    chunks = make_chunks()

    results = retrieve(
        "Does the company reimburse personal purchases?",
        chunks,
    )

    assert results
    assert results[0]["source"] == "expense_policy.txt"


def test_unrelated_question_returns_no_results():
    chunks = make_chunks()

    results = retrieve(
        "What is the company's maternity leave policy?",
        chunks,
    )

    assert results == []




def test_empty_question_raises_value_error():
    chunks = make_chunks()

    with pytest.raises(ValueError, match="Question cannot be empty"):
        retrieve("", chunks)


def test_invalid_top_k_raises_value_error():
    chunks = make_chunks()

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        retrieve("annual leave", chunks, top_k=0)


def test_invalid_min_score_raises_value_error():
    chunks = make_chunks()

    with pytest.raises(ValueError, match="min_score must be between 0 and 1"):
        retrieve("annual leave", chunks, min_score=1.5)
