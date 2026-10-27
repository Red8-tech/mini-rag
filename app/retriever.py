
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.chunker import Chunk, chunk_documents
from app.document_loader import load_documents


GENERIC_TERMS = {
    "company",
    "policy",
    "policies",
    "information",
    "please",
    "tell",
    "what",
    "does",
    "do",
    "is",
    "are",
    "the",
    "a",
    "an",
    "of",
    "for",
    "about",
    "how",
    "many",
    "can",
    "i",
    "me",
    "my",
    "their",
    "employees",
}


def meaningful_terms(text: str) -> set[str]:
    """Extract terms while ignoring generic question wording."""

    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

    return {
        word
        for word in words
        if len(word) > 1 and word not in GENERIC_TERMS
    }


def retrieve(
    question: str,
    chunks: list[Chunk],
    top_k: int = 3,
    min_score: float = 0.15,
) -> list[dict[str, str | float | int]]:
    """Retrieve chunks using TF-IDF and meaningful-term overlap."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if not 0 <= min_score <= 1:
        raise ValueError("min_score must be between 0 and 1.")

    if not chunks:
        return []

    question_terms = meaningful_terms(question)

    if not question_terms:
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )

    chunk_texts = [chunk["text"] for chunk in chunks]

    try:
        chunk_vectors = vectorizer.fit_transform(chunk_texts)
        question_vector = vectorizer.transform([question])
    except ValueError:
        return []

    scores = cosine_similarity(
        question_vector,
        chunk_vectors,
    ).flatten()

    ranked_indices = scores.argsort()[::-1]
    results: list[dict[str, str | float | int]] = []

    for index in ranked_indices:
        score = float(scores[index])
        chunk = chunks[int(index)]
        chunk_terms = meaningful_terms(chunk["text"])

        # Require at least one meaningful term to overlap.
        overlap = question_terms.intersection(chunk_terms)

        if not overlap:
            continue

        # Policy-topic questions must match the requested topic,
        # not merely generic words such as "company" or "policy".
        question_lower = question.lower()

        if "maternity" in question_lower and "maternity" not in chunk_terms:
            continue

        if "parental" in question_lower and "parental" not in chunk_terms:
            continue

        if score < min_score:
            continue

        results.append(
            {
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": score,
            }
        )

        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    documents = load_documents()
    chunks = chunk_documents(documents)

    questions = [
        "How many annual leave days do employees receive?",
        "What is the deadline for expense claims?",
        "What is the company's maternity leave policy?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")

        results = retrieve(question, chunks)

        if not results:
            print("No relevant chunks found.")
            continue

        for result in results:
            print(
                f"\nSource: {result['source']} | "
                f"Chunk: {result['chunk_id']} | "
                f"Score: {result['score']:.3f}"
            )
            print(result["text"])
