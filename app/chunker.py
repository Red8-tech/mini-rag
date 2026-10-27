from typing import TypedDict


class Document(TypedDict):
    source: str
    content: str


class Chunk(TypedDict):
    source: str
    chunk_id: int
    text: str


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[Chunk]:
    """Split documents into overlapping text chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks: list[Chunk] = []

    for document in documents:
        text = document["content"].strip()

        if not text:
            continue

        start = 0
        chunk_id = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "source": document["source"],
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                    }
                )
                chunk_id += 1

            if end == len(text):
                break

            start = end - overlap

    return chunks



if __name__ == "__main__":
    from app.document_loader import load_documents

    documents = load_documents()
    chunks = chunk_documents(documents)

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")

    for chunk in chunks:
        print(
            f"\nSource: {chunk['source']} | "
            f"Chunk: {chunk['chunk_id']}"
        )
        print(chunk["text"])
