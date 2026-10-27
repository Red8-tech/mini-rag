from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"


def load_documents(
    directory: Path = DOCUMENTS_DIR,
) -> list[dict[str, str]]:
    """Load text documents and preserve their source filenames."""

    documents: list[dict[str, str]] = []

    for file_path in sorted(directory.glob("*.txt")):
        content = file_path.read_text(encoding="utf-8-sig").strip()

        if not content:
            continue

        documents.append(
            {
                "source": file_path.name,
                "content": content,
            }
        )

    return documents


if __name__ == "__main__":
    loaded_documents = load_documents()

    print(f"Loaded {len(loaded_documents)} documents.")

    for document in loaded_documents:
        print(f"\nSource: {document['source']}")
        print(document["content"])
