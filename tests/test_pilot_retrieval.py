
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore


def main():
    # 1. Load the embedding model
    embedder = EmbeddingModel()

    # 2. Connect to the existing persistent Qdrant collection
    store = VectorStore(
        storage_path="storage/qdrant_pilot",
        collection_name="aviation_pilot",
        vector_size=384,
    )

    queries = [
        "Main battery charger failure on an aircraft",
        "Aircraft EICAS reports main battery charger failure",
        "Aircraft flashlight battery not charging",
    ]

    try:
        for query in queries:
            # 3. Convert the query into an embedding
            query_vector = embedder.embed_query(query)

            # 4. Retrieve the five most similar indexed chunks
            results = store.search(query_vector, limit=5)

            print("\n" + "=" * 80)
            print(f"QUERY: {query}")
            print("=" * 80)
            print(f"Results returned: {len(results)}\n")

            # 5. Display ranked results
            for rank, result in enumerate(results, start=1):
                metadata = result.get("metadata", {})

                print(f"--- Result {rank} ---")
                print(f"Score: {result['score']:.4f}")
                print(f"Chunk ID: {result.get('chunk_id')}")
                print(
                    "Source ID: "
                    f"{result.get('source_record_id')}"
                )
                print(
                    "Aircraft: "
                    f"{metadata.get('aircraft_make', '')} "
                    f"{metadata.get('aircraft_model', '')}"
                )
                print(
                    "Component: "
                    f"{metadata.get('component_part_name', '')}"
                )
                print(
                    "Component condition: "
                    f"{metadata.get('component_part_condition', '')}"
                )

                text = result.get("text", "")
                preview = text[:500].replace("\n", " ")
                print(f"Text: {preview}")
                print()

    finally:
        # 6. Close the local Qdrant connection
        store.close()


if __name__ == "__main__":
    main()
