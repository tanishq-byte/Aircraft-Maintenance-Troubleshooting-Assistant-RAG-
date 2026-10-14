
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore
from src.generation.answer_generator import AnswerGenerator


def main():
    print("Loading embedding model...")
    embedder = EmbeddingModel()

    print("Connecting to Qdrant...")
    store = VectorStore(
        storage_path="storage/qdrant_pilot",
        collection_name="aviation_pilot",
        vector_size=384,
    )

    generator = AnswerGenerator(model="qwen2.5:3b")

    question = (
        "What does the available evidence report about "
        "main aircraft battery charger failures?"
    )

    try:
        # 1. Embed the user's question
        print(f"\nQuestion: {question}")
        print("\nRetrieving relevant records...")

        query_vector = embedder.embed_query(question)

        # 2. Retrieve more candidates to allow relevance filtering
        candidate_results = store.search(
            query_vector,
            limit=10,
        )

        print(f"Candidate results: {len(candidate_results)}")

        # 3. Filter candidates to retain charger-related records
        results = []

        for result in candidate_results:
            metadata = result.get("metadata", {})
            text = result.get("text", "").lower()
            component = str(
                metadata.get("component_part_name", "")
            ).lower()

            if "charger" in text or "charger" in component:
                results.append(result)

        # Keep the highest-ranked matching records
        results = results[:5]

        print(f"Relevant candidates retained: {len(results)}")

        for result in results:
            print(
                f"- {result.get('source_record_id')} "
                f"(score: {result.get('score', 0.0):.4f})"
            )

        # 4. Generate an answer from the retained evidence
        print("\nGenerating evidence-grounded answer...")

        answer = generator.generate(
            question=question,
            results=results,
        )

        print("\n" + "=" * 70)
        print("RAG ANSWER")
        print("=" * 70)
        print(answer)

        # 5. Display the source records supplied to the generator
        print("\n" + "=" * 70)
        print("EVIDENCE SOURCES")
        print("=" * 70)

        if results:
            for result in results:
                metadata = result.get("metadata", {})

                print(
                    f"- {result.get('source_record_id')} | "
                    f"Component: "
                    f"{metadata.get('component_part_name', 'Unknown')} | "
                    f"Score: {result.get('score', 0.0):.4f}"
                )
        else:
            print("No charger-related records were retained.")

    finally:
        store.close()


if __name__ == "__main__":
    main()
