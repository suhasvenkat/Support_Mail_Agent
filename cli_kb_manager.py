"""CLI tool to manage FAISS knowledge base."""

import argparse
import sys
from pathlib import Path

from src.services.faiss_store import FAISSStore
from data.sample_documents import get_sample_documents


def load_kb(index_path: str = "data/faiss_index"):
    """Load sample documents into FAISS index."""
    print("\n" + "=" * 70)
    print("📚 Loading Knowledge Base Documents")
    print("=" * 70)

    faiss_store = FAISSStore(index_path=index_path)
    sample_docs = get_sample_documents()

    documents = [doc["content"].strip() for doc in sample_docs]
    metadatas = [
        {
            "category": doc["category"],
            "source": doc["source"],
        }
        for doc in sample_docs
    ]
    ids = [doc["id"] for doc in sample_docs]

    print(f"\n📄 Loading {len(sample_docs)} documents...")
    success = faiss_store.add_documents(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

    if success:
        stats = faiss_store.get_stats()
        print(f"\n✅ Knowledge Base Loaded Successfully!")
        print(f"   Total documents: {stats['total_documents']}")
        print(f"   Index size: {stats['index_file_size']} bytes")
    else:
        print("\n❌ Failed to load knowledge base!")
        return False

    print("=" * 70 + "\n")
    return True


def search_kb(query: str, top_k: int = 5, index_path: str = "data/faiss_index"):
    """Search the knowledge base."""
    faiss_store = FAISSStore(index_path=index_path)

    print(f"\n🔍 Searching for: '{query}'")
    print(f"   (top {top_k} results)\n")

    results = faiss_store.search(query=query, top_k=top_k)

    if not results:
        print("❌ No results found.")
        return

    for i, result in enumerate(results, 1):
        print(f"{'─' * 70}")
        print(f"Result {i}: {result['id']}")
        print(f"Relevance Score: {result['similarity']:.1%}")
        print(f"Category: {result['metadata'].get('category', 'N/A')}")
        print(f"Source: {result['metadata'].get('source', 'N/A')}")
        print(f"\n{result['content'][:300]}...")
        print()


def list_documents(index_path: str = "data/faiss_index"):
    """List all documents in the knowledge base."""
    faiss_store = FAISSStore(index_path=index_path)

    if not faiss_store.document_store:
        print("❌ Knowledge base is empty. Run 'load' command first.")
        return

    print("\n" + "=" * 70)
    print("📚 Knowledge Base Documents")
    print("=" * 70 + "\n")

    for i, doc in enumerate(faiss_store.document_store, 1):
        print(f"{i}. [{doc['id']}]")
        print(f"   Category: {doc['metadata'].get('category', 'N/A')}")
        print(f"   Source: {doc['metadata'].get('source', 'N/A')}")
        print(f"   Preview: {doc['content'][:100]}...")
        print()


def get_stats(index_path: str = "data/faiss_index"):
    """Display knowledge base statistics."""
    faiss_store = FAISSStore(index_path=index_path)
    stats = faiss_store.get_stats()

    print("\n" + "=" * 70)
    print("📊 Knowledge Base Statistics")
    print("=" * 70)
    print(f"Total Documents: {stats['total_documents']}")
    print(f"Index Initialized: {'✅ Yes' if stats['index_initialized'] else '❌ No'}")
    print(f"Index File Size: {stats['index_file_size']:,} bytes")

    if stats['index_initialized']:
        print(f"\n💡 Ready to process queries!")
    else:
        print(f"\n⚠️  Run 'load' command to initialize index")

    print("=" * 70 + "\n")


def delete_index(index_path: str = "data/faiss_index", confirm: bool = True):
    """Delete the knowledge base index."""
    if confirm:
        response = input("⚠️  Are you sure you want to delete the index? (yes/no): ")
        if response.lower() != "yes":
            print("Cancelled.")
            return

    faiss_store = FAISSStore(index_path=index_path)
    if faiss_store.delete_index():
        print("✅ Index deleted successfully.")
    else:
        print("❌ Failed to delete index.")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manage FAISS Knowledge Base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_kb_manager.py load              # Load sample documents
  python cli_kb_manager.py search "billing"  # Search knowledge base
  python cli_kb_manager.py list              # List all documents
  python cli_kb_manager.py stats             # Show statistics
        """,
    )

    parser.add_argument(
        "command",
        choices=["load", "search", "list", "stats", "delete"],
        help="Command to execute",
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        help="Search query (for 'search' command)",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (for 'search' command)",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="data/faiss_index",
        help="Path to FAISS index",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompts",
    )

    args = parser.parse_args()

    # Handle commands
    if args.command == "load":
        load_kb(index_path=args.index_path)

    elif args.command == "search":
        if not args.query:
            print("❌ Error: --query is required for 'search' command")
            sys.exit(1)
        search_kb(query=args.query, top_k=args.top_k, index_path=args.index_path)

    elif args.command == "list":
        list_documents(index_path=args.index_path)

    elif args.command == "stats":
        get_stats(index_path=args.index_path)

    elif args.command == "delete":
        delete_index(index_path=args.index_path, confirm=not args.yes)


if __name__ == "__main__":
    main()
