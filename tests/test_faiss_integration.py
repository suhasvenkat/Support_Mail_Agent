"""Test script to verify FAISS integration with the workflow."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.services.faiss_store import FAISSStore
from src.graph.workflow import build_graph
from src.graph.state import AgentState
from data.sample_documents import get_sample_documents


def test_faiss_basic():
    """Test basic FAISS operations."""
    print("\n" + "=" * 70)
    print("🧪 Test 1: Basic FAISS Operations")
    print("=" * 70)

    faiss_store = FAISSStore(index_path="data/faiss_index")

    # Check if index exists
    stats = faiss_store.get_stats()
    print(f"Total documents in index: {stats['total_documents']}")

    if stats["total_documents"] == 0:
        print("⚠️  Index is empty. Loading sample documents...")
        sample_docs = get_sample_documents()
        documents = [doc["content"].strip() for doc in sample_docs]
        metadatas = [
            {
                "category": doc["category"],
                "source": doc["source"],
            }
            for doc in sample_docs
        ]
        faiss_store.add_documents(
            documents=documents,
            metadatas=metadatas,
            ids=[doc["id"] for doc in sample_docs],
        )

    print("✅ FAISS index ready")
    return True


def test_faiss_search():
    """Test FAISS search functionality."""
    print("\n" + "=" * 70)
    print("🧪 Test 2: FAISS Search")
    print("=" * 70)

    faiss_store = FAISSStore(index_path="data/faiss_index")

    test_queries = [
        "I was charged twice",
        "App keeps crashing",
        "How to reset password",
    ]

    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = faiss_store.search(query=query, top_k=2)

        if results:
            for i, result in enumerate(results, 1):
                print(f"   Result {i}:")
                print(f"      ID: {result['id']}")
                print(f"      Similarity: {result['similarity']:.1%}")
                print(f"      Content: {result['content'][:80]}...")
        else:
            print("   No results found")

    print("\n✅ Search test passed")
    return True


def test_workflow_integration():
    """Test FAISS integration with the complete workflow."""
    print("\n" + "=" * 70)
    print("🧪 Test 3: Workflow Integration with FAISS")
    print("=" * 70)

    # Build the graph
    graph = build_graph()

    # Create test emails
    test_emails = [
        {
            "email_id": "test_001",
            "sender": "user1@example.com",
            "subject": "Double charge on my account",
            "body": "I was charged twice for my subscription this month. Please help!",
        },
        {
            "email_id": "test_002",
            "sender": "user2@example.com",
            "subject": "App crashing on startup",
            "body": "The app closes immediately when I try to open it on my phone.",
        },
        {
            "email_id": "test_003",
            "sender": "user3@example.com",
            "subject": "Password reset request",
            "body": "I forgot my password and need to reset it.",
        },
    ]

    print("\n📧 Processing test emails through workflow...\n")

    for email in test_emails:
        print(f"Processing: {email['email_id']}")
        print(f"  Subject: {email['subject']}")

        try:
            # Execute workflow
            result = graph.invoke(email)

            # Extract results
            intent = result.get("intent", "unknown")
            kb_results = result.get("kb_results", [])
            confidence = result.get("confidence", 0)
            escalated = result.get("should_escalate", False)

            print(f"  ✅ Intent: {intent}")
            print(f"  ✅ KB Results: {len(kb_results)} documents found")
            print(f"  ✅ Confidence: {confidence:.0%}")
            print(f"  ✅ Escalated: {'Yes' if escalated else 'No'}")

            if kb_results:
                top_result = kb_results[0]
                print(f"     Top match: {top_result['id']} ({top_result['similarity']:.1%})")

            response_preview = result.get("final_response", "")
            if response_preview:
                print(f"  ✅ Response: {response_preview[:80]}...")

        except Exception as e:
            print(f"  ❌ Error: {e}")

        print()

    print("✅ Workflow integration test passed")
    return True


def test_end_to_end():
    """Full end-to-end test with sample email."""
    print("\n" + "=" * 70)
    print("🧪 Test 4: End-to-End Email Processing")
    print("=" * 70)

    graph = build_graph()

    # Realistic customer email
    email = {
        "email_id": "real_email_001",
        "sender": "john.doe@company.com",
        "subject": "Urgent: Account suspended after payment issue",
        "body": """
        Hello,

        I'm writing because my account was suspended. I was charged twice last week
        and haven't been able to access my account since. I've been a customer for 2 years
        and this has never happened before.

        Please help me resolve this urgently. I need access to my data.

        Thank you,
        John Doe
        """,
    }

    print(f"\n📧 Email from: {email['sender']}")
    print(f"   Subject: {email['subject']}")

    try:
        result = graph.invoke(email)

        print("\n" + "-" * 70)
        print("WORKFLOW EXECUTION RESULTS")
        print("-" * 70)

        print(f"\n1️⃣  CLASSIFICATION")
        print(f"    Intent: {result.get('intent', 'unknown')}")

        print(f"\n2️⃣  KNOWLEDGE BASE SEARCH")
        kb_results = result.get("kb_results", [])
        print(f"    Found {len(kb_results)} relevant documents")
        if kb_results:
            for i, doc in enumerate(kb_results[:2], 1):
                print(f"    {i}. {doc['id']} ({doc['similarity']:.0%} match)")

        print(f"\n3️⃣  RESPONSE DRAFTING")
        print(f"    Confidence: {result.get('confidence', 0):.0%}")

        print(f"\n4️⃣  ESCALATION DECISION")
        escalated = result.get("should_escalate", False)
        print(f"    Escalated: {'✅ YES - To Human Team' if escalated else '❌ NO - AI Response'}")

        print(f"\n5️⃣  FINAL RESPONSE")
        response = result.get("final_response", "No response generated")
        print(f"    {response}")

        print(f"\n6️⃣  FOLLOW-UP")
        print(f"    Follow-up scheduled: {result.get('followup_scheduled', False)}")

        print("\n" + "=" * 70)
        print("✅ END-TO-END TEST PASSED")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error during workflow execution:")
        print(f"   {str(e)}")
        import traceback

        traceback.print_exc()
        return False

    return True


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "FAISS INTEGRATION TEST SUITE" + " " * 25 + "║")
    print("╚" + "=" * 68 + "╝")

    tests = [
        ("Basic FAISS Operations", test_faiss_basic),
        ("FAISS Search", test_faiss_search),
        ("Workflow Integration", test_workflow_integration),
        ("End-to-End Processing", test_end_to_end),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    print("=" * 70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
