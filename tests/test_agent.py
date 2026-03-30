"""
test_agent.py – Automated test suite for UET MCP-RAG system
Run: python tests/test_agent.py
"""

import sys
import os
import json
import time
import requests

API_BASE = "http://localhost:8000"

# ── Test cases ────────────────────────────────────────────────────────────────
TEST_CASES = [
    # --- Department-related (should get valid answers) ---
    {"query": "Who is the chairperson of Computer Science?",          "expect_valid": True,  "category": "faculty"},
    {"query": "Who is the dean of Electrical Engineering?",           "expect_valid": True,  "category": "faculty"},
    {"query": "What programs does Software Engineering offer?",       "expect_valid": True,  "category": "programs"},
    {"query": "List the BS programs in Civil Engineering",            "expect_valid": True,  "category": "programs"},
    {"query": "What are the MS programs in Computer Science?",        "expect_valid": True,  "category": "programs"},
    {"query": "What are the admission requirements for CS?",          "expect_valid": True,  "category": "admission"},
    {"query": "Eligibility criteria for Electrical Engineering",      "expect_valid": True,  "category": "admission"},
    {"query": "What is the fee structure for undergraduate programs?","expect_valid": True,  "category": "fees"},
    {"query": "Are there any scholarships available at UET?",         "expect_valid": True,  "category": "fees"},
    {"query": "What lab facilities does Mechanical Engineering have?","expect_valid": True,  "category": "facilities"},
    {"query": "Tell me about the Computer Science department",        "expect_valid": True,  "category": "general"},
    {"query": "What departments are offered at UET Lahore?",          "expect_valid": True,  "category": "general"},
    {"query": "What is the UET university prospectus about?",         "expect_valid": True,  "category": "general"},

    # --- Tricky (borderline UET-related) ---
    {"query": "Does UET have a hostel?",                              "expect_valid": True,  "category": "tricky"},
    {"query": "Where is UET located?",                               "expect_valid": True,  "category": "tricky"},
    {"query": "What sports facilities are at UET?",                  "expect_valid": True,  "category": "tricky"},
    {"query": "How many students are at UET?",                       "expect_valid": True,  "category": "tricky"},

    # --- Out of scope (should be rejected) ---
    {"query": "What is the capital of France?",                      "expect_valid": False, "category": "out_of_scope"},
    {"query": "Tell me a joke",                                      "expect_valid": False, "category": "out_of_scope"},
    {"query": "What is the stock price of Apple?",                   "expect_valid": False, "category": "out_of_scope"},
    {"query": "Write a Python program to sort a list",               "expect_valid": False, "category": "out_of_scope"},
    {"query": "What is 2 + 2?",                                      "expect_valid": False, "category": "out_of_scope"},
]


def run_tests():
    print("=" * 70)
    print("  UET MCP-RAG System – Automated Test Suite")
    print("=" * 70)

    # Health check
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        if not r.json().get("agent_ready"):
            print("⚠️  API not ready. Start: python backend/main.py")
            sys.exit(1)
        print("✅  API health check passed\n")
    except Exception as e:
        print(f"❌  Cannot reach API at {API_BASE}: {e}")
        sys.exit(1)

    results = {"passed": 0, "failed": 0, "errors": 0}
    category_stats: dict[str, dict] = {}

    for i, tc in enumerate(TEST_CASES, 1):
        query    = tc["query"]
        expected = tc["expect_valid"]
        category = tc["category"]

        if category not in category_stats:
            category_stats[category] = {"passed": 0, "failed": 0}

        try:
            t0 = time.perf_counter()
            resp = requests.post(
                f"{API_BASE}/chat",
                json={"message": query},
                timeout=120,
            )
            elapsed = round(time.perf_counter() - t0, 2)
            data = resp.json()

            is_valid = data.get("is_valid", False)
            passed   = is_valid == expected

            status = "✅ PASS" if passed else "❌ FAIL"
            results["passed" if passed else "failed"] += 1
            category_stats[category]["passed" if passed else "failed"] += 1

            print(f"[{i:02d}] {status} | {elapsed:5.1f}s | [{category}]")
            print(f"       Q: {query[:65]}")
            if not passed:
                print(f"       Expected valid={expected}, got valid={is_valid}")
                print(f"       A: {data.get('answer', '')[:120]}")
            print()

        except Exception as e:
            results["errors"] += 1
            print(f"[{i:02d}] 💥 ERROR | [{category}] {query[:50]}")
            print(f"       {e}\n")

    # Summary
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    total = len(TEST_CASES)
    print(f"  Total:  {total}")
    print(f"  Passed: {results['passed']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Errors: {results['errors']}")
    print(f"  Score:  {round(results['passed']/total*100, 1)}%")
    print()
    print("  By Category:")
    for cat, stats in category_stats.items():
        tot = stats["passed"] + stats["failed"]
        print(f"    {cat:<15} {stats['passed']}/{tot}")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
