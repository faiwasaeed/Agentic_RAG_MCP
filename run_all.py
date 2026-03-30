"""
run_all.py – Convenience launcher: starts FastAPI backend + Streamlit frontend
"""

import subprocess
import sys
import os
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    print("=" * 60)
    print("  UET MCP-RAG System Launcher")
    print("=" * 60)

    # Ensure ingestion has been done
    vector_db = os.path.join(ROOT, "data", "vector_db")
    if not os.path.exists(vector_db):
        print("\n⚠️  Vector DB not found. Running data ingestion first …\n")
        result = subprocess.run(
            [sys.executable, os.path.join(ROOT, "backend", "ingest.py")],
            cwd=ROOT,
        )
        if result.returncode != 0:
            print("❌  Ingestion failed. Make sure UET_Prospectus.pdf is in data/")
            sys.exit(1)

    print("\n🚀 Starting FastAPI backend on http://localhost:8000 …")
    backend = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "backend", "main.py")],
        cwd=os.path.join(ROOT, "backend"),
    )
    time.sleep(3)

    print("🎨 Starting Streamlit frontend on http://localhost:8501 …\n")
    frontend = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run",
         os.path.join(ROOT, "frontend", "app.py"),
         "--server.port", "8501"],
        cwd=ROOT,
    )

    print("=" * 60)
    print("  ✅ System running!")
    print("  Chat UI  → http://localhost:8501")
    print("  API Docs → http://localhost:8000/docs")
    print("  Press Ctrl+C to stop all services")
    print("=" * 60)

    try:
        backend.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down …")
        backend.terminate()
        frontend.terminate()


if __name__ == "__main__":
    main()
