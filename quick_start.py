"""
DEBATEAI - Quick Start Verification
====================================
Run this script to verify everything is set up correctly.

Usage: python quick_start.py
"""

import sys
from pathlib import Path

def check_file_structure():
    """Check if all required files exist"""
    print("\n" + "=" * 60)
    print("Checking File Structure")
    print("=" * 60)
    
    required_files = {
        'core/interfaces.py': 'Base interfaces',
        'core/orchestrator.py': 'Debate orchestrator',
        'core/hybrid_retriever.py': 'Hybrid search',
        'core/document_processor.py': 'Document processing',
        'retrievers/faiss_retriever.py': 'FAISS search',
        'retrievers/bm25_retriever.py': 'BM25 search',
        'agents/pro.py': 'Pro agent',
        'agents/con.py': 'Con agent',
        'agents/judge.py': 'Judge agent',
        'agents/reporter.py': 'Reporter agent',
        'config/agents_config.yaml': 'Agent config',
        'cli.py': 'Command-line interface',
    }
    
    missing = []
    found = []
    
    for file_path, description in required_files.items():
        if Path(file_path).exists():
            print(f"✅ {file_path}")
            found.append(file_path)
        else:
            print(f"❌ {file_path} - MISSING!")
            missing.append(file_path)
    
    print(f"\n📊 Status: {len(found)}/{len(required_files)} files found")
    
    if missing:
        print("\n⚠️  Missing files:")
        for f in missing:
            print(f"   - {f}")
        return False
    else:
        print("\n✅ All required files present!")
        return True


def check_dependencies():
    """Check if required Python packages are installed"""
    print("\n" + "=" * 60)
    print("Checking Dependencies")
    print("=" * 60)
    
    required_packages = [
        ('llama_index', 'llama-index'),
        ('faiss', 'faiss-cpu or faiss-gpu'),
        ('rank_bm25', 'rank-bm25'),
        ('sentence_transformers', 'sentence-transformers'),
        ('ollama', 'ollama'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('pypdf', 'pypdf'),
        ('yaml', 'pyyaml'),
    ]
    
    missing = []
    
    for package, install_name in required_packages:
        try:
            __import__(package)
            print(f"✅ {install_name}")
        except ImportError:
            print(f"❌ {install_name} - NOT INSTALLED!")
            missing.append(install_name)
    
    if missing:
        print("\n⚠️  Missing packages. Install with:")
        print(f"   pip install {' '.join(missing)}")
        return False
    else:
        print("\n✅ All dependencies installed!")
        return True


def check_ollama():
    """Check Gemini API key instead of Ollama."""
    print("\n" + "=" * 60)
    print("Checking Gemini API")
    print("=" * 60)

    import os
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        print("✅ GEMINI_API_KEY found")
        return True
    else:
        print("❌ GEMINI_API_KEY not set!")
        print("   Set it with: export GEMINI_API_KEY=your_key_here")
        return False

def check_gpu():
    """GPU not needed for cloud deployment."""
    print("\n" + "=" * 60)
    print("Deployment Mode")
    print("=" * 60)
    print("✅ Running in CPU/Cloud mode")
    print("   GPU not required — using Gemini API for LLMs")
    return True




def check_data():
    """Check if data folders exist"""
    print("\n" + "=" * 60)
    print("Checking Data Folders")
    print("=" * 60)
    
    data_folders = ['data/raw', 'data/processed', 'outputs']
    
    for folder in data_folders:
        path = Path(folder)
        if path.exists():
            files = list(path.glob('*'))
            print(f"✅ {folder}/ ({len(files)} files)")
        else:
            print(f"⚠️  {folder}/ - creating...")
            path.mkdir(parents=True, exist_ok=True)
    
    # Check if raw data exists
    raw_files = list(Path('data/raw').rglob('*.*'))
    if raw_files:
        print(f"\n📄 Found {len(raw_files)} files in data/raw/")
        print("   Ready to process!")
    else:
        print(f"\n⚠️  No files in data/raw/")
        print("   Add PDFs, CSVs, or TXT files to get started")
    
    return True


def main():
    """Run all checks"""
    print("\n" + "=" * 60)
    print("🤖 DEBATEAI - QUICK START VERIFICATION")
    print("=" * 60)
    
    checks = [
        ("File Structure", check_file_structure),
        ("Dependencies", check_dependencies),
        ("Ollama", check_ollama),
        ("GPU", check_gpu),
        ("Data", check_data),
    ]
    
    results = []
    for name, check_func in checks:
        result = check_func()
        results.append((name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n" + "=" * 60)
        print("🎉 ALL CHECKS PASSED!")
        print("=" * 60)
        print("\nYou're ready to run DEBATEAI!")
        print("\nNext steps:")
        print("  1. Add documents to data/raw/")
        print("  2. Run: python cli.py")
        print("  3. Choose option 1 to index documents")
        print("  4. Choose option 2 to run your first debate!")
    else:
        print("\n" + "=" * 60)
        print("⚠️  SOME CHECKS FAILED")
        print("=" * 60)
        print("\nFix the issues above and run this script again.")
        print("See INTEGRATION_GUIDE.md for detailed instructions.")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
