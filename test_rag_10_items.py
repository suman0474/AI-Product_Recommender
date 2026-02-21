import os
import sys
import logging
import json

backend_path = os.path.abspath(os.path.join(os.getcwd(), 'backend'))
sys.path.insert(0, backend_path)

from dotenv import load_dotenv
load_dotenv(os.path.join(backend_path, '.env'))

from taxonomy_rag.rag import get_taxonomy_rag

# Disable noisy infrastructure logs
logging.getLogger("common.infrastructure.caching").setLevel(logging.ERROR)
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

items = [
    "flow",
    "pressure",
    "temperature",
    "sensor",
    "valve",
    "mounting",
    "bracket",
    "transmitter",
    "differential",
    "Level"
]

def run_tests():
    rag = get_taxonomy_rag()
    
    print("======================================================")
    print(" TAXONOMY RAG - COSINE SIMILARITY TOP-3 TEST EXECUTION")
    print("======================================================\n")
    
    total_files_retrieved = 0
    total_specs_extracted = 0
    
    results = {}
    
    for item in items:
        print(f"=======================================")
        print(f" ANALYZING PRODUCT TYPE: {item.upper()}")
        print(f"=======================================")
        
        query = f"Technical specifications and manuals for {item}"
        print(f"[*] Generated Cosine Query: '{query}'")
        
        try:
            # 1. Get Top 3 Files — now returns a List[str]
            files = rag.get_top_files_by_similarity(query=query, top_k=3)
            file_count = len(files)

            if file_count > 0:
                total_files_retrieved += file_count
                print(f"  [+] Retrieved {file_count} files from Vector Database.")
                for i, f in enumerate(files):
                    print(f"\n  --- File {i+1} Raw Fragment ---")
                    print("  " + f[:250].strip().replace("\n", "\n  ") + "\n  ... (truncated)")
            else:
                print("  [-] Retrieved 0 files.")
                print(f"\n--- RAW FRAGMENT OF RETRIEVED FILES ---")
                results[item] = {"error": "No documents found"}
                continue

            # 2. Extract Specifications — iterates per file and merges
            print(f"\n  [*] Extracting specifications from {file_count} files (one by one)...")
            specs = rag.extract_specifications_from_files(item, files)
            
            spec_count = len(specs)
            total_specs_extracted += spec_count
            print(f"[+] SUCCESS: Extracted {spec_count} conceptual functional specifications.\n")
            
            print("--- EXTRACTED SPECIFICATIONS JSON ---")
            print(json.dumps(specs, indent=2))
            print("\n\n")
            
            results[item] = specs
            
        except Exception as e:
            print(f"[!] Error processing {item}: {e}")
            results[item] = {"error": str(e)}
            
    print("======================================================")
    print(" OVERALL TEST SUMMARY")
    print("======================================================")
    print(f"Total items tested: {len(items)}")
    print(f"Total files theoretically retrieved via Cosine Similarity: {total_files_retrieved}")
    print(f"Total engineering data points extracted: {total_specs_extracted}")
    print("======================================================")

if __name__ == "__main__":
    run_tests()
