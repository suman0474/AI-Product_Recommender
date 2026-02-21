import sys
import os
import json

backend_path = os.path.abspath(os.path.join('.', 'backend'))
sys.path.insert(0, backend_path)

from taxonomy_rag.rag import SpecificationRetriever, get_taxonomy_rag
from common.services.schema_service import SchemaService

def analyze_descriptions():
    # 1. Load the list of 71 items we processed earlier
    product_types_file = "product_types_segregated.txt"
    items_to_check = []
    
    if os.path.exists(product_types_file):
        with open(product_types_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("- "):
                    name = line[2:].split(":")[0].strip()
                    items_to_check.append(name)
    
    if not items_to_check:
        print("Could not find product_types_segregated.txt. Please ensure it exists.")
        return

    print(f"Loaded {len(items_to_check)} items to check.")
    
    # 2. Check using the standard SpecificationRetriever (how integration.py gets specs)
    retriever = SpecificationRetriever()
    print(f"Retriever mode: {retriever.mode}")
    
    # Check using TaxonomyRAG directly for definitions
    rag = get_taxonomy_rag()
    
    found_in_retriever = []
    found_in_rag = []
    missing_everywhere = []
    
    for item_name in items_to_check:
        found = False
        
        # Check retriever (JSON catalog / MongoDB)
        spec = retriever.get_specification(item_name)
        if spec and ('description' in spec or 'definition' in spec or 'standardized_description' in spec):
            found_in_retriever.append(item_name)
            found = True
            
        # Check Taxonomy RAG
        if not found:
            rag_results = rag.retrieve(query=item_name, top_k=1)
            if rag_results and rag_results[0].get('score', 0) > 0.8:
                content = rag_results[0].get('content', '')
                if 'Definition:' in content:
                    found_in_rag.append(item_name)
                    found = True
                    
        if not found:
            missing_everywhere.append(item_name)
            
    print("\n--- RESULTS ---")
    print(f"Total checked: {len(items_to_check)}")
    print(f"Found via SpecificationRetriever (MongoDB/Catalog): {len(found_in_retriever)}")
    print(f"Found via TaxonomyRAG Vector Store: {len(found_in_rag)}")
    print(f"Missing standardized descriptions in system: {len(missing_everywhere)}")
    
    if len(missing_everywhere) > 0:
        print("\nFirst 20 missing items:")
        for item in missing_everywhere[:20]:
            print(f"- {item}")
            
    # Save the analysis
    with open("description_system_check.txt", "w", encoding="utf-8") as f:
        f.write("System-wide Standardized Description Check\n")
        f.write("=========================================\n\n")
        f.write(f"Total Items Checked: {len(items_to_check)}\n")
        f.write(f"Found in Database/Catalog: {len(found_in_retriever)}\n")
        f.write(f"Found in Vector Store RAG: {len(found_in_rag)}\n")
        f.write(f"Missing Built-in Descriptions: {len(missing_everywhere)}\n\n")
        if missing_everywhere:
            f.write("Missing Items:\n")
            for item in missing_everywhere:
                f.write(f"- {item}\n")

if __name__ == "__main__":
    analyze_descriptions()
