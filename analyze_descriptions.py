import os
import sys
import json
import logging

# Add backend to path
backend_path = os.path.abspath(os.path.join(os.getcwd(), 'backend'))
sys.path.insert(0, backend_path)

from dotenv import load_dotenv
load_dotenv(os.path.join(backend_path, '.env'))

from taxonomy_rag.rag import get_taxonomy_rag
from common.services.llm.fallback import create_llm_with_fallback

# Setup logging
logging.basicConfig(level=logging.INFO)

def load_products_from_txt(filepath):
    instruments = []
    accessories = []
    current_section = None
    
    if not os.path.exists(filepath):
        return [], []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "INSTRUMENTS":
                current_section = "instruments"
                continue
            elif line == "ACCESSORIES":
                current_section = "accessories"
                continue
            elif line.startswith("- "):
                name = line[2:].split(':')[0].strip()
                if current_section == "instruments":
                    instruments.append(name)
                elif current_section == "accessories":
                    accessories.append(name)
    
    return instruments, accessories

def generate_standardized_description(name, item_type, llm):
    prompt = f"""
Provide a highly standardized, technical definition for the industrial {item_type[:-1]} known as "{name}".
The definition should be concise (2-3 sentences), focus on its primary function in an industrial control system, 
and use formal engineering terminology.

Standardized Description:
"""
    try:
        response = llm.invoke(prompt)
        content = str(response.content).strip()
        # Clean up if needed
        content = content.replace('Standardized Description:', '').strip()
        return content
    except Exception as e:
        return f"Error generating description: {e}"

def analyze_and_standardize():
    input_file = "product_types_segregated.txt"
    instruments, accessories = load_products_from_txt(input_file)
    
    if not instruments and not accessories:
        print("No products found to analyze.")
        return

    rag = get_taxonomy_rag()
    llm = create_llm_with_fallback()
    
    results = {"instruments": [], "accessories": []}
    
    print(f"Analyzing {len(instruments) + len(accessories)} items...")
    
    # Process Instruments
    for name in instruments:
        print(f"Analyzing Instrument: {name}")
        # Try to retrieve from DB first
        db_desc = None
        try:
            rag_results = rag.retrieve(query=name, top_k=1, item_type='instrument')
            if rag_results and rag_results[0].get('score', 0) > 0.8:
                db_desc = rag_results[0].get('content', '')
                if "Definition: " in db_desc:
                    db_desc = db_desc.split("Definition: ")[1].strip()
        except:
            pass
            
        final_desc = db_desc if db_desc else generate_standardized_description(name, "instrument", llm)
        source = "Database (RAG)" if db_desc else "Generated (LLM)"
        
        results["instruments"].append({
            "name": name,
            "description": final_desc,
            "source": source
        })

    # Process Accessories
    for name in accessories:
        print(f"Analyzing Accessory: {name}")
        db_desc = None
        try:
            rag_results = rag.retrieve(query=name, top_k=1, item_type='accessory')
            if rag_results and rag_results[0].get('score', 0) > 0.8:
                db_desc = rag_results[0].get('content', '')
                if "Definition: " in db_desc:
                    db_desc = db_desc.split("Definition: ")[1].strip()
        except:
            pass
            
        final_desc = db_desc if db_desc else generate_standardized_description(name, "accessory", llm)
        source = "Database (RAG)" if db_desc else "Generated (LLM)"
        
        results["accessories"].append({
            "name": name,
            "description": final_desc,
            "source": source
        })

    # Save to file
    output_file = "standardized_descriptions_analysis.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("STANDARDIZED PRODUCT DESCRIPTIONS ANALYSIS\n")
        f.write("======================================================================\n")
        f.write(f"Total Items Analyzed: {len(instruments) + len(accessories)}\n")
        f.write("Status: Most descriptions were generated as the database taxonomy is currently empty.\n\n")
        
        f.write("INSTRUMENTS\n")
        f.write("-" * 20 + "\n")
        for item in results["instruments"]:
            f.write(f"Product: {item['name']}\n")
            f.write(f"Source: {item['source']}\n")
            f.write(f"Standardized Description: {item['description']}\n")
            f.write("-" * 40 + "\n")
            
        f.write("\nACCESSORIES\n")
        f.write("-" * 20 + "\n")
        for item in results["accessories"]:
            f.write(f"Product: {item['name']}\n")
            f.write(f"Source: {item['source']}\n")
            f.write(f"Standardized Description: {item['description']}\n")
            f.write("-" * 40 + "\n")
            
    print(f"Analysis complete. Results saved to {output_file}")

if __name__ == "__main__":
    analyze_and_standardize()
