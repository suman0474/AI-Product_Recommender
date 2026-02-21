import os
import sys
import json
import logging
from tqdm import tqdm

# Add backend to path
backend_path = os.path.abspath(os.path.join(os.getcwd(), 'backend'))
sys.path.insert(0, backend_path)

from dotenv import load_dotenv
load_dotenv(os.path.join(backend_path, '.env'))

from taxonomy_rag.rag import get_taxonomy_rag
from common.services.llm.fallback import create_llm_with_fallback
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def generate_standardized_description(name, item_type, chain):
    try:
        response = chain.invoke({"name": name, "item_type": item_type[:-1]})
        content = str(response).strip().strip('*').strip()
        content = content.replace('Standardized Description:', '').replace('Definition:', '').strip()
        return content
    except Exception as e:
        logger.error(f"Error generating description for {name}: {e}")
        return ""

def generate_and_ingest():
    input_file = "product_types_segregated.txt"
    instruments, accessories = load_products_from_txt(input_file)
    
    if not instruments and not accessories:
        print("No products found to analyze.")
        return

    rag = get_taxonomy_rag()
    
    llm = create_llm_with_fallback(temperature=0.0)
    
    prompt = ChatPromptTemplate.from_template("""
Provide a highly standardized, technical definition for the industrial {item_type} known as "{name}".
The definition should be concise (2-3 sentences), focus on its primary function in an industrial control system, 
and use formal engineering terminology.

Return ONLY the definition text without any heading or label or asterisks.
""")
    chain = prompt | llm | StrOutputParser()
    
    taxonomy_data = {"instruments": [], "accessories": []}
    
    print(f"Generating descriptions for {len(instruments) + len(accessories)} items and preparing ingestion payload...")
    
    # Process Instruments
    print("Processing Instruments...")
    for name in tqdm(instruments):
        desc = generate_standardized_description(name, "instruments", chain)
        if desc:
            taxonomy_data["instruments"].append({
                "name": name,
                "category": "Instrument",
                "definition": desc,
                "aliases": []
            })
        else:
            print(f"Failed to generate description for instrument: {name}")

    # Process Accessories
    print("\nProcessing Accessories...")
    for name in tqdm(accessories):
        desc = generate_standardized_description(name, "accessories", chain)
        if desc:
            taxonomy_data["accessories"].append({
                "name": name,
                "category": "Accessory",
                "definition": desc,  # Now explicitly indexed due to our rag.py update
                "related_instruments": [],
                "aliases": []
            })
        else:
            print(f"Failed to generate description for accessory: {name}")
            
    # Save the generated descriptions to data/taxonomy.json
    taxonomy_path = os.path.join(backend_path, 'data', 'taxonomy.json')
    os.makedirs(os.path.dirname(taxonomy_path), exist_ok=True)
    with open(taxonomy_path, 'w', encoding='utf-8') as f:
        json.dump(taxonomy_data, f, indent=4)
    print(f"\nSaved generated taxonomy data to {taxonomy_path}")
    
    # Ingest into Pinecone vector store
    try:
        rag.index_taxonomy(taxonomy_data)
        print("Ingestion complete.")
    except Exception as e:
        print(f"Failed to ingest: {e}")

if __name__ == "__main__":
    generate_and_ingest()
