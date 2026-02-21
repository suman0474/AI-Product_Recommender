import os
import sys
import json

# Add backend to path
backend_path = os.path.abspath(os.path.join(os.getcwd(), 'backend'))
sys.path.insert(0, backend_path)

from dotenv import load_dotenv
load_dotenv(os.path.join(backend_path, '.env'))

from common.core.mongodb_manager import mongodb_manager, is_mongodb_available
from common.core.azure_blob_file_manager import azure_blob_file_manager
from common.services.schema_service import schema_service
from common.services.llm.fallback import create_llm_with_fallback

def check_connections():
    print("Checking Connections...")
    mongo_health = mongodb_manager.health_check()
    print(f"MongoDB Status: {mongo_health.get('status')} (Connected: {mongo_health.get('connected')})")
    
    azure_health = azure_blob_file_manager.health_check()
    print(f"Azure Blob Status: {azure_health.get('status')} (Connected: {azure_health.get('connected')})")
    
    return mongo_health.get('connected'), azure_health.get('connected')

def get_product_types():
    print("\nFetching Product Types...")
    types = schema_service.get_all_product_types()
    print(f"Found {len(types)} product types.")
    return types

def process_product_types(product_types):
    print("\nProcessing and Segregating Product Types...")
    # We will use the LLM to segregate and describe
    llm = create_llm_with_fallback()
    
    types_str = ", ".join(product_types)
    prompt = f"""
    You are an expert in industrial instrumentation and automation.
    I have a list of product types: {types_str}
    
    Please segregate these into two categories: "Instruments" and "Accessories".
    For each product type, provide a very short one-sentence description.
    
    Return the result in the following JSON format:
    {{
        "instruments": [
            {{"name": "Product Name", "description": "Short description"}}
        ],
        "accessories": [
            {{"name": "Product Name", "description": "Short description"}}
        ]
    }}
    """
    
    try:
        response = llm.invoke(prompt)
        # Handle cases where response might be wrapped in ```json ... ```
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        data = json.loads(content)
        return data
    except Exception as e:
        print(f"Error processing with LLM: {e}")
        return None

def main():
    mongo_connected, azure_connected = check_connections()
    
    if not mongo_connected:
        print("Warning: MongoDB not connected. Results might be limited.")
        
    product_types = get_product_types()
    
    if not product_types:
        print("No product types found.")
        return

    processed_data = process_product_types(product_types)
    
    if processed_data:
        output_file = "product_types_segregated.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("PRODUCT TYPES SEGREGATION\n")
            f.write("="*70 + "\n\n")
            
            f.write("INSTRUMENTS\n")
            f.write("-" * 20 + "\n")
            for item in processed_data.get("instruments", []):
                f.write(f"- {item['name']}: {item['description']}\n")
            
            f.write("\nACCESSORIES\n")
            f.write("-" * 20 + "\n")
            for item in processed_data.get("accessories", []):
                f.write(f"- {item['name']}: {item['description']}\n")
        
        print(f"\nSaved segregated product types to {output_file}")
    else:
        print("Failed to process product types.")

if __name__ == "__main__":
    main()
