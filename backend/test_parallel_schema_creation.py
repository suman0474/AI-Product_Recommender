
import sys
import os
import time
import uuid
import logging
import concurrent.futures
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add backend to path
sys.path.append(os.path.abspath("."))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ParallelTest")

try:
    from common.tools.validation_tool import ValidationTool
except ImportError as e:
    logger.error(f"Failed to import ValidationTool: {e}")
    sys.exit(1)

PRODUCT_TYPES = [
    "pressure transmitter",
    "flow meter",
    "level transmitter",
    "control valve",
    "temperature sensor"
]

def test_schema_creation(product_type):
    session_id = f"test_par_{uuid.uuid4().hex[:8]}"
    logger.info(f"[{product_type}] Starting validation test (Session: {session_id})")
    
    start_time = time.time()
    try:
        # Initialize tool for each request to ensure isolation if needed, 
        # though it's designed to be reusable.
        tool = ValidationTool(enable_ppi=True, enable_standards_enrichment=True)
        
        # Simulate user input
        user_input = f"I need a {product_type} for an industrial application."
        
        result = tool.validate(
            user_input=user_input,
            expected_product_type=product_type, # Pass expected to skip extraction if efficient
            session_id=session_id
        )
        
        duration = time.time() - start_time
        success = result.get("success", False)
        
        # Count fields to verify schema population
        schema = result.get("schema", {})
        mandatory_count = 0
        if "mandatory" in schema:
            mandatory_count = len(schema["mandatory"])
        elif "mandatory_requirements" in schema:
            mandatory_count = len(schema["mandatory_requirements"])
            
        is_valid = result.get("is_valid", False)
        
        logger.info(f"[{product_type}] Finished in {duration:.2f}s. Success: {success}. Mandatory Fields: {mandatory_count}")
        return {
            "product_type": product_type,
            "success": success,
            "duration": duration,
            "mandatory_fields": mandatory_count,
            "is_valid": is_valid,
            "error": result.get("error")
        }
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"[{product_type}] Failed after {duration:.2f}s: {e}")
        return {
            "product_type": product_type,
            "success": False,
            "duration": duration,
            "error": str(e)
        }

def run_parallel_tests():
    print(f"\nStarting parallel tests for {len(PRODUCT_TYPES)} products: {', '.join(PRODUCT_TYPES)}")
    print("="*80)
    
    start_total = time.time()
    
    results = []
    # Use ThreadPoolExecutor for I/O bound tasks (API calls)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(PRODUCT_TYPES)) as executor:
        future_to_product = {executor.submit(test_schema_creation, pt): pt for pt in PRODUCT_TYPES}
        for future in concurrent.futures.as_completed(future_to_product):
            product = future_to_product[future]
            try:
                data = future.result()
                results.append(data)
            except Exception as exc:
                logger.error(f"{product} generated an exception: {exc}")
                results.append({
                    "product_type": product,
                    "success": False,
                    "duration": 0,
                    "error": str(exc)
                })
    
    total_duration = time.time() - start_total
    print("\n" + "="*80)
    print(f"PARALLEL TEST RESULTS (Total Time: {total_duration:.2f}s)")
    print(f"{'PRODUCT TYPE':<25} | {'STATUS':<10} | {'TIME':<8} | {'FIELDS':<8} | {'VALID':<8}")
    print("-" * 80)
    
    for res in results:
        status = "PASS" if res["success"] else "FAIL"
        error_msg = f" ({res.get('error')})" if res.get('error') else ""
        print(f"{res['product_type']:<25} | {status:<10} | {res['duration']:<6.2f}s | {res.get('mandatory_fields', 0):<8} | {str(res.get('is_valid', False)):<8}{error_msg}")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_parallel_tests()
