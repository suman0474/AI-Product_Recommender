# advanced_parameters.py
# Advanced parameter discovery using pure LLM-based approach.
# Focuses on latest parameters added in the past 6 months for the detected product type.

import json
import logging
from typing import Dict, Any, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import re
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# --- Pydantic Models ---
class VendorInfo(BaseModel):
    """Vendor information"""
    vendor: str
    confidence: float = 0.9

# --- Main Discovery Class ---
class AdvancedParametersDiscovery:
    """Discovers advanced parameters from vendor websites"""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY1")
        )

    def get_top_vendors(self, product_type: str) -> List[str]:
        """Get top 5 vendors for the product type"""
        prompt = ChatPromptTemplate.from_template("""
List the top 5 vendors for "{product_type}" with recent innovations (past 6 months).
Return ONLY a JSON array of vendor names: ["ABB", "Emerson", "Siemens"]
""")
        try:
            response = (prompt | self.llm | StrOutputParser()).invoke({"product_type": product_type})
            vendors = json.loads(response.strip().replace('```json', '').replace('```', ''))
            return vendors[:5] if isinstance(vendors, list) else []
        except:
            return ["ABB", "Emerson", "Siemens", "Endress+Hauser", "Honeywell"][:5]

    def get_vendor_parameters(self, vendor: str, product_type: str) -> List[str]:
        """Get latest parameters from LLM for a vendor"""
        try:
            return self._query_vendor_parameters(vendor, product_type)
        except Exception as e:
            logging.warning(f"Failed to get parameters for {vendor}: {e}")
            return []

    def _query_vendor_parameters(self, vendor: str, product_type: str) -> List[str]:
        """Query LLM for latest parameters from a vendor"""
        prompt = ChatPromptTemplate.from_template("""
List 6 LATEST advanced parameters for {vendor}'s {product_type} (past 6 months).
Focus on: IoT, Industry 4.0, AI, protocols, cybersecurity, cloud.
Return JSON array in snake_case: ["opc_ua_server", "mqtt_protocol"]
""")
        try:
            response = (prompt | self.llm | StrOutputParser()).invoke({"vendor": vendor, "product_type": product_type})
            params = json.loads(response.strip().replace('```json', '').replace('```', ''))
            return [re.sub(r'[^a-z0-9_]', '', p.lower().replace(' ', '_')) for p in params if isinstance(p, str)][:6]
        except:
            return []

    def get_generic_parameters(self, product_type: str) -> List[str]:
        """Get generic advanced parameters for product type"""
        prompt = ChatPromptTemplate.from_template("""
List 15 latest advanced parameters for {product_type} (past 6 months).
Focus on: IoT, Industry 4.0, AI, protocols, cybersecurity, cloud.
Return JSON array in snake_case.
""")
        try:
            response = (prompt | self.llm | StrOutputParser()).invoke({"product_type": product_type})
            params = json.loads(response.strip().replace('```json', '').replace('```', ''))
            return [str(p) for p in params][:15] if isinstance(params, list) else []
        except:
            return []

def get_existing_parameters(product_type: str) -> set:
    """Get existing parameters from schema"""
    try:
        from loading import load_requirements_schema
        schema = load_requirements_schema(product_type)
        if not schema:
            return set()
        
        params = set()
        for req_type in ["mandatory_requirements", "optional_requirements"]:
            for fields in schema.get(req_type, {}).values():
                if isinstance(fields, dict):
                    params.update(fields.keys())
        return params
    except:
        return set()

def convert_parameters_to_human_readable(params: List[str]) -> Dict[str, str]:
    """Convert multiple parameters to human-readable format using LLM (batch processing)"""
    if not params:
        return {}
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",  # Faster model
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        prompt = ChatPromptTemplate.from_template("""
Convert to human-readable. Capitalize acronyms (OPC UA, MQTT, IoT, AI, HART, IEC).
Parameters: {params}
Return JSON: {{"key": "Name"}}
""")
        
        response = (prompt | llm | StrOutputParser()).invoke({"params": json.dumps(params[:30])})
        result = json.loads(response.strip().replace('```json', '').replace('```', ''))
        return result if isinstance(result, dict) else {}
    except Exception as e:
        logging.warning(f"Batch conversion failed: {e}")
        return {p: p.replace('_', ' ').title() for p in params}

def discover_advanced_parameters(product_type: str) -> Dict[str, Any]:
    """Discover latest advanced parameters for product type"""
    try:
        discovery = AdvancedParametersDiscovery()
        existing = get_existing_parameters(product_type)
        vendors = discovery.get_top_vendors(product_type)
        
        logging.info(f"Querying {len(vendors)} vendors in parallel...")
        
        # Collect parameters from all vendors IN PARALLEL
        all_params = set()
        vendor_data = []
        
        # Use ThreadPoolExecutor for parallel vendor queries
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(discovery.get_vendor_parameters, vendor, product_type): vendor for vendor in vendors}
            
            for future in futures:
                vendor = futures[future]
                try:
                    params = future.result(timeout=15)  # 15 second timeout per vendor
                    if params:
                        all_params.update(params)
                        vendor_data.append({
                            "vendor": vendor, 
                            "parameters": params
                        })
                        logging.info(f"✓ {vendor}: {len(params)} parameters")
                except Exception as e:
                    logging.warning(f"✗ {vendor} failed: {e}")
        
        # Normalize and deduplicate
        normalized_map = {}
        for param in all_params:
            norm = param.lower().replace('_', '')
            if norm and norm not in normalized_map:
                normalized_map[norm] = param
        
        # Filter against existing schema
        existing_norm = {p.lower().replace('_', '') for p in existing}
        new_params = [p for norm, p in normalized_map.items() if norm not in existing_norm][:15]
        
        # Batch convert only unique new parameters (faster)
        logging.info(f"Converting {len(new_params)} parameters to human-readable...")
        human_readable_map = convert_parameters_to_human_readable(new_params)
        
        # Add human-readable names to vendor data (only for displayed params)
        for vd in vendor_data:
            vd["parameters"] = [
                {"key": p, "name": human_readable_map.get(p, p.replace('_', ' ').title())}
                for p in vd["parameters"][:6]  # Limit to 6 per vendor
            ]
        
        # Create parameters with human-readable names
        unique_parameters_with_names = [
            {"key": param, "name": human_readable_map.get(param, param.replace('_', ' ').title())}
            for param in new_params
        ]
        
        return {
            "product_type": product_type,
            "vendor_parameters": vendor_data,
            "unique_parameters": unique_parameters_with_names,
            "total_vendors_searched": len(vendor_data),
            "total_unique_parameters": len(new_params),
            "existing_parameters_filtered": len(all_params) - len(new_params)
        }
    except Exception as e:
        logging.error(f"Discovery failed: {e}")
        discovery = AdvancedParametersDiscovery()
        generic_params = discovery.get_generic_parameters(product_type)
        
        # Batch convert fallback parameters
        human_readable_map = convert_parameters_to_human_readable(generic_params)
        
        return {
            "product_type": product_type,
            "unique_parameters": [
                {"key": p, "name": human_readable_map.get(p, p.replace('_', ' ').title())}
                for p in generic_params
            ],
            "fallback": True
        }

