import logging
import json
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class TaxonomyIntegrationAdapter:
    """
    Adapter to bridge Solution Agent data with Product Search Workflow.
    Ensures that normalized taxonomy data and sample inputs are correctly
    formatted for downstream consumption.
    """

    @staticmethod
    def prepare_search_payload(
        normalized_items: List[Dict[str, Any]],
        solution_name: str = "Solution",
        solution_id: str = "",
        sample_inputs: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Convert normalized Solution Agent items into Product Search 'required_products' format.
        
        Args:
            normalized_items: List of items with 'canonical_name' and specs.
            solution_name: Name of the solution context.
            solution_id: ID of the solution session.
            sample_inputs: Map of item_name -> sample_input_string.

        Returns:
            Dict compatible with ProductSearchWorkflow.process_from_solution_workflow
        """
        required_products = []
        
        for item in normalized_items:
            # 1. Use Canonical Name -> Product Name -> Name
            product_type = item.get("canonical_name") or item.get("product_name") or item.get("name") or "Unknown Product"
            
            # 2. Flatten specs for search context
            specs = item.get("specifications", {})
            
            # 3. Attach sample input if available
            # We try to match by canonical name first, then raw name
            raw_name = item.get("name", "")
            sample_input = item.get("sample_input", "")
            if not sample_input and sample_inputs:
                sample_input = sample_inputs.get(product_type) or sample_inputs.get(raw_name) or ""

            product_entry = {
                "product_type": product_type,
                "quantity": item.get("quantity", 1),
                "application": f"{solution_name} - {item.get('category', 'General')}",
                "requirements": specs,
                "sample_input": sample_input,
                "original_name": raw_name,
                "taxonomy_matched": item.get("taxonomy_matched", False)
            }
            
            required_products.append(product_entry)

        return {
            "source": "solution_workflow",
            "solution_id": solution_id,
            "solution_name": solution_name,
            "required_products": required_products,
            "total_products": len(required_products)
        }

    @staticmethod
    def verify_normalization_status(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check if items have been properly normalized by Taxonomy RAG.
        Returns stats and list of un-normalized items.
        """
        total = len(items)
        normalized_count = sum(1 for i in items if i.get("taxonomy_matched"))
        
        missing = []
        for i in items:
            if not i.get("taxonomy_matched"):
                missing.append(i.get("name", "Unknown"))
                
        return {
            "total": total,
            "normalized": normalized_count,
            "coverage_pct": (normalized_count / total * 100) if total > 0 else 0,
            "missing_normalization": missing
        }
