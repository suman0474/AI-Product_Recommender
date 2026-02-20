import logging
import time
from typing import Any, Dict, List, Optional

from .normalization_agent import TaxonomyNormalizationAgent
from .rag import SpecificationRetriever

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
            
            # Add user/catalog specs if available (from solution_integration logic)
            if "user_specifications" in item:
                product_entry["user_specifications"] = item["user_specifications"]
            if "catalog_specifications" in item:
                product_entry["catalog_specifications"] = item["catalog_specifications"]
            
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

    @staticmethod
    def prepare_for_search_workflow(
        solution_deep_agent_output: Dict[str, Any],
        mongodb_uri: Optional[str] = None,
        json_catalog_path: Optional[str] = None,
        memory=None
    ) -> Dict[str, Any]:
        """
        Complete pipeline for processing Solution Deep Agent output.
        
        This is the main orchestrator that bridges the Solution workflow
        with the Search workflow.
        
        Pipeline:
        1. Extract instruments and accessories from solution output
        2. Batch normalize using Taxonomy RAG
        3. Retrieve specifications from MongoDB/JSON for each item
        4. Aggregate specifications at product level
        5. Prepare final payload for Search Deep Agent
        """
        start_time = time.time()
        
        logger.info("[TaxonomyIntegration] Starting pipeline")
        
        # Extract input data
        instruments = solution_deep_agent_output.get("identified_instruments", [])
        accessories = solution_deep_agent_output.get("identified_accessories", [])
        solution_name = solution_deep_agent_output.get("solution_name", "Solution")
        user_input = solution_deep_agent_output.get("user_input", "")
        conversation_history = solution_deep_agent_output.get("conversation_history", [])
        
        total_items = len(instruments) + len(accessories)
        
        logger.info(
            f"[TaxonomyIntegration] Processing {total_items} items: "
            f"{len(instruments)} instruments, {len(accessories)} accessories"
        )
        
        try:
            # Step 1: Batch Normalize
            logger.info("[TaxonomyIntegration] Step 1: Batch normalization")
            normalizer = TaxonomyNormalizationAgent(memory=memory)
            
            normalization_result = normalizer.batch_normalize_solution_items(
                instruments=instruments,
                accessories=accessories,
                conversation_history=conversation_history,
                user_input=user_input
            )
            
            standardized_instruments = normalization_result["standardized_instruments"]
            standardized_accessories = normalization_result["standardized_accessories"]
            normalization_stats = normalization_result["normalization_stats"]
            
            logger.info(
                f"[TaxonomyIntegration] Normalization complete: "
                f"{normalization_stats['match_rate']*100:.1f}% matched"
            )
            
            # Step 2: Retrieve Specifications
            logger.info("[TaxonomyIntegration] Step 2: Specification retrieval")
            retriever = SpecificationRetriever(
                mongodb_uri=mongodb_uri,
                json_catalog_path=json_catalog_path
            )
            
            # Combine all items for batch retrieval
            all_normalized_items = standardized_instruments + standardized_accessories
            
            spec_results = retriever.get_specifications_batch(all_normalized_items)
            
            # Close MongoDB connection if used
            retriever.close()
            
            # Step 3: Aggregate and prepare final payload
            logger.info("[TaxonomyIntegration] Step 3: Aggregating results")
            
            items_with_specifications = []
            specs_found = 0
            
            for idx, item in enumerate(all_normalized_items):
                item_key = f"item_{idx}"
                spec_data = spec_results.get(item_key, {})
                
                # Merge normalized item with specification data
                aggregated_item = {
                    "id": idx + 1,
                    "original_name": item.get("product_name") or item.get("name", ""),
                    "canonical_name": item.get("canonical_name", ""),
                    "category": item.get("category", "unknown"),
                    "quantity": item.get("quantity", 1),
                    "taxonomy_matched": item.get("taxonomy_matched", False),
                    "match_source": item.get("match_source", "unknown"),
                    "specifications": spec_data.get("specifications", {}),
                    "spec_found": spec_data.get("spec_found", False),
                }
                
                # Include user-provided specs if available
                if "user_specifications" in item:
                    aggregated_item["user_specifications"] = item["user_specifications"]
                
                items_with_specifications.append(aggregated_item)
                
                if spec_data.get("spec_found"):
                    specs_found += 1
            
            # Calculate specification statistics
            spec_stats = {
                "total_items": total_items,
                "specifications_found": specs_found,
                "specifications_missing": total_items - specs_found,
                "spec_found_rate": round(specs_found / total_items, 3) if total_items > 0 else 0.0
            }
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"[TaxonomyIntegration] Pipeline complete in {processing_time_ms}ms: "
                f"{specs_found}/{total_items} specs found ({spec_stats['spec_found_rate']*100:.1f}%)"
            )
            
            return {
                "success": True,
                "solution_name": solution_name,
                "total_items": total_items,
                "items_with_specifications": items_with_specifications,
                "normalization_stats": normalization_stats,
                "specification_stats": spec_stats,
                "processing_time_ms": processing_time_ms,
                # For backward compatibility
                "standardized_instruments": standardized_instruments,
                "standardized_accessories": standardized_accessories,
            }
            
        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[TaxonomyIntegration] Pipeline failed: {e}", exc_info=True)
            
            return {
                "success": False,
                "error": str(e),
                "solution_name": solution_name,
                "total_items": total_items,
                "processing_time_ms": processing_time_ms
            }

    @staticmethod
    def prepare_search_payload_for_item(
        item: Dict[str, Any],
        solution_name: str = "Solution",
        solution_id: str = ""
    ) -> Dict[str, Any]:
        """
        Prepare a single item for the Search Deep Agent workflow.
        """
        return {
            "product_type": item.get("canonical_name", item.get("original_name", "")),
            "sample_input": f"{item.get('quantity', 1)}x {item.get('canonical_name', '')}",
            "solution_name": solution_name,
            "solution_id": solution_id,
            "user_specifications": item.get("user_specifications", {}),
            "catalog_specifications": item.get("specifications", {}),
            "quantity": item.get("quantity", 1),
            "category": item.get("category", "unknown"),
            "taxonomy_matched": item.get("taxonomy_matched", False)
        }

    @staticmethod
    def prepare_batch_search_payload(
        pipeline_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Prepare all items for batch processing by Search Deep Agent.
        """
        if not pipeline_result.get("success"):
            logger.error("[TaxonomyIntegration] Cannot prepare search payload: pipeline failed")
            return []
        
        solution_name = pipeline_result.get("solution_name", "Solution")
        items = pipeline_result.get("items_with_specifications", [])
        
        payloads = []
        for item in items:
            payload = TaxonomyIntegrationAdapter.prepare_search_payload_for_item(
                item=item,
                solution_name=solution_name
            )
            payloads.append(payload)
        
        logger.info(f"[TaxonomyIntegration] Prepared {len(payloads)} search payloads")
        
        return payloads

# Top-level functions for compatibility with original solution_integration.py
def prepare_for_search_workflow(*args, **kwargs):
    return TaxonomyIntegrationAdapter.prepare_for_search_workflow(*args, **kwargs)

def prepare_search_payload_for_item(*args, **kwargs):
    return TaxonomyIntegrationAdapter.prepare_search_payload_for_item(*args, **kwargs)

def prepare_batch_search_payload(*args, **kwargs):
    return TaxonomyIntegrationAdapter.prepare_batch_search_payload(*args, **kwargs)
