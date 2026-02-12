"""
Final Verification Script
Run this after deletion to ensure the system still works
"""

from prompts_library import load_prompt_sections

def verify_consolidated_prompts():
    """Verify all consolidated prompt files load correctly."""
    print("=" * 80)
    print("FINAL VERIFICATION - POST-DELETION")
    print("=" * 80)
    
    consolidated_files = [
        'standards_deep_agent_prompts',
        'llm_specs_prompts',
        'sales_workflow_prompts',
        'solution_workflow_prompts',
        'rag_prompts',
        'index_rag_prompts',
        'intent_prompts',
        'ranking_prompts',
        'potential_product_index_prompts',
        'optimized_parallel_agent_prompts',
        'sales_agent_prompts',
        'shared_agent_prompts'
    ]
    
    all_good = True
    total_sections = 0
    
    for file in consolidated_files:
        try:
            sections = load_prompt_sections(file)
            section_count = len(sections)
            total_sections += section_count
            print(f"✓ {file}: {section_count} sections")
        except Exception as e:
            print(f"✗ {file}: FAILED - {e}")
            all_good = False
    
    print("\n" + "=" * 80)
    if all_good:
        print(f"✅ SUCCESS! All {len(consolidated_files)} files work correctly")
        print(f"📊 Total sections available: {total_sections}")
        print("\n🎉 Consolidation complete and verified!")
    else:
        print("⚠️  Some files failed to load - please investigate")
    print("=" * 80)
    
    return all_good

if __name__ == "__main__":
    verify_consolidated_prompts()
