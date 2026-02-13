"""
Simple test to validate V2 prompt is loaded correctly.
Avoids circular import issues by testing prompt loading directly.
"""

import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("="*70)
print("V2 INTENT CLASSIFICATION PROMPT VALIDATION")
print("="*70)

# Test 1: Check if V2 file exists
print("\n1. Checking if V2 prompt file exists...")
v2_path = os.path.join(os.path.dirname(__file__), '..', 'prompts_library', 'intent_classification_prompts_v2.txt')
v2_exists = os.path.exists(v2_path)
print(f"   V2 prompt file exists: {v2_exists}")
if v2_exists:
    print(f"   [OK] Path: {v2_path}")
    file_size = os.path.getsize(v2_path)
    print(f"   [OK] Size: {file_size:,} bytes")
else:
    print(f"   [FAIL] V2 prompt file not found at: {v2_path}")
    sys.exit(1)

# Test 2: Check if V1 backup exists
print("\n2. Checking if V1 backup was created...")
v1_backup_path = os.path.join(os.path.dirname(__file__), '..', 'prompts_library', 'intent_classification_prompts_v1_backup.txt')
v1_backup_exists = os.path.exists(v1_backup_path)
print(f"   V1 backup exists: {v1_backup_exists}")
if v1_backup_exists:
    print(f"   [OK] Backup created at: {v1_backup_path}")
else:
    print(f"   [WARN] No backup found (this is OK if running for first time)")

# Test 3: Validate V2 prompt content
print("\n3. Validating V2 prompt content...")
with open(v2_path, 'r', encoding='utf-8') as f:
    v2_content = f.read()

# Check for V2-specific improvements
v2_markers = {
    "Clear scope definition": "Acceptable Topics:",
    "Out-of-scope section": "Out-of-Scope Topics",
    "Enhanced INVALID examples": "How to program a PLC",
    "Decision flow": "DECISION FLOW",
    "Edge cases section": "EDGE CASES & DISAMBIGUATION",
    "Confidence thresholds": "Confidence Threshold:",
}

print("   Checking for V2 improvements:")
all_present = True
for marker_name, marker_text in v2_markers.items():
    present = marker_text in v2_content
    status = "[OK]" if present else "[FAIL]"
    print(f"   {status} {marker_name}: {'Present' if present else 'MISSING'}")
    if not present:
        all_present = False

if not all_present:
    print("\n   [FAIL] V2 prompt is missing some expected improvements!")
    sys.exit(1)

# Test 4: Check intent_tools.py was updated
print("\n4. Checking if intent_tools.py uses V2 prompt...")
intent_tools_path = os.path.join(os.path.dirname(__file__), '..', 'tools', 'intent_tools.py')
with open(intent_tools_path, 'r', encoding='utf-8') as f:
    intent_tools_content = f.read()

v2_loading = 'intent_classification_prompts_v2' in intent_tools_content
fallback_logic = 'except' in intent_tools_content and 'FileNotFoundError' in intent_tools_content

print(f"   V2 prompt referenced: {v2_loading}")
print(f"   Fallback logic present: {fallback_logic}")

if v2_loading and fallback_logic:
    print("   [OK] intent_tools.py correctly configured for V2 with fallback")
else:
    print("   [FAIL] intent_tools.py not properly configured")
    sys.exit(1)

# Test 5: Load prompts using the library
print("\n5. Testing prompt loading with prompts_library...")
try:
    from prompts_library import load_prompt_sections

    # Try loading V2
    try:
        prompts_v2 = load_prompt_sections("intent_classification_prompts_v2", default_section="CLASSIFICATION")
        print("   [OK] V2 prompts loaded successfully")
        print(f"   [OK] Sections found: {list(prompts_v2.keys())}")

        # Check main sections
        has_classification = "CLASSIFICATION" in prompts_v2
        has_quick = "QUICK_CLASSIFICATION" in prompts_v2

        print(f"   [OK] CLASSIFICATION section: {'Present' if has_classification else 'MISSING'}")
        print(f"   [OK] QUICK_CLASSIFICATION section: {'Present' if has_quick else 'MISSING'}")

        if has_classification:
            classification_length = len(prompts_v2["CLASSIFICATION"])
            print(f"   [OK] CLASSIFICATION prompt length: {classification_length:,} characters")

            # Should be significantly longer than V1 due to improvements
            if classification_length > 5000:
                print(f"   [OK] V2 prompt is comprehensive (>5000 chars)")
            else:
                print(f"   [WARN] V2 prompt might be incomplete ({classification_length} chars)")

    except Exception as e:
        print(f"   [FAIL] Failed to load V2 prompts: {e}")
        sys.exit(1)

except ImportError as e:
    print(f"   [FAIL] Failed to import prompts_library: {e}")
    sys.exit(1)

# Test 6: Compare V2 improvements over V1
print("\n6. Comparing V2 improvements over V1...")
v1_path = os.path.join(os.path.dirname(__file__), '..', 'prompts_library', 'intent_classification_prompts_v1_backup.txt')
if os.path.exists(v1_path):
    with open(v1_path, 'r', encoding='utf-8') as f:
        v1_content = f.read()

    v1_length = len(v1_content)
    v2_length = len(v2_content)

    print(f"   V1 length: {v1_length:,} characters")
    print(f"   V2 length: {v2_length:,} characters")
    print(f"   Increase: {v2_length - v1_length:,} characters ({((v2_length/v1_length - 1) * 100):.1f}% larger)")

    if v2_length > v1_length:
        print(f"   [OK] V2 is more comprehensive than V1")
    else:
        print(f"   [WARN] V2 should be larger than V1")
else:
    print("   [WARN] V1 backup not found for comparison")

# Summary
print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)
print("[OK] V2 prompt file exists and is valid")
print("[OK] V2 contains all expected improvements")
print("[OK] intent_tools.py configured to use V2 with fallback")
print("[OK] Prompt loading works correctly")
print("="*70)
print("\nPath 1 (Quick Fix) completed successfully!")
print("\nNext steps:")
print("1. Run the full test suite: python backend/tests/test_intent_classification_v2_validation.py")
print("2. Test with real queries via the API")
print("3. Monitor classification accuracy over the next few days")
print("="*70 + "\n")
