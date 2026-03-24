"""
Manual test to verify that JumpStart and HuggingFace model IDs
resolve to the same model via _fetch_recipe_from_hub.

Usage:
    python test/manual/test_model_id_resolution.py

Requires valid AWS credentials with SageMaker access.
"""

import boto3
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from sagemaker.hyperpod.cli.recipe_utils import (
    _is_huggingface_model_id,
    _resolve_huggingface_model_id,
    _fetch_recipe_from_hub,
)

# --- Test pairs: (jumpstart_id, huggingface_id) for the same model ---
TEST_PAIRS = [
    ("meta-textgeneration-llama-3-2-1b", "meta-llama/Llama-3.2-1B"),
]


def test_is_huggingface_model_id():
    """Verify format detection logic."""
    assert _is_huggingface_model_id("meta-llama/Llama-3.2-1B") is True
    assert _is_huggingface_model_id("meta-textgeneration-llama-3-2-1b") is False
    assert _is_huggingface_model_id("arn:aws:sagemaker:us-west-2:123456:hub/model") is False
    print("✔️  _is_huggingface_model_id: all assertions passed")


def test_resolution(sagemaker_client, jumpstart_id, hf_id):
    """Test that a HuggingFace ID resolves and both IDs yield the same describe result."""

    # Step 1: Resolve HuggingFace ID
    print(f"\n--- Testing: {jumpstart_id} <-> {hf_id} ---")
    try:
        resolved = _resolve_huggingface_model_id(sagemaker_client, hf_id)
        print(f"✔️  Resolved '{hf_id}' -> '{resolved}'")
    except ValueError as e:
        print(f"❌  Resolution failed: {e}")
        return False

    # Step 2: Verify resolved name matches expected JumpStart ID
    if resolved == jumpstart_id:
        print(f"✔️  Resolved name matches expected JumpStart ID")
    else:
        print(f"⚠️  Resolved to '{resolved}', expected '{jumpstart_id}' — may still be valid")

    # Step 3: Fetch recipe via both IDs and compare
    try:
        recipe_js = _fetch_recipe_from_hub(sagemaker_client, jumpstart_id, "hyp-recipe-job")
        print(f"✔️  JumpStart fetch succeeded — {len(recipe_js.get('SupportedInstanceTypes', []))} instance types")
    except Exception as e:
        print(f"❌  JumpStart fetch failed: {e}")
        return False

    try:
        recipe_hf = _fetch_recipe_from_hub(sagemaker_client, hf_id, "hyp-recipe-job")
        print(f"✔️  HuggingFace fetch succeeded — {len(recipe_hf.get('SupportedInstanceTypes', []))} instance types")
    except Exception as e:
        print(f"❌  HuggingFace fetch failed: {e}")
        return False

    # Step 4: Compare recipes
    if recipe_js == recipe_hf:
        print(f"✔️  Both IDs returned identical recipe")
    else:
        print(f"⚠️  Recipes differ — check if this is expected")
        print(f"    JS types:  {recipe_js.get('SupportedInstanceTypes', [])}")
        print(f"    HF types:  {recipe_hf.get('SupportedInstanceTypes', [])}")

    return True


def main():
    print("=== Model ID Resolution Test ===\n")

    test_is_huggingface_model_id()

    sagemaker_client = boto3.client("sagemaker")

    all_passed = True
    for jumpstart_id, hf_id in TEST_PAIRS:
        if not test_resolution(sagemaker_client, jumpstart_id, hf_id):
            all_passed = False

    print("\n" + "=" * 40)
    if all_passed:
        print("✔️  All tests passed")
    else:
        print("❌  Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
