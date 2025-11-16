"""
Test script to verify OpenRouter configuration loading with environment variables.
This tests that ${OPENROUTER_API_KEY} is properly expanded from .env file.
"""

import os
import sys
from pathlib import Path
import importlib.util

# Load config module directly without going through __init__.py
config_file = Path(__file__).parent / "evolve_agent" / "config.py"
spec = importlib.util.spec_from_file_location("config_module", config_file)
config_module = importlib.util.module_from_spec(spec)
sys.modules["config_module"] = config_module
spec.loader.exec_module(config_module)


def test_openrouter_config():
    """Test loading OpenRouter config with env var expansion"""

    print("Testing OpenRouter Configuration")
    print("=" * 80)

    # Load the OpenRouter config
    config_path = "configs/openrouter_config.yaml"
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        return False

    try:
        config = config_module.load_config(config_path)
        print(f"[OK] Successfully loaded config from {config_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        return False

    # Check LLM configuration
    print("\n" + "=" * 80)
    print("LLM Configuration:")
    print("-" * 80)
    print(f"API Base: {config.llm.api_base}")
    print(f"API Key: {config.llm.api_key[:20]}..." if config.llm.api_key else "None")
    print(f"Models: {[m.name for m in config.llm.models]}")

    # Check if API key was expanded (should not contain "${")
    if config.llm.api_key and "${" in config.llm.api_key:
        print("[ERROR] API key was not expanded! Still contains ${}")
        return False
    elif config.llm.api_key and config.llm.api_key.startswith("sk-or-"):
        print("[OK] API key properly expanded from environment variable")
    else:
        print("[WARNING] API key doesn't look like an OpenRouter key")

    # Check Reward Model configuration
    print("\n" + "=" * 80)
    print("Reward Model Configuration:")
    print("-" * 80)
    print(f"Model Type: {config.rewardmodel.model_type}")
    print(f"Model Name: {config.rewardmodel.model_name}")
    print(f"Base URL: {config.rewardmodel.base_url}")
    print(f"API Key: {config.rewardmodel.api_key[:20]}..." if config.rewardmodel.api_key else "None")

    # Check if reward model API key was expanded
    if config.rewardmodel.api_key and "${" in config.rewardmodel.api_key:
        print("[ERROR] Reward model API key was not expanded! Still contains ${}")
        return False
    elif config.rewardmodel.api_key and config.rewardmodel.api_key.startswith("sk-or-"):
        print("[OK] Reward model API key properly expanded from environment variable")
    else:
        print("[WARNING] Reward model API key doesn't look like an OpenRouter key")

    # Check best solution directory config
    print("\n" + "=" * 80)
    print("Best Solution Configuration:")
    print("-" * 80)
    print(f"Best Solution Dir: {config.best_solution_dir or '(default: output_dir/best_solution)'}")
    print("[OK] Best solution auto-save system configured")

    # Summary
    print("\n" + "=" * 80)
    print("Summary:")
    print("-" * 80)

    all_good = True

    checks = [
        ("Config file loaded", True),
        ("LLM API key expanded", config.llm.api_key and "${" not in config.llm.api_key),
        ("Reward model API key expanded", config.rewardmodel.api_key and "${" not in config.rewardmodel.api_key),
        ("OpenRouter base URL set", config.llm.api_base == "https://openrouter.ai/api/v1"),
        ("Reward model base URL set", config.rewardmodel.base_url == "https://openrouter.ai/api/v1"),
    ]

    for check_name, passed in checks:
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {check_name}")
        all_good = all_good and passed

    print("=" * 80)
    if all_good:
        print("[OK] All checks passed! OpenRouter configuration is working correctly.")
        return True
    else:
        print("[FAIL] Some checks failed. Please review the configuration.")
        return False


if __name__ == "__main__":
    success = test_openrouter_config()
    sys.exit(0 if success else 1)
