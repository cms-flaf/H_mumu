#!/usr/bin/env python3
"""
Test that Setup can successfully load global.yaml for all eras.
This script verifies that the FLAF Setup class can load the configuration
for each Run3 era without errors.
"""

import sys
import os
from unittest import mock

# Add FLAF to Python path BEFORE any imports
ana_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_path = os.path.join(ana_path, "FLAF")
if flaf_path not in sys.path:
    sys.path.insert(0, flaf_path)
sys.path.insert(0, ana_path)

# Mock ROOT and other heavy dependencies before importing
sys.modules["ROOT"] = mock.MagicMock()

# Now import Setup
from FLAF.Common.Setup import Setup

# List of all Run3 eras to test
ERAS = [
    "Run3_2022",
    "Run3_2022EE",
    "Run3_2023",
    "Run3_2023BPix",
    "Run3_2024",
]


def test_setup_loading():
    """Test that Setup can load global.yaml for all eras."""
    failed_eras = []

    print(f"Testing Setup loading for {len(ERAS)} eras...")
    print(f"Analysis path: {ana_path}")
    print("-" * 80)

    for era in ERAS:
        print(f"\nTesting era: {era}")
        try:
            # Create Setup instance - this will load global.yaml and other configs
            setup = Setup(ana_path=ana_path, period=era)

            # Verify that global_params were loaded
            assert setup.global_params is not None, "global_params is None"
            assert len(setup.global_params.keys()) > 0, "global_params is empty"

            # Verify that physics model was loaded
            assert setup.phys_model is not None, "phys_model is None"

            print(f"✓ Successfully loaded Setup for {era}")
            print(
                f"  - Config paths considered: {len(setup.global_params.considered_paths)}"
            )
            print(f"  - Global params keys: {len(list(setup.global_params.keys()))}")
            print(f"  - Physics model: {setup.phys_model.name}")

        except Exception as e:
            print(f"✗ Failed to load Setup for {era}")
            print(f"  Error: {e}")
            import traceback

            traceback.print_exc()
            failed_eras.append((era, str(e)))

    print("\n" + "=" * 80)
    if failed_eras:
        print(f"FAILED: {len(failed_eras)}/{len(ERAS)} eras failed to load:")
        for era, error in failed_eras:
            print(f"  - {era}: {error}")
        return 1
    else:
        print(f"SUCCESS: All {len(ERAS)} eras loaded successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(test_setup_loading())
