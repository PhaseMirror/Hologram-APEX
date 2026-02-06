#!/usr/bin/env python3
"""
Basic validation script for pikernel structure (no NumPy required).

This validates the module structure and imports without executing tests.
"""

import sys
import os
import importlib.util
from pathlib import Path


def check_module_exists(module_path):
    """Check if a Python module file exists."""
    return os.path.exists(module_path)


def check_module_imports(module_path, module_name):
    """Try to load module and check for syntax errors."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            return False, "Could not create module spec"
        
        # Note: We can't actually import because numpy is missing
        # but we can at least check the file is valid Python
        with open(module_path, 'r') as f:
            code = f.read()
            compile(code, module_path, 'exec')
        
        return True, "Module syntax valid"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Run validation checks."""
    print("=" * 70)
    print("π-kernel Structure Validation")
    print("=" * 70)
    
    base_dir = Path(__file__).parent.parent
    pikernel_dir = base_dir / "pikernel"
    
    if not pikernel_dir.exists():
        print(f"\n❌ ERROR: pikernel directory not found at {pikernel_dir}")
        return 1
    
    print(f"\n✓ Found pikernel directory at {pikernel_dir}")
    
    # Expected modules
    modules = [
        "__init__.py",
        "projectors.py",
        "l1proj.py",
        "certificates.py",
        "kernel.py",
        "ledger.py",
        "ledgerposeidon.py",
        "poseidon.py",
        "spectral.py",
        "rns.py",
        "routing.py",
        "mub_audit.py",
        "hologram_adapter.py",
        "example.py",
        "example_spectral_poseidon.py",
        "README.md",
    ]
    
    print("\nModule Files:")
    print("-" * 70)
    
    all_exist = True
    for module in modules:
        module_path = pikernel_dir / module
        exists = module_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {module}")
        if not exists:
            all_exist = False
    
    if not all_exist:
        print("\n❌ ERROR: Some module files are missing")
        return 1
    
    print("\n✓ All module files present")
    
    # Check syntax of Python modules
    print("\nModule Syntax Validation:")
    print("-" * 70)
    
    python_modules = [m for m in modules if m.endswith('.py')]
    all_valid = True
    
    for module in python_modules:
        module_path = pikernel_dir / module
        module_name = module[:-3]  # Remove .py
        
        valid, msg = check_module_imports(module_path, module_name)
        status = "✓" if valid else "✗"
        print(f"  {status} {module}: {msg}")
        
        if not valid:
            all_valid = False
    
    if not all_valid:
        print("\n❌ ERROR: Some modules have syntax errors")
        return 1
    
    print("\n✓ All modules have valid Python syntax")
    
    # Check test files
    print("\nTest Files:")
    print("-" * 70)
    
    tests_dir = base_dir / "tests"
    test_files = [
        "test_pikernel_projectors.py",
        "test_pikernel_l1proj.py",
        "test_pikernel_certificates.py",
        "test_pikernel_kernel.py",
        "test_pikernel_integration.py",
    ]
    
    all_tests_exist = True
    for test_file in test_files:
        test_path = tests_dir / test_file
        exists = test_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {test_file}")
        if not exists:
            all_tests_exist = False
    
    if not all_tests_exist:
        print("\n❌ ERROR: Some test files are missing")
        return 1
    
    print("\n✓ All test files present")
    
    # Check module line counts
    print("\nModule Statistics:")
    print("-" * 70)
    
    total_lines = 0
    for module in python_modules:
        module_path = pikernel_dir / module
        with open(module_path, 'r') as f:
            lines = len(f.readlines())
            total_lines += lines
            print(f"  {module:35s} {lines:5d} lines")
    
    print(f"  {'Total:':35s} {total_lines:5d} lines")
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ VALIDATION PASSED")
    print("=" * 70)
    print("\nStructure validation complete. All files present and syntactically valid.")
    print("\nTo run tests (requires NumPy):")
    print("  pip install numpy pytest")
    print("  pytest tests/test_pikernel_*.py -v")
    print("\nTo run examples (requires NumPy):")
    print("  python pikernel/example.py")
    print("  python pikernel/example_spectral_poseidon.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
