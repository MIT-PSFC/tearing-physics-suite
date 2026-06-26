#Discover and run all test scripts in the tests/ directory.

"""
run_tests.py

Discover and run all test scripts in the tests/ directory.

Test output is printed by default. Use --quiet to suppress output.

Usage:
    cd /home/stubenj9/tearing-physics-suite/tests
    uv run run_tests.py [--list] [--test=NAME] [--all] [--quiet]

Options:
    --list          List all available test scripts and exit
    --test=NAME     Run a specific test (e.g., --test=delta_prime)
                    Can be used multiple times
    --all           Run all tests (default if no specific tests specified)
    --quiet         Suppress test output (show only summary)
    --help          Show this help message
"""

import os
import sys
import subprocess
from pathlib import Path


def discover_tests(test_dir=os.path.join(os.environ['TPSHOME'], 'tests')):
    """
    Discover all Python test scripts in the tests/ directory.
    
    Parameters
    ----------
    test_dir : Path, optional
        Directory to search (default: current directory).
        
    Returns
    -------
    dict
        {test_name: test_path, ...}
    """
    if test_dir is None:
        test_dir = Path.cwd()
    else:
        test_dir = Path(test_dir)
    
    tests = {}
    
    # Find all .py files in tests/ (non-recursive, so maxdepth=1)
    for py_file in sorted(test_dir.glob("*.py")):
        # Skip run_tests.py itself
        if py_file.name == "run_tests.py":
            continue
        
        # Extract test name (remove .py extension)
        test_name = py_file.stem
        tests[test_name] = py_file
    
    return tests


def list_tests(tests):
    """Print available tests."""
    if not tests:
        print("No test scripts found.")
        return
    
    print("=" * 60)
    print("Available Test Scripts")
    print("=" * 60)
    for i, name in enumerate(sorted(tests.keys()), 1):
        print(f"{i:2d}. {name}")
    print("=" * 60)


def run_test(test_path, verbose=False):
    """
    Run a single test script and return result.
    
    Parameters
    ----------
    test_path : Path
        Path to test script.
    verbose : bool
        If True, stream output in real-time as the test runs.
        If False, capture output and display it after completion.
        
    Returns
    -------
    dict
        {
            'name': test_name,
            'path': test_path,
            'returncode': int,
            'success': bool,
            'output': str,
            'error': str,
        }
    """
    test_name = test_path.stem
    print(f"\n{'=' * 60}")
    print(f"Running: {test_name}")
    print(f"{'=' * 60}")
    
    try:
        if verbose:
            # Stream output in real-time (new logic)
            result = subprocess.run(
                [sys.executable, str(test_path)],
                timeout=3600,  # 1 hour timeout
            )
            success = result.returncode == 0
            status = "✓ PASSED" if success else "✗ FAILED"
            print(f"\n{status} (exit code: {result.returncode})")
            
            return {
                'name': test_name,
                'path': test_path,
                'returncode': result.returncode,
                'success': success,
                'output': '',
                'error': '',
            }
        else:
            # Capture output for display after completion (old logic)
            result = subprocess.run(
                [sys.executable, str(test_path)],
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            
            success = result.returncode == 0
            
            # Display output
            if not success:
                # Always show full output on failure
                if result.stdout:
                    print("STDOUT:")
                    print(result.stdout)
                if result.stderr:
                    print("STDERR:")
                    print(result.stderr)
            else:
                # Print last few lines if successful
                lines = result.stdout.split('\n') if result.stdout else []
                for line in lines[-10:]:
                    if line.strip():
                        print(line)
            
            status = "✓ PASSED" if success else "✗ FAILED"
            print(f"\n{status} (exit code: {result.returncode})")
            
            return {
                'name': test_name,
                'path': test_path,
                'returncode': result.returncode,
                'success': success,
                'output': result.stdout,
                'error': result.stderr,
            }
        
    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT (>1 hour)")
        return {
            'name': test_name,
            'path': test_path,
            'returncode': -1,
            'success': False,
            'output': '',
            'error': 'Timeout',
        }
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return {
            'name': test_name,
            'path': test_path,
            'returncode': -1,
            'success': False,
            'output': '',
            'error': str(e),
        }


def main():
    """Main entry point."""
    # Parse arguments
    test_names = []
    list_only = False
    verbose = True  # Default to True; can be suppressed with --quiet
    run_all = False
    
    for arg in sys.argv[1:]:
        if arg == "--help":
            print(__doc__)
            sys.exit(0)
        elif arg == "--list":
            list_only = True
        elif arg == "--all":
            run_all = True
        elif arg == "--quiet":
            verbose = False
        elif arg.startswith("--test="):
            test_names.append(arg.split("=", 1)[1])
        else:
            print(f"Unknown argument: {arg}")
            print(__doc__)
            sys.exit(1)
    
    # Print warning about verbose output default
    if verbose:
        print("============================================================")
        print("Test output is being printed. Use --quiet to suppress output.")
        print("============================================================")
        print()
    
    # Discover tests
    tests = discover_tests()
    
    if not tests:
        print("No test scripts found in current directory.")
        sys.exit(1)
    
    # List only
    if list_only:
        list_tests(tests)
        sys.exit(0)
    
    # Determine which tests to run
    if not test_names and not run_all:
        # Default: run all
        tests_to_run = sorted(tests.keys())
    elif run_all:
        tests_to_run = sorted(tests.keys())
    else:
        # Run specified tests
        tests_to_run = []
        for name in test_names:
            if name in tests:
                tests_to_run.append(name)
            else:
                # Try partial match
                matches = [t for t in tests if name.lower() in t.lower()]
                if matches:
                    tests_to_run.extend(matches)
                else:
                    print(f"Warning: No test matching '{name}' found.")
        tests_to_run = sorted(set(tests_to_run))
    
    if not tests_to_run:
        print("No tests to run.")
        sys.exit(1)
    
    # Run tests
    print(f"\nRunning {len(tests_to_run)} test(s)...\n")
    
    results = []
    for test_name in tests_to_run:
        result = run_test(tests[test_name], verbose=verbose)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for res in results if res['success'])
    failed = sum(1 for res in results if not res['success'])
    
    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"{status} {result['name']:<40s} (exit: {result['returncode']})")
    
    print("-" * 60)
    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    print("=" * 60)
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
