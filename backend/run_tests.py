#!/usr/bin/env python3
"""
Test runner script for the bioacoustic web application backend.

This script provides a convenient way to run tests with different configurations
and filters using the minimal dependency testing framework.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --unit             # Run only unit tests
    python run_tests.py --integration      # Run only integration tests
    python run_tests.py --fast             # Skip slow tests
    python run_tests.py --module database  # Run specific module tests
    python run_tests.py --verbose          # Verbose output
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path


def run_pytest_command(args):
    """Run pytest with the specified arguments."""
    cmd = ['python', '-m', 'pytest']
    cmd.extend(args)

    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)

    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user.")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Run bioacoustic web app backend tests")

    # Test selection options
    parser.add_argument('--unit', action='store_true',
                       help='Run only unit tests')
    parser.add_argument('--integration', action='store_true',
                       help='Run only integration tests')
    parser.add_argument('--fast', action='store_true',
                       help='Skip slow tests')

    # Module selection
    parser.add_argument('--module', choices=['database', 'utilities', 'api', 'integration'],
                       help='Run tests for specific module')

    # Output options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet output')
    parser.add_argument('--coverage', action='store_true',
                       help='Run with coverage (requires pytest-cov)')

    # Other options
    parser.add_argument('--timing', action='store_true',
                       help='Show test timing information')
    parser.add_argument('--failed-first', action='store_true',
                       help='Run failed tests first')

    args = parser.parse_args()

    # Build pytest arguments
    pytest_args = ['tests/']

    # Test selection markers
    if args.unit:
        pytest_args.extend(['-m', 'unit'])
    elif args.integration:
        pytest_args.extend(['-m', 'integration'])

    if args.fast:
        pytest_args.extend(['-m', 'not slow'])

    # Module selection
    if args.module:
        module_map = {
            'database': 'tests/test_database.py',
            'utilities': 'tests/test_utilities.py',
            'api': 'tests/test_api_endpoints.py',
            'integration': 'tests/test_integration.py'
        }
        pytest_args = [module_map[args.module]]

    # Output options
    if args.verbose:
        pytest_args.append('-v')
    elif args.quiet:
        pytest_args.append('-q')

    # Coverage
    if args.coverage:
        pytest_args.extend(['--cov=modules', '--cov=main'])

    # Timing
    if args.timing:
        pytest_args.extend(['--durations=10'])

    # Failed first
    if args.failed_first:
        pytest_args.append('--lf')

    # Check if we're in the right directory
    if not Path('tests').exists():
        print("Error: tests directory not found. Please run this script from the backend directory.")
        return 1

    # Run the tests
    return run_pytest_command(pytest_args)


if __name__ == '__main__':
    sys.exit(main())