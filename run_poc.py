#!/usr/bin/env python3
"""
Thin entry point for the TypeSafe LLM-as-a-Judge POC.

The implementation lives in the `typesafe_poc` package (see its docstring
for the pipeline overview and module layout); this script just wires it up
so the POC keeps its documented `python run_poc.py ...` usage.
"""

import sys

from typesafe_poc.cli import main

if __name__ == "__main__":
    sys.exit(main())
