import os
import sys

# Ensure repository root on sys.path for local imports during tests
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

