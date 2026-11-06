import os
import tempfile

# Use a throw-away project directory so tests never touch real data.
os.environ["NETRA_HOME"] = tempfile.mkdtemp(prefix="netra_test_")
