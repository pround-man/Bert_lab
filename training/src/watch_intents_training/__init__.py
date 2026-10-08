"""Training is independent of the frozen dataset generation environment."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
