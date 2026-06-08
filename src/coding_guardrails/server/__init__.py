"""cg-owned llama.cpp server stack.

This module lets coding-guardrails own its llama.cpp build and model cache,
fully decoupled from LM Studio. Any user can reproduce the same server:

    cg server build                    # compile pinned llama.cpp (has gemma-4 fix)
    cg server download <model>         # fetch GGUF into cg's own cache
    cg server start --model <model>    # launch llama-server
    cg server status                   # running? version? listening?
    cg server stop
"""

from coding_guardrails.server.paths import (
    binary_path,
    build_dir,
    data_dir,
    log_file,
    models_dir,
    pid_file,
    run_dir,
    source_dir,
)
from coding_guardrails.server.version import (
    PINNED_COMMIT,
    PINNED_SHORT,
    binary_version,
    installed_commit,
    is_up_to_date,
)

__all__ = [
    "binary_path",
    "binary_version",
    "build_dir",
    "data_dir",
    "installed_commit",
    "is_up_to_date",
    "log_file",
    "models_dir",
    "pid_file",
    "PINNED_COMMIT",
    "PINNED_SHORT",
    "run_dir",
    "source_dir",
]
