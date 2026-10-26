# Copyright 2026-     FlagOS Contributors
"""Default ``raw`` entry → ``raw_hccl`` (Cube MM + HCCL AllReduce).

HCCL ring/tree scales with world; oneshot TLE does not. This is the practical
AscendC-Commit analogue available without AICPU device APIs.
"""

from __future__ import annotations

from gemm_ar_raw_hccl import IMPLEMENTED, prepare, run_gemm_ar

BACKEND = "raw"
