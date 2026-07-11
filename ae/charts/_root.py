"""Single source of truth for where chart data lives.

Upstream the chart modules resolved data files relative to the research
monorepo (``Path(__file__).parent.parent.parent``). In the AE package every
data file is staged under one cache dir instead, so all of them read from
``$TENSORDEX_AE_CACHE`` (default ``ae/cache``). ``download_cache.py`` populates
it from the published Hugging Face dataset.
"""
import os
from pathlib import Path

CACHE = Path(
    os.environ.get(
        "TENSORDEX_AE_CACHE",
        Path(__file__).resolve().parent.parent / "cache",
    )
).resolve()

# Charts referenced two roots upstream; both now map onto the cache dir, whose
# internal layout (results.db, tests/output/…, data/…, compression_data/…,
# model_level_reduction/…) mirrors the original relative paths.
PROJECT_ROOT = CACHE
PLOTS_DIR = CACHE
