"""Startup work that `soup train` repeats for nothing, measured on the Kaggle image.

Profiling the whole `soup train` command (cProfile, Kaggle T4 image) showed that a large part
of the time before the first step is not the model:

- `importlib.metadata.packages_distributions()` walks every installed distribution and stats
  every file in its RECORD (about 145,000 files on the Kaggle image). transformers calls it when
  it is imported (before our code runs) and trl calls it again when it is imported (inside
  `apply()`), and both keep the result in a module constant. The second call returns the same
  mapping in the same process, so trl gets a copy of the first result.
- `datasets` names the cache file of every `Dataset.map` by hashing the mapped function with
  dill. Soup's and TRL's map functions close over the tokenizer, so each map pickles the whole
  151,643-token tokenizer just to compute a cache key. A random cache key is the library's own
  fallback (used whenever hashing fails); the mapped data is the same, only the cache file name
  differs.
- our Triton kernels can compile while Python imports and the data is prepared, instead of
  after the model has loaded (see warmup.py).
"""

import contextlib
import copy
import sys


@contextlib.contextmanager
def reuse_package_scan():
    """While active, `packages_distributions()` returns a copy of the mapping transformers
    already computed in this process (if it did); restored afterwards."""
    import importlib.metadata as metadata

    mapping = getattr(sys.modules.get("transformers.utils.import_utils"),
                      "PACKAGE_DISTRIBUTION_MAPPING", None)
    if not isinstance(mapping, dict):
        yield
        return
    original = metadata.packages_distributions

    def packages_distributions():
        return copy.deepcopy(mapping)

    metadata.packages_distributions = packages_distributions
    try:
        yield
    finally:
        metadata.packages_distributions = original


def random_dataset_fingerprints():
    """Use datasets' random-fingerprint fallback instead of pickling every map function."""
    from datasets import arrow_dataset, fingerprint

    if getattr(fingerprint.update_fingerprint, "_soup_kernel", False):
        return

    def update_fingerprint(fingerprint_, transform, transform_args):
        return fingerprint.generate_random_fingerprint()

    update_fingerprint._soup_kernel = True
    fingerprint.update_fingerprint = update_fingerprint  # used by @fingerprint_transform
    arrow_dataset.update_fingerprint = update_fingerprint  # imported by name in Dataset.map


def base_model_from_argv(argv=None):
    """The `base:` of the config `soup train --config <path>` was started with, if any."""
    argv = sys.argv if argv is None else argv
    try:
        if len(argv) < 2 or argv[1] != "train" or "--config" not in argv:
            return None
        path = argv[argv.index("--config") + 1]
        import yaml

        with open(path, encoding="utf-8") as handle:
            base = (yaml.safe_load(handle) or {}).get("base")
        return base if isinstance(base, str) else None
    except Exception:  # noqa: BLE001
        return None
