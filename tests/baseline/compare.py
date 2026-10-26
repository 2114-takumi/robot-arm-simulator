"""Compare two captured simulation bundles without modifying either bundle."""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


class Comparator:
    def __init__(self, args):
        self.args = args
        self.failures = 0

    def fail(self, message):
        self.failures += 1
        print(f"FAIL {message}")

    def metadata(self, reference, candidate, path):
        numeric = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool)
        if numeric(reference) and numeric(candidate):
            finite = math.isfinite(reference) and math.isfinite(candidate)
            matches = reference == candidate if self.args.exact else np.isclose(
                candidate, reference, rtol=self.args.rtol, atol=self.args.atol,
                equal_nan=False,
            )
            if not finite or not matches:
                self.fail(f"{path}: reference={reference!r}, candidate={candidate!r}")
        elif type(reference) is not type(candidate):
            self.fail(f"{path}: types differ ({type(reference).__name__}, {type(candidate).__name__})")
        elif isinstance(reference, dict):
            self.keys(set(reference), set(candidate), path)
            for key in sorted(reference.keys() & candidate.keys()):
                self.metadata(reference[key], candidate[key], f"{path}.{key}")
        elif isinstance(reference, list):
            if len(reference) != len(candidate):
                self.fail(f"{path}: lengths differ ({len(reference)}, {len(candidate)})")
            for index, (expected, actual) in enumerate(zip(reference, candidate)):
                self.metadata(expected, actual, f"{path}[{index}]")
        elif reference != candidate:
            self.fail(f"{path}: reference={reference!r}, candidate={candidate!r}")

    def keys(self, reference, candidate, path):
        if reference != candidate:
            self.fail(
                f"{path}: missing keys={sorted(reference - candidate)}, "
                f"extra keys={sorted(candidate - reference)}"
            )

    def arrays(self, reference_path, candidate_path):
        with np.load(reference_path, allow_pickle=False) as reference, np.load(
            candidate_path, allow_pickle=False
        ) as candidate:
            self.keys(set(reference.files), set(candidate.files), "arrays")
            for key in sorted(set(reference.files) & set(candidate.files)):
                expected, actual = reference[key], candidate[key]
                path = f"arrays.{key}"
                if expected.shape != actual.shape:
                    self.fail(f"{path}: shapes differ ({expected.shape}, {actual.shape})")
                    continue
                if expected.dtype != actual.dtype:
                    self.fail(f"{path}: dtypes differ ({expected.dtype}, {actual.dtype})")
                    continue
                if expected.dtype.kind not in "biufc":
                    self.fail(f"{path}: unsupported nonnumeric dtype {expected.dtype}")
                    continue
                if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(actual)):
                    self.fail(f"{path}: nonfinite values are not allowed")
                    continue
                identical = np.array_equal(expected, actual)
                if not identical:
                    working_dtype = np.clongdouble if np.iscomplexobj(expected) else np.longdouble
                    difference = np.abs(actual.astype(working_dtype) - expected.astype(working_dtype))
                    print(f"DIFF {path}: max_abs_difference={np.max(difference):.17g}")
                matches = identical if self.args.exact or expected.dtype.kind == "b" else np.allclose(
                    actual, expected, rtol=self.args.rtol, atol=self.args.atol,
                    equal_nan=False,
                )
                if not matches:
                    self.fail(f"{path}: values differ")


def read_json(path):
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--exact", action="store_true", help="Require identical numerical values.")
    args = parser.parse_args(argv)
    if any(not math.isfinite(value) or value < 0 for value in (args.rtol, args.atol)):
        parser.error("--rtol and --atol must be finite and nonnegative")

    comparator = Comparator(args)
    try:
        for filename in ("summary.json", "plots.json"):
            comparator.metadata(
                read_json(args.reference_dir / filename),
                read_json(args.candidate_dir / filename),
                filename,
            )
        comparator.arrays(args.reference_dir / "arrays.npz", args.candidate_dir / "arrays.npz")
        reference_env = read_json(args.reference_dir / "environment.json")
        candidate_env = read_json(args.candidate_dir / "environment.json")
        for name, environment in (("reference", reference_env), ("candidate", candidate_env)):
            if not isinstance(environment, dict) or environment.get("status") != "complete":
                comparator.fail(f"{name} environment.json: status must be 'complete'")
        if isinstance(reference_env, dict) and isinstance(candidate_env, dict):
            reference_versions = reference_env.get("versions", {})
            candidate_versions = candidate_env.get("versions", {})
            if reference_versions != candidate_versions:
                print(f"NOTICE versions differ: reference={reference_versions!r}, candidate={candidate_versions!r}")
        if reference_env != candidate_env:
            print("NOTICE environment.json differs; version, source and commit metadata are informational.")
    except (OSError, ValueError, TypeError, OverflowError) as error:
        comparator.fail(f"cannot compare bundles: {error}")

    if comparator.failures:
        print(f"FAIL {comparator.failures} comparison issue(s)")
        return 1
    mode = "exact" if args.exact else f"rtol={args.rtol:g}, atol={args.atol:g}"
    print(f"PASS bundles match ({mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
