#!/usr/bin/env python3
"""Compare two .npz/.npy files (golden reference first) and report errors.

For each array (matched in order) the script prints:
- index and names (if .npz entry names available)
- dtype and shape
- first N elements from golden and test (side-by-side)
- mean absolute error (MAE) and mean relative error (MRE)
- pass rate using |a-b| < atol + rtol*|b| (PASS if 100%)

Usage:
    python compare_npz.py golden.npz test.npz
    python compare_npz.py golden.npz test.npz --max 20
    python compare_npz.py golden.npy test.npy

The relative error uses: abs(diff) / (abs(golden) + eps) with eps=1e-12 by default.
"""

import argparse
import os
import numpy as np
import textwrap
from typing import List, Tuple, Dict, Any, Optional


def load_arrays(path: str) -> List[Tuple[str, np.ndarray]]:
  """Load arrays from .npz or .npy. Return list of (name, array) in file order."""
  base = os.path.basename(path)
  _, ext = os.path.splitext(path)
  ext = ext.lower()
  try:
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.ndarray):
      if ext != ".npy":
        raise RuntimeError(
            f"File '{path}' has .npy content but extension is '{ext}'; expected .npy")
      return [(base, loaded)]
    elif hasattr(loaded, "files"):
      # NpzFile
      if ext != ".npz":
        loaded.close()
        raise RuntimeError(
            f"File '{path}' has .npz content but extension is '{ext}'; expected .npz")
      names = loaded.files
      result = [(name, loaded[name]) for name in names]
      loaded.close()
      return result
    else:
      raise RuntimeError(f"Unexpected loaded type for '{path}': {type(loaded)}")
  except Exception as e:
    raise RuntimeError(f"Failed to load '{path}': {e}")


def format_elem(x, dtype):
  try:
    k = np.dtype(dtype).kind
    if k in ("i", "u"):
      return str(int(x))
    if k == "f":
      v = float(x)
      if not np.isfinite(v):
        return repr(v)
      # if value is effectively rounded to 6 decimals, show 6 decimals
      if abs(v - round(v, 6)) < 1e-12:
        return f"{v:.6f}"
      # otherwise show more precision (up to 12 significant digits)
      return f"{v:.12g}"
    if k == "b":
      return str(bool(x))
    return repr(x.item() if hasattr(x, "item") else x)
  except Exception:
    return repr(x)


def format_number(x: float) -> str:
  """Format a numeric float to show at least 6 decimals, allow more when needed."""
  try:
    v = float(x)
  except Exception:
    return repr(x)
  if not np.isfinite(v):
    return repr(v)
  if abs(v - round(v, 6)) < 1e-12:
    return f"{v:.6f}"
  return f"{v:.12g}"


def near(
    a: float,
    b: float,
    atol: Optional[float] = None,
        rtol: Optional[float] = None) -> bool:
  """Check if a is close to b using |a - b| < atol + rtol * |b|."""
  if atol is None:
    atol = 1e-8
  if rtol is None:
    rtol = 1e-5
  return abs(a - b) < atol + rtol * abs(b)


def compare_arrays(golden: np.ndarray,
                   test: np.ndarray,
                   max_print: int = 20,
                   eps: float = 1e-12,
                   atol: Optional[float] = None,
                   rtol: Optional[float] = None) -> Dict[str,
                                                         Any]:
  """Compare two arrays and return a summary dict."""
  # check shape
  if golden.shape != test.shape:
    return {
        "ok": False,
        "reason": f"shape_mismatch: golden{golden.shape} vs test{test.shape}"}

  # Set default atol/rtol based on dtype if not specified by user
  if atol is None and rtol is None:
    if golden.dtype == np.float16:
      atol = 1e-3
      rtol = 1e-3
    elif golden.dtype == np.float32:
      atol = 1e-5
      rtol = 1e-5
    else:
      # For other dtypes, use general defaults
      atol = 1e-8
      rtol = 1e-5
  elif atol is None:
    atol = 0
  elif rtol is None:
    rtol = 0
  # If both are specified, use the specified values regardless of dtype

  info: Dict[str, Any] = {}
  info["dtype_golden"] = golden.dtype
  info["dtype_test"] = test.dtype
  info["shape"] = golden.shape

  # flatten for printing/computation
  g_flat = golden.ravel()
  t_flat = test.ravel()
  n = g_flat.size

  # prepare preview
  m = min(max_print, n)
  preview = []
  for i in range(m):
    preview.append(
        (format_elem(g_flat[i], golden.dtype), format_elem(t_flat[i], test.dtype)))

  info["preview"] = preview
  info["preview_count"] = m

  # numeric comparisons
  kind_g = np.dtype(golden.dtype).kind
  kind_t = np.dtype(test.dtype).kind

  numeric_kinds = set(["f", "i", "u", "b"])

  if (kind_g in numeric_kinds) and (kind_t in numeric_kinds):
    # compute using float64 for safety
    g_float = g_flat.astype(np.float64)
    t_float = t_flat.astype(np.float64)
    diff = t_float - g_float
    abs_err = np.abs(diff)
    mae = float(np.mean(abs_err)) if abs_err.size else float("nan")
    max_abs = float(np.max(abs_err)) if abs_err.size else float("nan")
    # relative: |diff| / (|golden| + eps)
    denom = np.abs(g_float) + eps
    rel = abs_err / denom
    mre = float(np.mean(rel)) if rel.size else float("nan")
    max_rel = float(np.max(rel)) if rel.size else float("nan")

    info.update({"ok": True, "mae": mae, "max_abs": max_abs,
                "mre": mre, "max_rel": max_rel})

    # top-k largest absolute errors (default 5)
    k = min(5, n)
    if n > 0:
      idxs = np.argsort(-abs_err)[:k]
      topk = []
      for idx in idxs:
        multi_idx = tuple(int(x)
                          for x in np.unravel_index(int(idx), golden.shape))
        g_s = format_elem(g_flat[int(idx)], golden.dtype)
        t_s = format_elem(t_flat[int(idx)], test.dtype)
        topk.append({"flat_index": int(idx),
                     "index": multi_idx,
                     "golden": g_s,
                     "test": t_s,
                     "abs_err": float(abs_err[int(idx)]),
                     "rel_err": float(rel[int(idx)])})
      info["topk"] = topk
      # compute top-k relative errors (ignore non-finite relative errors)
      rel_mask = np.isfinite(rel)
      if np.any(rel_mask):
        rel_for_sort = np.where(rel_mask, rel, -np.inf)
        ridxs = np.argsort(-rel_for_sort)[:k]
        topk_rel = []
        for idx in ridxs:
          if not rel_mask[int(idx)]:
            continue
          multi_idx = tuple(
              int(x) for x in np.unravel_index(int(idx), golden.shape))
          g_s = format_elem(g_flat[int(idx)], golden.dtype)
          t_s = format_elem(t_flat[int(idx)], test.dtype)
          topk_rel.append({"flat_index": int(idx),
                           "index": multi_idx,
                           "golden": g_s,
                           "test": t_s,
                           "abs_err": float(abs_err[int(idx)]),
                           "rel_err": float(rel[int(idx)])})
        info["topk_rel"] = topk_rel
      else:
        info["topk_rel"] = []

    # compute pass rate using near function
    pass_count = 0
    for i in range(n):
      if near(float(t_flat[i]), float(g_flat[i]), atol, rtol):
        pass_count += 1
    pass_rate = (pass_count / n) * 100.0 if n > 0 else 100.0
    info["pass_rate"] = pass_rate
    info["atol"] = atol
    info["rtol"] = rtol
  else:
    # non-numeric comparison: not supported
    raise RuntimeError(
        f"Non-numeric dtypes not supported: golden={golden.dtype}, test={test.dtype}")

  return info


def print_comparison(index: int, name_g: str, name_t: str, result: Dict[str, Any]):
  print(f"=== Array #{index} ===")
  if name_g == name_t:
    print(f"Name: {name_g}")
  else:
    print(f"Golden name: {name_g} | Test name: {name_t}")

  if not result.get("ok", False):
    print(f"Comparison failed: {result.get('reason')}")
    print()
    return

  print(
      f"Shape: {result['shape']} | dtype(golden)={result['dtype_golden']} dtype(test)={result['dtype_test']}")

  # print preview side-by-side
  m = result.get("preview_count", 0)
  if m > 0:
    print("\nFirst {} elements (golden -> test):".format(m))
    # produce aligned columns
    lines = []
    for i, (g_val, t_val) in enumerate(result["preview"]):
      lines.append(f"{i:3d}: {g_val} -> {t_val}")
    print(textwrap.fill("\n".join(lines), width=120))

  # print numeric summary if present
  if "mae" in result:
    # print top-k largest absolute errors if available
    if "topk" in result and result["topk"]:
      print()
      print("Top {} largest absolute errors:".format(
          len(result["topk"])))
      for item in result["topk"]:
        idx = item["index"]
        flat = item["flat_index"]
        abs_s = format_number(item["abs_err"])
        rel_s = format_number(item["rel_err"])
        print(
            f"Index {flat} {idx}: {item['golden']} -> {item['test']} | abs_err={abs_s} | rel_err={rel_s}")
    # print top-k largest relative errors if available
    if "topk_rel" in result and result["topk_rel"]:
      print()
      print("Top {} largest relative errors:".format(
          len(result["topk_rel"])))
      for item in result["topk_rel"]:
        idx = item["index"]
        flat = item["flat_index"]
        abs_s = format_number(item["abs_err"])
        rel_s = format_number(item["rel_err"])
        print(
            f"Index {flat} {idx}: {item['golden']} -> {item['test']} | abs_err={abs_s} | rel_err={rel_s}")
    print()
    print(f"Mean absolute error (MAE): {result['mae']}")
    print(f"Max absolute error: {result['max_abs']}")
    print(f"Mean relative error (MRE): {result['mre']}")
    print(f"Max relative error: {result['max_rel']}")
    # print pass rate
    pass_rate = result.get("pass_rate", 100.0)
    atol = result.get("atol", 1e-8)
    rtol = result.get("rtol", 1e-5)
    if pass_rate == 100.0:
      print(
          f"Pass rate: PASS (using |test - golden| < {atol} + {rtol} * |golden|)")
    else:
      print(
          f"Pass rate: {pass_rate:.2f}% (using |test - golden| < {atol} + {rtol} * |golden|)")

  print()


def allclose(
        golden_path: str,
        test_path: str,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        eps: float = 1e-12) -> bool:
  """Check if all arrays in golden and test files are close within tolerances.

  Args:
      golden_path: Path to the golden/reference .npz or .npy file.
      test_path: Path to the test .npz or .npy file.
      atol: Absolute tolerance. If None, uses defaults based on dtype.
      rtol: Relative tolerance. If None, uses defaults based on dtype.
      eps: Epsilon for relative error calculation.

  Returns:
      True if all arrays pass the closeness check.

  Raises:
      RuntimeError: If files cannot be loaded or arrays cannot be compared.
      ValueError: If the number of arrays in files do not match.
  """
  try:
    gold_list = load_arrays(golden_path)
  except Exception as e:
    raise RuntimeError(f"Failed to load golden file '{golden_path}': {e}")

  try:
    test_list = load_arrays(test_path)
  except Exception as e:
    raise RuntimeError(f"Failed to load test file '{test_path}': {e}")

  if len(gold_list) != len(test_list):
    raise ValueError(
        f"Number of arrays mismatch: golden has {len(gold_list)}, test has {len(test_list)}")

  for i in range(len(gold_list)):
    name_g, arr_g = gold_list[i]
    name_t, arr_t = test_list[i]
    try:
      res = compare_arrays(arr_g, arr_t, max_print=0, eps=eps, atol=atol, rtol=rtol)
    except Exception as e:
      raise RuntimeError(f"Failed to compare array {i} ({name_g} vs {name_t}): {e}")
    if not res.get("ok", False):
      raise RuntimeError(f"Array {i} comparison failed: {res.get('reason')}")
    if res.get("pass_rate", 0) != 100.0:
      return False

  return True


def main():
  parser = argparse.ArgumentParser(
      description="Compare two .npz/.npy files (golden first)")
  parser.add_argument("golden", help="Golden/reference .npz or .npy file")
  parser.add_argument("test", help="Test .npz or .npy file to compare")
  parser.add_argument("--max", "-m", type=int, default=20,
                      help="Number of elements to print per array (default 20)")
  parser.add_argument("--eps", type=float, default=1e-12,
                      help="Epsilon added to denominator for relative error")
  parser.add_argument(
      "--atol",
      type=float,
      default=None,
      help="Absolute tolerance for near check (default: 1e-3 for fp16, 1e-5 for fp32, 1e-8 for others)")
  parser.add_argument(
      "--rtol",
      type=float,
      default=None,
      help="Relative tolerance for near check (default: 1e-3 for fp16, 1e-5 for fp32, 1e-5 for others)")

  args = parser.parse_args()

  try:
    gold_list = load_arrays(args.golden)
  except Exception as e:
    print(f"Error loading golden file: {e}")
    return

  try:
    test_list = load_arrays(args.test)
  except Exception as e:
    print(f"Error loading test file: {e}")
    return

  count = min(len(gold_list), len(test_list))
  if len(gold_list) != len(test_list):
    print(
        f"Warning: different number of arrays: golden={len(gold_list)} test={len(test_list)}. Comparing first {count} arrays.")

  for i in range(count):
    name_g, arr_g = gold_list[i]
    name_t, arr_t = test_list[i]
    try:
      res = compare_arrays(
          arr_g,
          arr_t,
          max_print=args.max,
          eps=args.eps,
          atol=args.atol,
          rtol=args.rtol)
    except Exception as e:
      res = {"ok": False, "reason": f"exception during compare: {e}"}
    print_comparison(i, name_g, name_t, res)

  print(f"Compared {count} array(s)")


if __name__ == "__main__":
  main()
