#!/usr/bin/env python3
"""Generate .npz files from concise specs.

Spec format examples:
  s32[2048,16,128](1)   -> int32 array shape (2048,16,128) filled with 1
  f32[10](0.5)          -> float32 vector of length 10 filled with 0.5
  f32[10](-1.5,1.5)     -> float32 random array in range [-1.5, 1.5]
  u8[64,64]             -> uint8 random array shape (64,64)

CLI:
  python generate_npz.py out.npz SPEC [SPEC ...] [--names name1,name2] [--seed N]

If --names is omitted, variable names default to arr0, arr1, ...
"""

import argparse
import re
import ast
import numpy as np
import sys
from typing import Tuple, Optional, Any


DTYPE_MAP = {
    "s8": np.int8,
    "u8": np.uint8,
    "s16": np.int16,
    "u16": np.uint16,
    "s32": np.int32,
    "u32": np.uint32,
    "s64": np.int64,
    "u64": np.uint64,
    "f16": np.float16,
    "f32": np.float32,
    "f64": np.float64,
    "b": np.bool_,
    "bool": np.bool_,
}

SPEC_RE = re.compile(
    r"^(?P<dtype>[a-z0-9]+)\[(?P<shape>[0-9,\s]+)\](?:\((?P<const>.*)\))?$",
    re.IGNORECASE)


def parse_spec(spec: str):
  m = SPEC_RE.match(spec.strip())
  if not m:
    raise ValueError(f"Invalid spec: '{spec}'")

  dtype_key = m.group("dtype").lower()
  if dtype_key not in DTYPE_MAP:
    raise ValueError(
        f"Unknown dtype '{dtype_key}' in spec '{spec}'. Supported: {', '.join(DTYPE_MAP.keys())}")
  dtype = DTYPE_MAP[dtype_key]

  shape_str = m.group("shape")
  shape = tuple(int(x.strip()) for x in shape_str.split(",") if x.strip())
  if any(s < 0 for s in shape):
    raise ValueError(f"Negative shape dimension in spec '{spec}'")

  const_raw = m.group("const")
  if const_raw is None:
    const_val = None
  else:
    # try to safely evaluate as Python literal (int, float, tuple, etc.)
    try:
      const_val = ast.literal_eval(const_raw)
    except Exception:
      # fall back to string
      const_val = const_raw

  return dtype, shape, const_val


def random_array(dtype: np.dtype, shape: Tuple[int, ...], rng: np.random.Generator,
                 low: Optional[float] = None, high: Optional[float] = None):
  kind = np.dtype(dtype).kind
  if kind in ("i", "u"):
    info = np.iinfo(dtype)
    if low is None:
      low = max(info.min, -1000)
    if high is None:
      high = min(info.max, 1000)

    # Clip to dtype range
    l_val = int(max(info.min, low))
    h_val = int(min(info.max, high))

    if l_val > h_val:
      raise ValueError(f"Invalid range [{low}, {high}] for dtype {dtype}")

    return rng.integers(l_val, h_val, size=shape, dtype=dtype, endpoint=True)
  elif kind == "f":
    if low is not None and high is not None:
      if low > high:
        raise ValueError(f"Invalid range [{low}, {high}] for float generation")
      return rng.uniform(low, high, size=shape).astype(dtype)
    return rng.standard_normal(size=shape).astype(dtype)
  elif kind == "b":
    return rng.integers(0, 2, size=shape, dtype=np.uint8).astype(bool)
  else:
    raise ValueError(f"Unsupported dtype kind '{kind}'")


def cast_const_to_dtype(val: Any, dtype: np.dtype) -> Any:
  # handle booleans
  if np.dtype(dtype).kind == "b":
    if isinstance(val, str):
      vlow = val.strip().lower()
      if vlow in ("1", "true", "t", "yes", "y"):
        return True
      if vlow in ("0", "false", "f", "no", "n"):
        return False
      raise ValueError(f"Cannot cast '{val}' to bool")
    return bool(val)

  # numeric cast
  try:
    return np.array(val, dtype=dtype).item()
  except Exception:
    # last resort: try float then cast
    return dtype.type(float(val))


def generate_npz(specs: Tuple[str, ...], output_path: str,
                 names: Optional[Tuple[str, ...]] = None, seed: Optional[int] = None):
  """Generate and save NPZ file from specs.

  Args:
    specs: Tuple of spec strings.
    output_path: Path to save the .npz file.
    names: Optional tuple of variable names.
    seed: Optional random seed.
  """
  if names is not None and len(names) != len(specs):
    raise ValueError(
        f"Number of names ({len(names)}) must match number of specs ({len(specs)})")

  rng = np.random.default_rng(seed)

  arrays = {}
  for i, spec in enumerate(specs):
    try:
      dtype, shape, const_val = parse_spec(spec)
    except Exception as e:
      raise ValueError(f"Error parsing spec #{i} '{spec}': {e}")

    if const_val is None:
      arr = random_array(dtype, shape, rng)
    elif isinstance(const_val, (tuple, list)) and len(const_val) == 2:
      # Use as range (min, max)
      try:
        arr = random_array(dtype, shape, rng, low=const_val[0], high=const_val[1])
      except Exception as e:
        raise ValueError(
            f"Error generating random range {const_val} for spec '{spec}': {e}")
    else:
      # Use as constant
      try:
        cast_val = cast_const_to_dtype(const_val, dtype)
      except Exception as e:
        raise ValueError(f"Cannot cast constant '{const_val}' to dtype {dtype}: {e}")
      arr = np.full(shape, cast_val, dtype=dtype)

    varname = names[i] if names is not None else f"arr{i}"
    arrays[varname] = arr
    print(f"Prepared `{varname}`: spec={spec}, dtype={arr.dtype}, shape={arr.shape}")

  try:
    np.savez(output_path, **arrays)
    print(f"Saved {len(arrays)} arrays to '{output_path}'")
  except Exception as e:
    raise RuntimeError(f"Failed saving to '{output_path}': {e}")


def main(argv=None):
  parser = argparse.ArgumentParser(
      description="Generate .npz files from concise specs")
  parser.add_argument("out", help="Output .npz path")
  parser.add_argument("specs", nargs="+",
                      help="One or more specs like s32[10,20](1) or f32[5]")
  parser.add_argument(
      "--names", help="Comma-separated variable names (optional)")
  parser.add_argument("--seed", type=int, default=None,
                      help="Random seed (optional)")

  args = parser.parse_args(argv)

  names = None
  if args.names:
    names = tuple(n.strip() for n in args.names.split(",") if n.strip())

  try:
    generate_npz(tuple(args.specs), args.out, names=names, seed=args.seed)
  except Exception as e:
    raise SystemExit(f"Error: {e}")


if __name__ == "__main__":
  main()
