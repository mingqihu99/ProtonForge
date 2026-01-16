#!/usr/bin/env python3
"""Generate .npz files from concise specs.

Spec format examples:
  s32[2048,16,128](1)   -> int32 array shape (2048,16,128) filled with 1
  f32[10](0.5)          -> float32 vector of length 10 filled with 0.5
  f32[10](-1.5,1.5)     -> float32 random array in range [-1.5, 1.5]
  u8[64,64]             -> uint8 random array shape (64,64)

CLI:
  python generate_npz.py SPEC [SPEC ...] [-o data.npz] [--names name1,name2] [--seed N]
  python generate_npz.py --hlo module.txt [--ranges RANGES] [-o data.npz] [--name_pattern ir_args_name]

If --names is omitted, variable names default to input_0, input_1, ...
or extracted HLO names if --name_pattern=ir_args_name is used.
"""

import argparse
import re
import ast
import numpy as np
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


def extract_hlo_specs(hlo_module_path: str):
  """Extracts shape specs and names from the ENTRY line of an HLO module file."""
  with open(hlo_module_path, "r") as f:
    content = f.read()

  # Find the ENTRY computation block
  # Format: ENTRY name (Arg_0: f32[...], Arg_1: ...) -> ...
  entry_match = re.search(r"ENTRY\s+%?[\w.]+\s*\((.*?)\)\s*->", content)
  if not entry_match:
    return [], []

  params_str = entry_match.group(1)
  # Find all "name: shape" pairs
  matches = re.findall(r"([\w.]+):\s+([a-z0-9]+\[[0-9, ]+\])", params_str)
  if matches:
    names = [m[0] for m in matches]
    specs = [m[1] for m in matches]
    return specs, names
  else:
    # Fallback to just shapes if names are missing
    specs = re.findall(r"[a-z0-9]+\[[0-9, ]+\]", params_str)
    return specs, []


def merge_range_specs(base_specs: list[str], range_specs: Optional[str]) -> list[str]:
  """Merges base shape specs with range/constant specs."""
  if not range_specs:
    return base_specs

  # Work on a copy
  final_specs = list(base_specs)

  # 1) Try indexed format: "0:(-1,1) 3:(-3,3)"
  indexed_matches = re.findall(r"(\d+):\((.*?)\)", range_specs)
  if indexed_matches:
    for idx_str, range_val in indexed_matches:
      idx = int(idx_str)
      if 0 <= idx < len(final_specs):
        if range_val.strip():
          final_specs[idx] = f"{final_specs[idx]}({range_val})"
    return final_specs

  # 2) Try sequential format: "(-1,1) () (-3,3)"
  sequential_matches = re.findall(r"\((.*?)\)", range_specs)
  if sequential_matches:
    for i, range_val in enumerate(sequential_matches):
      if i < len(final_specs):
        if range_val.strip():
          final_specs[i] = f"{final_specs[i]}({range_val})"
    return final_specs

  return final_specs


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


def generate_npz(specs: Optional[Tuple[str, ...]] = None,
                 output_path: str = "data.npz",
                 names: Optional[Tuple[str, ...]] = None,
                 seed: Optional[int] = None,
                 hlo_module_path: Optional[str] = None,
                 range_specs: Optional[str] = None,
                 name_pattern: str = "input_x"):
  """Generate and save NPZ file from specs or HLO module.

  Args:
    specs: Tuple of spec strings.
    output_path: Path to save the .npz file.
    names: Optional tuple of variable names.
    seed: Optional random seed.
    hlo_module_path: Optional path to HLO module to extract shapes from.
    range_specs: Optional range specifications to merge with HLO shapes.
    name_pattern: Name pattern to use if 'names' is not provided.
                  'input_x' (default) or 'ir_args_name'.
  """
  hlo_names = []
  if hlo_module_path:
    base_specs, hlo_names = extract_hlo_specs(hlo_module_path)
    if not base_specs:
      raise ValueError(f"No input shapes found in HLO module: {hlo_module_path}")

    final_specs = merge_range_specs(base_specs, range_specs)
    specs = tuple(final_specs)

  if specs is None:
    raise ValueError("Must provide either 'specs' or 'hlo_module_path'")

  if names is None:
    if name_pattern == "ir_args_name" and hlo_names:
      names = tuple(hlo_names)
    else:
      names = tuple(f"input_{i}" for i in range(len(specs)))

  if len(names) != len(specs):
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

    varname = names[i]
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
  parser.add_argument("specs", nargs="*",
                      help="One or more specs like s32[10,20](1) or f32[5]")
  parser.add_argument(
      "--hlo", help="Path to HLO module to extract shapes of inputs")
  parser.add_argument(
      "--ranges", help="Range specs for HLO inputs (sequential or id:range)")
  parser.add_argument("-o", "--out", default="data.npz",
                      help="Output .npz path (default: data.npz)")
  parser.add_argument(
      "--names", help="Comma-separated variable names (optional)")
  parser.add_argument(
      "--name_pattern", choices=["input_x", "ir_args_name"], default="input_x",
      help="Variable naming pattern if --names is not provided (default: input_x)")
  parser.add_argument("--seed", type=int, default=None,
                      help="Random seed (optional)")

  args = parser.parse_args(argv)

  names = None
  if args.names:
    names = tuple(n.strip() for n in args.names.split(",") if n.strip())

  try:
    generate_npz(tuple(args.specs) if args.specs else None,
                 args.out,
                 names=names,
                 seed=args.seed,
                 hlo_module_path=args.hlo,
                 range_specs=args.ranges,
                 name_pattern=args.name_pattern)
  except Exception as e:
    raise SystemExit(f"Error: {e}")


if __name__ == "__main__":
  main()
