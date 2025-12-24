#!/usr/bin/env python3
"""Print shapes and first N elements of arrays in .npz or .npy files.

Usage:
    python print_npz.py file.npz
    python print_npz.py file.npz --max 500
    python print_npz.py file.npy
    python print_npz.py file.npy --max 500
"""

import argparse
import textwrap
import numpy as np
import os


def main():
  parser = argparse.ArgumentParser(
      description="Print shapes and first N elements of arrays in .npz or .npy files."
  )
  parser.add_argument("file", help="Path to .npz or .npy file")
  parser.add_argument("--max", "-m", type=int, default=1000,
                      help="Max elements to print per array (default 1000)")
  args = parser.parse_args()

  file_path = args.file
  _, ext = os.path.splitext(file_path)
  ext = ext.lower()

  if ext == ".npy":
    load_and_print_npy(file_path, args.max)
  elif ext == ".npz":
    load_and_print_npz(file_path, args.max)
  else:
    print(f"Unsupported file type: '{ext}'. Expected .npy or .npz")


def format_elem(x, dtype):
  try:
    k = np.dtype(dtype).kind
    if k in ("i", "u"):
      return str(int(x))
    if k == "f":
      return repr(float(x))
    if k == "b":
      return str(bool(x))
    # for string/bytes and other dtypes, fall back to repr
    return repr(x.item() if hasattr(x, "item") else x)
  except Exception:
    return repr(x)


def print_array_preview(name, arr, max_elements):
  """Print array name, dtype, shape, and first N elements."""
  print(f"Array `{name}`: dtype={arr.dtype}, shape={arr.shape}")

  flat = arr.ravel()
  n = min(max_elements, flat.size)

  # Prepare printable representations for the sample elements
  sample = [format_elem(x, arr.dtype) for x in flat[:n]]
  s = ", ".join(sample)
  if flat.size > n:
    s += ", ..."

  # Wrap the output for readability
  print("First {} elements:".format(n))
  print(textwrap.fill(s, width=120))
  print()


def load_and_print_npy(file_path, max_elements):
  """Load and print a single .npy file."""
  try:
    arr = np.load(file_path, allow_pickle=True)
    name = os.path.basename(file_path)
    print_array_preview(name, arr, max_elements)
  except Exception as e:
    print(f"Error loading `{file_path}`: {e}")


def load_and_print_npz(file_path, max_elements):
  """Load and print all arrays in a .npz file."""
  try:
    with np.load(file_path, allow_pickle=True) as data:
      names = data.files
      if not names:
        print(f"No arrays found in `{file_path}`")
        return

      for name in names:
        arr = data[name]
        print_array_preview(name, arr, max_elements)

  except Exception as e:
    print(f"Error loading `{file_path}`: {e}")


if __name__ == "__main__":
  main()
