# NPZ Tools

A collection of utilities for generating and comparing NumPy .npz/.npy files.

## Synopsis

### compare_npz.py
```python compare_npz.py GOLDEN_FILE TEST_FILE [OPTIONS]```

### generate_npz.py
```python generate_npz.py OUTPUT_FILE SPEC [SPEC ...] [OPTIONS]```

## Description

This tool contains Python scripts for working with NumPy array files (.npz and .npy formats).

- **compare_npz.py**: Compares two .npz/.npy files (golden reference first) and reports errors including mean absolute error (MAE), mean relative error (MRE), and pass rate using configurable tolerances.

- **generate_npz.py**: Generates .npz files from concise specifications, supporting various data types and shapes with optional constant values or random initialization.

## Options

### compare_npz.py options
- `-m, --max N` Number of elements to print per array (default 20)
- `--eps FLOAT` Epsilon added to denominator for relative error (default 1e-12)
- `--atol FLOAT` Absolute tolerance for near check (default 1e-8, auto-adjusted for fp16/fp32)
- `--rtol FLOAT` Relative tolerance for near check (default 1e-5, auto-adjusted for fp16/fp32)
- `-h, --help` Show help message and exit

### generate_npz.py options
- `--names NAME1,NAME2,...` Variable names for arrays (default: arr0, arr1, ...)
- `--seed N` Random seed for reproducible random arrays
- `-h, --help` Show help message and exit

## Examples

### Compare two .npz files
```
python compare_npz.py golden.npz test.npz  # use default atol, rtol
```

### Compare with custom tolerances
```
python compare_npz.py golden.npz test.npz --atol 1e-5 --rtol 1e-5
OR
python compare_npz.py golden.npz test.npz --atol 1e-5  # means rtol is zero, only cares atol
```

#### Sample Output
Output:
```
=== Array #0 ===
Name: arr0
Shape: (100,) | dtype(golden)=float32 dtype(test)=float32

First 20 elements (golden -> test):
  0: 1.000000 -> 1.000001    1: 2.000000 -> 2.000002    2: 3.000000 -> 3.000003
  3: 4.000000 -> 4.000004    4: 5.000000 -> 5.000005    5: 6.000000 -> 6.000006
  6: 7.000000 -> 7.000007    7: 8.000000 -> 8.000008    8: 9.000000 -> 9.000009
  9: 10.000000 -> 10.000010  10: 11.000000 -> 11.000011  11: 12.000000 -> 12.000012
  12: 13.000000 -> 13.000013  13: 14.000000 -> 14.000014  14: 15.000000 -> 15.000015
  15: 16.000000 -> 16.000016  16: 17.000000 -> 17.000017  17: 18.000000 -> 18.000018
  18: 19.000000 -> 19.000019  19: 20.000000 -> 20.000021

Top 5 largest absolute errors:
Index 99 99: 100.000000 -> 100.000099 | abs_err=0.000099 | rel_err=0.000001
Index 98 98: 99.000000 -> 99.000098 | abs_err=0.000098 | rel_err=0.000001
Index 97 97: 98.000000 -> 98.000097 | abs_err=0.000097 | rel_err=0.000001
Index 96 96: 97.000000 -> 97.000096 | abs_err=0.000096 | rel_err=0.000001
Index 95 95: 96.000000 -> 96.000095 | abs_err=0.000095 | rel_err=0.000001

Mean absolute error (MAE): 0.00005
Max absolute error: 0.000099
Mean relative error (MRE): 5e-07
Max relative error: 1e-06
Pass rate: PASS (using |test - golden| < 1e-05 + 1e-05 * |golden|)
```

### Library usage in python
```
from compare_npz import allclose
allclose("golden.npz", "test.npz", 1e-5)
```

### Generate a .npz file with multiple arrays
```
python generate_npz.py output.npz f32[1024,512](0.0) s32[256](1) u8[64,64]
```

### Generate a .npz file with given range
```
python generate_npz.py output.npz f32[10](-1.5,1.5)
```

### Generate with custom names
```
python generate_npz.py data.npz f32[100](0.5) f16[50] --names input_0,input_1
```

### Library usage in python
```
from generate_npz import generate_npz
specs = ("f32[1024,512](-1,1)", "s32[256](1)", "u8[64,64]")
output_path = "test_output.npz"
names = ("a", "b", "c")
generate_npz(specs, output_path, names=names)
```
