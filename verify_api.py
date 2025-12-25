import numpy as np
from generate_npz import generate_npz
import os

def test_generate_npz():
    output_path = "test_output.npz"
    specs = ("s32[2,2](1)", "f32[3](0.5)")
    names = ("a", "b")
    
    if os.path.exists(output_path):
        os.remove(output_path)
        
    print("Testing generate_npz API...")
    generate_npz(specs, output_path, names=names, seed=42)
    
    assert os.path.exists(output_path)
    data = np.load(output_path)
    assert "a" in data
    assert "b" in data
    np.testing.assert_array_equal(data["a"], np.ones((2, 2), dtype=np.int32))
    np.testing.assert_array_equal(data["b"], np.full((3,), 0.5, dtype=np.float32))
    print("API test passed!")
    
    os.remove(output_path)

if __name__ == "__main__":
    test_generate_npz()
