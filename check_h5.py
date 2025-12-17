import h5py
import numpy as np
import os

h5_path = "data.h5"

print("Opening file:", os.path.abspath(h5_path))
print("File size (MB):", os.path.getsize(h5_path)/1024/1024)

with h5py.File(h5_path, "r") as f:
    print("\n=== FILE METADATA ===")
    for k, v in f.attrs.items():
        print(f"{k}: {v}")

    print("\n=== SEGMENTS TABLE (raw) ===")
    segs_raw = f["segments_table"][:]
    print(segs_raw, "shape:", segs_raw.shape)

    # Normalize to shape (nSeg, 2)
    segs = np.array(segs_raw).squeeze()
    if segs.ndim == 1 and segs.size == 2:
        segs = segs.reshape(1, 2)
    print("\n=== SEGMENTS TABLE (normalized) ===")
    print(segs, "shape:", segs.shape)
    print("Number of segments:", segs.shape[0])

    seg_name = list(f["segments"].keys())[0]
    seg = f["segments"][seg_name]

    I = seg["current"][:].squeeze()
    V = seg["voltage"][:].squeeze()

    print(f"\n=== FIRST SEGMENT: {seg_name} ===")
    print("current shape:", I.shape, I.dtype)
    print("voltage shape:", V.shape, V.dtype)