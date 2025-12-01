"""
Build 10M index for evaluation
"""
import os
import sys
import numpy as np

os.chdir(r"c:\Users\mohab\Desktop\Uni\Courses\Year 5 -1st term\ADB\proj\vecdp")
sys.path.insert(0, os.getcwd())

from vec_db import VecDB

# Path to 20M dataset
DATA_20M = r"c:\Users\mohab\Desktop\Uni\Courses\Year 5 -1st term\ADB\proj\OpenSubtitles_en_20M_emb_64.dat"

print("=" * 60)
print("Creating 10M dataset and building index")
print("=" * 60)

# Create 10M subset if needed
if not os.path.exists("OpenSubtitles_en_10M_emb_64.dat"):
    print("\nCreating 10M subset from 20M dataset...")
    DIMENSION = 64
    dtype = 'float32'
    
    source = np.memmap(DATA_20M, dtype=dtype, mode='r', shape=(20_000_000, DIMENSION))
    dest = np.memmap("OpenSubtitles_en_10M_emb_64.dat", dtype=dtype, mode='w+', shape=(10_000_000, DIMENSION))
    
    batch_size = 100000
    for start in range(0, 10_000_000, batch_size):
        end = min(start + batch_size, 10_000_000)
        dest[start:end] = source[start:end]
        if start % 1_000_000 == 0:
            print(f"  Copied {start:,}/10,000,000 rows")
    
    dest.flush()
    del source, dest
    print("✓ 10M dataset created")
else:
    print("✓ 10M dataset already exists")

# Build index
print("\nBuilding 10M index...")
db = VecDB(
    database_file_path="OpenSubtitles_en_10M_emb_64.dat",
    index_file_path="saved_db_10m.csv",
    new_db=False
)
db._build_index()

# Check file size
size_mb = os.path.getsize("saved_db_10m.csv") / (1024 * 1024)
print(f"\n✓ 10M index created: saved_db_10m.csv ({size_mb:.2f} MB)")
print("\n" + "=" * 60)
print("Done! Now you have all 3 indexes ready")
print("=" * 60)
