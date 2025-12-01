"""
Complete setup for evaluation - creates all required files
"""
import os
import sys
import numpy as np

# Paths
VECDP_DIR = r"c:\Users\mohab\Desktop\Uni\Courses\Year 5 -1st term\ADB\proj\vecdp"
PROJ_DIR = r"c:\Users\mohab\Desktop\Uni\Courses\Year 5 -1st term\ADB\proj"

os.chdir(VECDP_DIR)

# Find the 20M dataset
dataset_20m = None
for possible_path in [
    os.path.join(PROJ_DIR, "OpenSubtitles_en_20M_emb_64.dat"),
    os.path.join(VECDP_DIR, "OpenSubtitles_en_20M_emb_64.dat"),
    "OpenSubtitles_en_20M_emb_64.dat"
]:
    if os.path.exists(possible_path):
        dataset_20m = possible_path
        print(f"✓ Found 20M dataset at: {dataset_20m}")
        break

if not dataset_20m:
    print("✗ ERROR: OpenSubtitles_en_20M_emb_64.dat not found!")
    print("Please download it from Google Drive first")
    sys.exit(1)

# Import VecDB
from vec_db import VecDB

DIMENSION = 64
dtype = 'float32'

print("\n" + "="*60)
print("Creating all required files for evaluation")
print("="*60)

# 1. Create 1M dataset if needed
print("\n1. Creating 1M dataset...")
dataset_1m = "OpenSubtitles_en_1M_emb_64.dat"
if not os.path.exists(dataset_1m):
    source = np.memmap(dataset_20m, dtype=dtype, mode='r', shape=(20_000_000, DIMENSION))
    dest = np.memmap(dataset_1m, dtype=dtype, mode='w+', shape=(1_000_000, DIMENSION))
    dest[:] = source[:1_000_000]
    dest.flush()
    print(f"✓ Created {dataset_1m}")
else:
    print(f"✓ {dataset_1m} already exists")

# 2. Create 10M dataset if needed
print("\n2. Creating 10M dataset...")
dataset_10m = "OpenSubtitles_en_10M_emb_64.dat"
if not os.path.exists(dataset_10m):
    source = np.memmap(dataset_20m, dtype=dtype, mode='r', shape=(20_000_000, DIMENSION))
    dest = np.memmap(dataset_10m, dtype=dtype, mode='w+', shape=(10_000_000, DIMENSION))
    
    batch_size = 100000
    for start in range(0, 10_000_000, batch_size):
        end = min(start + batch_size, 10_000_000)
        dest[start:end] = source[start:end]
        if start % 1_000_000 == 0:
            print(f"  Copied {start:,}/10,000,000")
    
    dest.flush()
    print(f"✓ Created {dataset_10m}")
else:
    print(f"✓ {dataset_10m} already exists")

# 3. Build 1M index
print("\n3. Building 1M index...")
if not os.path.exists("saved_db_1m.csv"):
    db_1m = VecDB(dataset_1m, "saved_db_1m.csv", new_db=False)
    db_1m._build_index()
    print(f"✓ Built saved_db_1m.csv ({os.path.getsize('saved_db_1m.csv')/1024/1024:.2f} MB)")
else:
    print(f"✓ saved_db_1m.csv already exists ({os.path.getsize('saved_db_1m.csv')/1024/1024:.2f} MB)")

# 4. Build 10M index
print("\n4. Building 10M index...")
if not os.path.exists("saved_db_10m.csv"):
    db_10m = VecDB(dataset_10m, "saved_db_10m.csv", new_db=False)
    db_10m._build_index()
    print(f"✓ Built saved_db_10m.csv ({os.path.getsize('saved_db_10m.csv')/1024/1024:.2f} MB)")
else:
    print(f"✓ saved_db_10m.csv already exists ({os.path.getsize('saved_db_10m.csv')/1024/1024:.2f} MB)")

# 5. Build 20M index
print("\n5. Building 20M index...")
if not os.path.exists("saved_db_20m.csv"):
    db_20m = VecDB(dataset_20m, "saved_db_20m.csv", new_db=False)
    db_20m._build_index()
    print(f"✓ Built saved_db_20m.csv ({os.path.getsize('saved_db_20m.csv')/1024/1024:.2f} MB)")
else:
    print(f"✓ saved_db_20m.csv already exists ({os.path.getsize('saved_db_20m.csv')/1024/1024:.2f} MB)")

print("\n" + "="*60)
print("✓ ALL FILES READY FOR EVALUATION!")
print("="*60)
print("\nNext steps:")
print("1. Run: python zip_indexes.py")
print("2. Upload zip files to Google Drive")
print("3. Update evaluation notebook with file IDs")
