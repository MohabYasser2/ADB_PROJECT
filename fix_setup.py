"""
Fix index filenames and verify setup for evaluation
"""
import os
import shutil

os.chdir(r"c:\Users\mohab\Desktop\Uni\Courses\Year 5 -1st term\ADB\proj\vecdp")

print("=" * 60)
print("Fixing index filenames for evaluation")
print("=" * 60)

# Rename .pkl files to .csv (evaluator expects .csv extension)
renames = [
    ("index_1m.pkl", "saved_db_1m.csv"),
    ("index_20m.pkl", "saved_db_20m.csv")
]

for old_name, new_name in renames:
    if os.path.exists(old_name):
        if os.path.exists(new_name):
            os.remove(new_name)
        shutil.copy2(old_name, new_name)
        size_mb = os.path.getsize(new_name) / (1024 * 1024)
        print(f"✓ Created {new_name} ({size_mb:.2f} MB)")
    else:
        print(f"✗ {old_name} not found!")

print("\n" + "=" * 60)
print("Checking for required files")
print("=" * 60)

required_files = [
    "vec_db.py",
    "requirements.txt",
    "saved_db_1m.csv",
    "saved_db_20m.csv"
]

all_good = True
for file in required_files:
    if os.path.exists(file):
        print(f"✓ {file}")
    else:
        print(f"✗ {file} MISSING!")
        all_good = False

if all_good:
    print("\n✅ All required files present!")
    print("\nNext steps:")
    print("1. Create a GitHub repository")
    print("2. Upload these files: vec_db.py, requirements.txt")
    print("3. Zip the indexes: run zip_indexes.py")
    print("4. Upload zipped indexes to Google Drive")
else:
    print("\n⚠️ Some files are missing - fix before proceeding")
