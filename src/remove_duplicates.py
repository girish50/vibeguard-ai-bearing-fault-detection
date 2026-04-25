import os
import hashlib

# 🔹 CHANGE THIS PATH
folder_path = r"E:\dataset"

hash_dict = {}   # {hash: [file1, file2, file3]}
total_files = 0

print("🔍 Scanning files...\n")

for root, dirs, files in os.walk(folder_path):
    for file in files:
        total_files += 1
        file_path = os.path.join(root, file)

        try:
            with open(file_path, 'rb') as f:
                hasher = hashlib.md5()
                while chunk := f.read(8192):
                    hasher.update(chunk)
                file_hash = hasher.hexdigest()

            # 🔹 Store ALL files with same hash
            if file_hash in hash_dict:
                hash_dict[file_hash].append(file_path)
            else:
                hash_dict[file_hash] = [file_path]

        except Exception as e:
            print(f"Error reading {file}: {e}")

print("✅ Total files scanned:", total_files)

# 🔹 Find duplicates (groups)
duplicate_groups = []
for file_list in hash_dict.values():
    if len(file_list) > 1:
        duplicate_groups.append(file_list)

print("⚠️ Total duplicate groups:", len(duplicate_groups))

# 🔹 Show detailed duplicates
print("\n📂 Duplicate Groups (which file matches which):\n")

for i, group in enumerate(duplicate_groups[:10], start=1):
    print(f"Group {i} (Repeated {len(group)} times):")
    for file in group:
        print("   ", file)
    print("-" * 50)

# 🔹 Count total duplicate files
total_duplicates = sum(len(group) - 1 for group in duplicate_groups)
print("\n⚠️ Total duplicate files (extra copies):", total_duplicates)

# 🔹 Save report to file (VERY USEFUL)
with open("duplicate_report.txt", "w") as report:
    for i, group in enumerate(duplicate_groups, start=1):
        report.write(f"Group {i} (Repeated {len(group)} times):\n")
        for file in group:
            report.write(f"   {file}\n")
        report.write("-" * 50 + "\n")

print("\n📄 Report saved as duplicate_report.txt")

# 🔹 Ask before deleting (keeps one file, deletes others)
delete = input("\nDo you want to delete duplicates (keeping one copy)? (yes/no): ")

if delete.lower() == "yes":
    for group in duplicate_groups:
        # keep first file, delete rest
        for file in group[1:]:
            os.remove(file)
    print("🗑️ Duplicates deleted (one copy kept)!")
else:
    print("❌ No files deleted.")