import os

file_path = "../dataset/2003.10.22.12.06.24"  # use your first file

print("Checking file:", file_path)

# Try reading raw
with open(file_path, "rb") as f:
    data = f.read(200)   # read first 200 bytes

print("\nRAW CONTENT:\n")
print(data)