import os
import sys

def rename_files(directory, old_prefix, new_prefix):
    for filename in os.listdir(directory):
        if filename.startswith(old_prefix):
            new_name = new_prefix + filename[len(old_prefix):]
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed {filename} -> {new_name}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python batch_rename.py <directory> <old_prefix> <new_prefix>")
        sys.exit(1)
    directory = sys.argv[1]
    old_prefix = sys.argv[2]
    new_prefix = sys.argv[3]
    rename_files(directory, old_prefix, new_prefix)
