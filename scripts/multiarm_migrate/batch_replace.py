import os

def batch_replace(directory, source_path, target_path, preview=False):
    # Read replacement texts
    with open(source_path, 'r', encoding='utf-8') as f:
        source_text = f.read()
    with open(target_path, 'r', encoding='utf-8') as f:
        target_text = f.read()
    
    files_to_modify = []
    # Walk through the directory
    for root, _, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
            except Exception:
                continue  # Skip unreadable files (e.g., binaries)
            if source_text in content:
                new_content = content.replace(source_text, target_text)
                files_to_modify.append((file_path, content, new_content))
    
    if preview:
        print("Preview of changes:")
        for file_path, old_content, new_content in files_to_modify:
            print(f"File: {file_path} will be modified.")
        return files_to_modify
    else:
        for file_path, old_content, new_content in files_to_modify:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(new_content)
        print("Batch replacement completed.")

# Example usage:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch replace text in files.")
    parser.add_argument("directory", help="Target directory to search")
    parser.add_argument("source_path", help="File path with source text")
    parser.add_argument("target_path", help="File path with target text")
    parser.add_argument("--execute", action="store_true", default=False, help="Preview changes without saving")
    args = parser.parse_args()
    batch_replace(args.directory, args.source_path, args.target_path, preview=not args.execute)
