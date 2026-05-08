import re
import os

# Get the directory where this script lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the absolute path to the api folder relative to script location
API_DIR = os.path.join(SCRIPT_DIR, '..', 'source', 'api')

# Normalize path
API_DIR = os.path.normpath(API_DIR)

print(f"Cleaning API rst files in: {API_DIR}")

def remove_module_contents_section(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex to match "Module contents" and everything after it
    new_content = re.split(r'\nModule contents\n[-=]+\n', content, maxsplit=1)[0]

    # Only overwrite if the section actually existed
    if new_content != content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Cleaned: {filepath}")
    else:
        print(f"Skipped (no section): {filepath}")

def main():
    for filename in os.listdir(API_DIR):
        if filename.endswith(".rst"):
            remove_module_contents_section(os.path.join(API_DIR, filename))

if __name__ == "__main__":
    main()
