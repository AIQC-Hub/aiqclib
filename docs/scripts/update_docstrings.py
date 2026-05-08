#!/usr/bin/env python3
import argparse
import pathlib
from google import genai

def get_python_files(folder_path: pathlib.Path):
    return list(folder_path.rglob("*.py"))

def load_prompt(prompt_path: pathlib.Path) -> str:
    return prompt_path.read_text(encoding="utf-8")

def is_empty_file(file_path: pathlib.Path) -> bool:
    # Skip files with zero size
    if file_path.stat().st_size == 0:
        return True
    # Skip files that only contain whitespace/comments
    content = file_path.read_text(encoding="utf-8")
    for line in content.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return False
    return True

def update_file(file_path: pathlib.Path, prompt: str, client: genai.Client, model_name: str):
    if is_empty_file(file_path):
        print(f"⏭ Skipped empty: {file_path}")
        return
    original = file_path.read_text(encoding="utf-8")
    payload = f"{prompt}\n\n{original}"
    try:
        resp = client.models.generate_content(model=model_name, contents=payload)
        file_path.write_text(resp.text, encoding="utf-8")
        print(f"✅ Updated: {file_path}")
    except Exception as e:
        print(f"❌ Failed: {file_path} — {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=pathlib.Path, help="Directory with Python files")
    parser.add_argument("prompt_file", type=pathlib.Path, help="File with prompt text")
    args = parser.parse_args()

    client = genai.Client()
    prompt = load_prompt(args.prompt_file)
    py_files = get_python_files(args.folder)
    if not py_files:
        print("No .py files found.")
        return

    #model_name = "gemini-2.5-flash"
    model_name = "gemini-3-flash-preview"

    for f in py_files:
        update_file(f, prompt, client, model_name)

if __name__ == "__main__":
    main()
