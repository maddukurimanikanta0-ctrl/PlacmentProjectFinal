"""
tests/check_line_lengths.py
---------------------------
Strict audit verifying that EVERY individual source-code file
(.py, .html, .css, .js) contains strictly FEWER than 300 lines of code.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTENSIONS = {".py", ".html", ".css", ".js"}
EXCLUDE_DIRS = {".venv", ".git", "__pycache__", ".idea", "node_modules"}
MAX_ALLOWED_LINES = 299


def check_source_files():
    violations = []
    inspected = []

    for file_path in ROOT.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in EXTENSIONS:
            continue
        if any(part in EXCLUDE_DIRS for part in file_path.parts):
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                line_count = len(lines)
                inspected.append((str(file_path.relative_to(ROOT)), line_count))
                if line_count > MAX_ALLOWED_LINES:
                    violations.append((str(file_path.relative_to(ROOT)), line_count))
        except UnicodeDecodeError:
            continue

    inspected.sort(key=lambda x: x[1], reverse=True)
    print(f"\nChecked {len(inspected)} source-code files.")
    print("Top 10 largest source files:")
    for path, count in inspected[:10]:
        print(f"  - {path}: {count} lines")

    if violations:
        print(f"\nFAILED: {len(violations)} files exceed {MAX_ALLOWED_LINES} lines:")
        for path, count in violations:
            print(f"  * {path}: {count} lines")
        raise SystemExit(1)
    else:
        print(f"\nPASSED: 100% of source files strictly contain fewer than 300 lines! (Max found: {inspected[0][1]} lines in {inspected[0][0]})\n")


if __name__ == "__main__":
    check_source_files()
