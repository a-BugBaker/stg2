import ast, pathlib, sys

files = list(pathlib.Path('stg').glob('*.py')) + list(pathlib.Path('scripts').glob('*.py'))
errs = []
for f in files:
    try:
        ast.parse(f.read_text(encoding='utf-8'))
        print(f"OK: {f}")
    except SyntaxError as e:
        errs.append(f"{f}: {e}")
        print(f"FAIL: {f}: {e}")

print()
if errs:
    for e in errs:
        print(e)
    sys.exit(1)
else:
    print("All files passed syntax check!")
