import ast, sys

files = [
    "src/trade_logging/__init__.py",
    "src/trade_logging/trade_event_server.py",
    "src/core/config.py",
    "src/main.py",
]

all_ok = True
for f in files:
    try:
        with open(f, "r", encoding="utf-8") as fh:
            src = fh.read()
        ast.parse(src)
        print("OK: " + f)
    except SyntaxError as e:
        print("FAIL: " + f + " -> " + str(e))
        all_ok = False

print("ALL PASSED" if all_ok else "ERRORS FOUND")
sys.exit(0 if all_ok else 1)

