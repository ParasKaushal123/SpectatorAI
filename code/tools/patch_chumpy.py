"""chumpy (needed to read MANO_RIGHT.pkl) breaks on Python >= 3.11 / NumPy >= 1.24. Patch it in place."""
import importlib.util
import pathlib

spec = importlib.util.find_spec("chumpy")
if spec is None:
    raise SystemExit("chumpy is not installed")
root = pathlib.Path(spec.submodule_search_locations[0])
n = 0
for f in root.glob("*.py"):
    s = f.read_text()
    o = s
    s = s.replace("from numpy import bool, int, float, complex, object, unicode, str, nan, inf", "from numpy import nan, inf")
    s = s.replace("inspect.getargspec", "inspect.getfullargspec")
    if s != o:
        f.write_text(s)
        n += 1
print(f"chumpy patched ({n} files changed)")
