"""Verify exact release inventory and bytes; ignore root Git metadata only."""
import hashlib, json
from pathlib import Path
root = Path(__file__).resolve().parent
expected = json.loads((root / "release_manifest.json").read_text())["files"]
actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
          if p.is_file() and p.relative_to(root).parts[0] != ".git"}
assert actual == set(expected) | {"release_manifest.json"}, ("inventory mismatch", actual ^ (set(expected) | {"release_manifest.json"}))
for name, digest in expected.items():
    path = root / name
    assert not path.is_symlink() and path.resolve().is_relative_to(root), name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
print("ARTIFACT_INVENTORY_AND_HASHES_PASS", len(expected), "files")
