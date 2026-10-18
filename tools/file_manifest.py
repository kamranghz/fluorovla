"""SHA-256 manifest of a directory tree.
usage: file_manifest.py <root> <manifest.txt>            write the manifest
       file_manifest.py <root> <manifest.txt> --verify   verify (exit 1 on any mismatching or missing file; extra files are reported)"""
import _bootstrap  # noqa: F401  isort:skip
import sys

from fluorovla.common.provenance import verify_file_manifest, write_file_manifest

root, mf = sys.argv[1], sys.argv[2]
if "--verify" in sys.argv:
    r = verify_file_manifest(root, mf)
    print(f"verified {r['checked']} files; mismatched/missing {len(r['bad'])}; extra {len(r['extra'])}")
    for b in r["bad"][:20]:
        print("BAD", b)
    sys.exit(1 if r["bad"] else 0)
print("hashed", write_file_manifest(root, mf))
