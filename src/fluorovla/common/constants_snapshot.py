"""Snapshot of the fixed physical / task constants (module-level numeric literals) of the scientific modules, read from the source with ``ast`` (no imports,
so it works without Warp or CUDA). ``configs/physics/fixed_constants.json`` stores the reviewed snapshot; ``tests/test_fixed_constants.py`` fails when the code
and the snapshot differ, so a constant can only change through a deliberate, documented update (protocol erratum)."""
from __future__ import annotations

import ast
from pathlib import Path

MODULES = ("task_contract.py", "teacher/privileged_teacher.py", "physics/engine.py", "physics/lumen_sdf.py", "physics/patient_episode.py",
           "anatomy/access_geometry.py", "anatomy/episode_config.py")


def _literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:  # noqa: BLE001
        pass
    try:  # simple arithmetic on literals / earlier names is not evaluated: keep the source text
        return "expr: " + ast.unparse(node)
    except Exception:  # noqa: BLE001
        return None


def snapshot(package_dir: str | Path | None = None) -> dict:
    root = Path(package_dir) if package_dir else Path(__file__).resolve().parents[1]
    out = {}
    for m in MODULES:
        tree = ast.parse((root / m).read_text(encoding="utf-8"))
        consts = {}
        for n in tree.body:
            targets = []
            if isinstance(n, ast.Assign):
                if len(n.targets) == 1 and isinstance(n.targets[0], ast.Tuple) and isinstance(n.value, ast.Tuple) and len(n.targets[0].elts) == len(n.value.elts):
                    for tn, vn in zip(n.targets[0].elts, n.value.elts):          # R_C, ZONE = 3.0, 4.72
                        if isinstance(tn, ast.Name) and tn.id.isupper():
                            v = _literal(vn)
                            if isinstance(v, (int, float, str, bool)):
                                consts[tn.id] = v
                    continue
                targets = [t for t in n.targets if isinstance(t, ast.Name)]
                value = n.value
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
                targets, value = [n.target], n.value
            else:
                continue
            for t in targets:
                if t.id.isupper() or (t.id.startswith("_") and t.id.lstrip("_").isupper()):
                    v = _literal(value)
                    if isinstance(v, (int, float, str, bool, tuple, list)) or v is None:
                        consts[t.id] = list(v) if isinstance(v, tuple) else v
        out[m] = consts
    return out
