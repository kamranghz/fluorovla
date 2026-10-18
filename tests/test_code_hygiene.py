"""No patient-specific scientific branching, no history-named canonical modules, banned terminology, fixed-constant guard."""
import ast
import json
import re

from fluorovla.common.constants_snapshot import snapshot
from conftest import ROOT

PATIENT_ID = re.compile(r"^s\d{4}$")


def _string_constants(tree):
    doc_nodes = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant):
            doc_nodes.add(id(n.body[0].value))
    return [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in doc_nodes]


def test_no_patient_identifiers_in_source_logic():
    hits = []
    for p in list((ROOT / "src").rglob("*.py")) + list((ROOT / "tools").rglob("*.py")):
        for c in _string_constants(ast.parse(p.read_text(encoding="utf-8"))):
            if PATIENT_ID.match(c.value):
                hits.append(f"{p.relative_to(ROOT)}:{c.lineno}:{c.value}")
    assert not hits, hits


def test_canonical_module_names_are_semantic():
    banned = re.compile(r"(stage\d|cvpr|design_v\d|pilot_v\d|final_v|_v\d$|_new|_fixed|_latest)")
    names = [p.stem for p in (ROOT / "src").rglob("*.py")] + [p.stem for p in (ROOT / "tools").rglob("*.py")]
    assert not [n for n in names if banned.search(n)]


def test_banned_terminology():
    word = "ora" + "cle"
    for p in list((ROOT / "src").rglob("*.py")) + list((ROOT / "tools").rglob("*.py")) + list(ROOT.glob("*.md")) + list((ROOT / "docs").rglob("*.md")):
        assert word not in p.read_text(encoding="utf-8", errors="ignore").lower(), p


def test_fixed_constants_match_reviewed_snapshot():
    reviewed = json.load(open(ROOT / "configs/physics/fixed_constants.json"))["constants"]
    assert snapshot() == reviewed
