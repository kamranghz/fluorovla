"""Lumen-containment audit (diagnostic only; no physics constant, Teacher rule, task definition or admission rule is changed).

usage: audit_containment.py <mode> ...

  selftest   synthetic analytic checks of the measurement code; any OS; ENGINEERING VALIDATION ONLY (not a scientific calibration)
  preflight  --configs-root DIR --out DIR                                                  environment, canonical constants, inputs (native Linux)
  calibrate  --configs-root DIR --out DIR                                                  locked numerical calibration on analytic synthetic geometry (native Linux, CUDA)
  trace      --configs-root DIR --reference-runs DIR --calibration FILE --out DIR [--resume]   read-only physics trace of the pre-declared cases (native Linux, CUDA)
  analyze    --configs-root DIR --trace-dir DIR --calibration FILE --out DIR [--resume]    CPU analysis of all configurations and of the traces (native Linux)
  verdict    --analysis-dir DIR --calibration FILE --out DIR                               narrow per-case findings from the locked calibration (native Linux)

Every mode that touches real patient data or produces scientific evidence refuses to run on a non-Linux platform. The pre-declared cases and the synthetic design values live in
configs/evaluation/containment_audit.json. Output directories are never overwritten.
"""
import _bootstrap  # noqa: F401  isort:skip
import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from fluorovla.anatomy import containment_audit as ca
from fluorovla.common.paths import repo_root
from fluorovla.common.provenance import capture_environment, code_hash, sha256_file, write_run_manifest

DEFAULT_DESIGN = "configs/evaluation/containment_audit.json"


def die(msg: str, code: int = 2):
    print(f"REFUSED: {msg}", file=sys.stderr)
    sys.exit(code)


def require_linux(mode: str):
    if not sys.platform.startswith("linux"):
        die(f"`{mode}` produces or consumes scientific evidence and runs on native Linux only (this platform: {sys.platform}). Use `selftest` for local engineering validation.")


def dump(obj, path):
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


def load_design(arg) -> dict:
    p = Path(arg) if arg else repo_root() / DEFAULT_DESIGN
    return json.load(open(p, encoding="utf-8"))


def trace_case_names(design: dict, tiers) -> list[str]:
    return [c["name"] for c in design["trace_cases"] if c["tier"] in tiers]


def all_config_names(configs_root: Path) -> list[str]:
    return sorted(d.name for d in configs_root.iterdir() if (d / "config.json").is_file())


def radius_coverage(root: Path, names, design: dict, constants: dict) -> dict:
    """Twin-radius range along the audited centerline stretch of every configuration (anatomy only) against the declared calibration radius bins."""
    from fluorovla.physics.patient_config import PatientConfig
    bins = ca.design_bins(design["numerical_calibration_design"])
    lo, hi, per = np.inf, -np.inf, {}
    for n in names:
        cfg = PatientConfig.from_json(root / n / "config.json")
        z = np.load(cfg.path_npz_path)
        _s0, _pts, rad = ca.audited_path_points(z["pts"], z["rad"], cfg.ct_offset * 1000.0, constants)
        per[n] = [float(rad.min()), float(rad.max())]
        lo, hi = min(lo, rad.min()), max(hi, rad.max())
    covered = bool(bins[0]["lo_mm"] <= lo and hi <= bins[-1]["hi_mm"])
    return {"audited_radius_range_mm": [float(lo), float(hi)], "declared_bins_range_mm": [bins[0]["lo_mm"], bins[-1]["hi_mm"]], "covered": covered, "per_configuration_mm": per,
            "per_configuration_covered": {n: bool(bins[0]["lo_mm"] <= r[0] and r[1] <= bins[-1]["hi_mm"]) for n, r in per.items()}}


def fresh_dir(path) -> Path:
    p = Path(path)
    if p.exists() and any(p.iterdir()):
        die(f"{p} exists and is not empty; evidence is never overwritten (use --resume where supported, or a new directory)")
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------------------------------------------- selftest
def mode_selftest(a):
    res = ca.run_selftest()
    for r in res:
        print(("PASS " if r["passed"] else "FAIL ") + r["check"])
    ok = all(r["passed"] for r in res)
    print(f"\n{sum(r['passed'] for r in res)}/{len(res)} synthetic checks passed. ENGINEERING VALIDATION ONLY: this is not the locked scientific calibration.")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        dump({"engineering_validation_only": True, "platform": sys.platform, "checks": res}, a.out)
    return 0 if ok else 1


# --------------------------------------------------------------------------------------------------------------- preflight
def mode_preflight(a):
    require_linux("preflight")
    from fluorovla.physics.patient_config import PatientConfig
    design, root, checks = load_design(a.design), Path(a.configs_root), []

    def chk(name, ok, detail=None):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    try:
        constants = ca.canonical_constants()
        chk("canonical constants derived from source", True, constants)
        live = ca.live_constants_match(constants)
        chk("source-derived constants equal the running physics modules", all(live.values()), live)
    except Exception as ex:  # noqa: BLE001
        chk("canonical constants derived from source", False, f"{type(ex).__name__}: {ex}")
    try:
        import warp as wp
        wp.init()
        chk("CUDA device available", wp.get_cuda_device_count() > 0, {"warp": wp.__version__, "devices": wp.get_cuda_device_count()})
    except Exception as ex:  # noqa: BLE001
        chk("CUDA device available", False, f"{type(ex).__name__}: {ex}")
    sphere = ca.validate_mesh_reference_on_sphere()
    chk("secondary mesh reference validated (informational; UNAVAILABLE is allowed)", True, {"status": "VALIDATED" if sphere["available"] else "UNAVAILABLE", **sphere})
    names = all_config_names(root)
    chk("episode configurations found", len(names) > 0, len(names))
    for n in trace_case_names(design, ("primary", "supplementary")):
        chk(f"trace case configuration present: {n}", n in names)
    for n in names:
        try:
            cfg = PatientConfig.from_json(root / n / "config.json")
            need = [cfg.ct_dir / "vessel_mask.npy", cfg.ct_dir / "metadata.json", cfg.path_npz_path, cfg.landing_zone_path]
            chk(f"inputs present and sign_scale = 1: {n}", all(p.is_file() for p in need) and cfg.sign_scale == 1.0, [str(p) for p in need if not p.is_file()])
        except Exception as ex:  # noqa: BLE001
            chk(f"configuration readable: {n}", False, f"{type(ex).__name__}: {ex}")
    try:
        cov = radius_coverage(root, names, design, ca.canonical_constants())
        trace_names = trace_case_names(design, ("primary", "supplementary"))
        chk("radius coverage of the declared calibration bins (informational: a sample outside every bin gets no tolerance and makes its event INDETERMINATE; this never aborts a run)", True,
            {"audited_radius_range_mm": cov["audited_radius_range_mm"], "declared_bins_range_mm": cov["declared_bins_range_mm"], "all_configurations_covered": cov["covered"],
             "configurations_not_covered": sorted(n for n, c in cov["per_configuration_covered"].items() if not c),
             "trace_cases": {n: {"radius_range_mm": cov["per_configuration_mm"].get(n), "covered": cov["per_configuration_covered"].get(n)} for n in trace_names}})
    except Exception as ex:  # noqa: BLE001
        chk("radius coverage of the declared calibration bins", False, f"{type(ex).__name__}: {ex}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dump({"checks": checks, "environment": capture_environment(), "code_sha256": code_hash()["combined"]}, out / "preflight.json")
    for c in checks:
        print(("PASS " if c["passed"] else "FAIL ") + c["check"])
    return 0 if all(c["passed"] for c in checks) else 1


# --------------------------------------------------------------------------------------------------------------- calibrate
def spacings_of(configs_root: Path, names) -> list[tuple]:
    from fluorovla.physics.patient_config import PatientConfig
    seen = {}
    for n in names:
        cfg = PatientConfig.from_json(configs_root / n / "config.json")
        sp = tuple(round(float(x), 9) for x in json.load(open(cfg.ct_dir / "metadata.json"))["spacing_zyx_mm"])
        seen[sp] = n
    return sorted(seen)


def mode_calibrate(a):
    if not a.engineering_only:
        require_linux("calibrate")
    design = load_design(a.design)
    constants = ca.canonical_constants()
    root = Path(a.configs_root)
    names = all_config_names(root)
    spacings = spacings_of(root, names)
    coverage = None
    if not a.engineering_only:
        cov = radius_coverage(root, names, design, constants)                              # informational: uncovered samples become INDETERMINATE later, the calibration never aborts for it
        coverage = {k: cov[k] for k in ("audited_radius_range_mm", "declared_bins_range_mm", "covered", "per_configuration_covered")}
        print(f"radius coverage: audited {cov['audited_radius_range_mm']} mm, declared bins {cov['declared_bins_range_mm']} mm, covered: {cov['covered']}", flush=True)
    out = fresh_dir(a.out)
    factory, engine_note = None, "engine path not calibrated"
    try:
        from fluorovla.physics.containment_probe import synthetic_engine_factory
        import warp as wp
        wp.init()
        if wp.get_cuda_device_count() > 0:
            live = ca.live_constants_match(constants)
            if not all(live.values()):
                die(f"source-derived constants differ from the running physics modules: {live}", 3)
            factory, engine_note = synthetic_engine_factory(out / "synthetic_twins"), "engine SDF path calibrated"
    except Exception as ex:  # noqa: BLE001
        engine_note = f"engine path unavailable: {type(ex).__name__}: {ex}"
    print(f"{engine_note}; spacings: {spacings}", flush=True)
    cal = ca.calibrate(design["numerical_calibration_design"], spacings, constants, engine_factory=factory)
    shutil.rmtree(out / "synthetic_twins", ignore_errors=True)
    env = capture_environment()
    locked = bool(sys.platform.startswith("linux") and not a.engineering_only and cal["checks"]["engine_path_calibrated"] and cal["checks"]["fullres_error_within_voxel_diagonal"])
    cal.update({"schema": 1, "purpose": "numerical tolerances of the containment audit measurements (analytic synthetic geometry)", "environment": env, "engine_note": engine_note,
                "locked": locked, "engineering_only": bool(a.engineering_only), "radius_coverage": coverage, "design_config_sha256": sha256_file(Path(a.design) if a.design else repo_root() / DEFAULT_DESIGN),
                "code_sha256": code_hash()["combined"]})
    from fluorovla.evaluation.containment_verdict import calibration_hash
    cal["calibration_sha256"] = calibration_hash(cal)
    name = "ENGINEERING_ONLY_numerical_tolerances.json" if a.engineering_only else "numerical_tolerances.json"
    dump(cal, out / name)
    (out / (name + ".sha256")).write_text(f"{cal['calibration_sha256']}  {name}\n")
    print(json.dumps({"locked": locked, "primary_reference": cal["primary_reference"]["name"], "tolerances_mm": {k: v["value_mm"] for k, v in cal["tolerances"].items()}, "checks": cal["checks"]}, indent=1))
    return 0 if (locked or a.engineering_only) else 1


# --------------------------------------------------------------------------------------------------------------- trace
def mode_trace(a):
    require_linux("trace")
    from fluorovla.evaluation.containment_verdict import CalibrationError, load_locked_calibration
    from fluorovla.evaluation.episode_comparison import compare_regression
    from fluorovla.evaluation.physics_mapping_summary import accepted_attempt
    from fluorovla.physics.containment_probe import ContainmentProbe
    from fluorovla.physics.episode_runner import run_physics_episode
    from fluorovla.physics.patient_config import PatientConfig
    try:
        cal = load_locked_calibration(a.calibration)
    except CalibrationError as ex:
        die(f"calibration not usable: {ex}", 3)
    design, root = load_design(a.design), Path(a.configs_root)
    out = Path(a.out)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    names = trace_case_names(design, a.tiers)
    for n in names:
        d = out / "traces" / n
        if a.resume and (d / "probe.npz").is_file():
            print(f"{n}: reused", flush=True)
            continue
        cfg = PatientConfig.from_json(root / n / "config.json")
        probe = ContainmentProbe()
        s = run_physics_episode(cfg, d / "run", "production", observer=probe)
        arrays, meta = probe.arrays()
        ref_ep = None
        if a.reference_runs:
            acc, _ = accepted_attempt(Path(a.reference_runs) / n)
            if acc is not None:
                ref_ep = json.load(open(acc / "run/episode_physics_ep.json"))
        inst = json.load(open(d / "run" / "episode_physics_production.json"))
        meta.update({"case": n, "status": s["status"], "decisions": s["decisions"], "original_status": ref_ep["summary"]["status"] if ref_ep else None,
                     "instrumented_matches_original_bit_identically": compare_regression(ref_ep, inst)["bit_identical_rows"] if ref_ep else None, "calibration_sha256": cal["calibration_sha256"]})
        np.savez_compressed(d / "probe.npz", **arrays)
        dump(meta, d / "probe_meta.json")
        print(f"{n}: {s['status']} after {s['decisions']} decisions; bit-identical to original: {meta['instrumented_matches_original_bit_identically']}", flush=True)
    write_run_manifest(out / f"manifest_{time.strftime('%Y%m%d_%H%M%S')}", "containment audit physics trace", names, {"configs_root": a.configs_root, "tiers": a.tiers, "calibration_sha256": cal["calibration_sha256"]}, "COMPLETED")
    return 0


# --------------------------------------------------------------------------------------------------------------- analyze
def mode_analyze(a):
    require_linux("analyze")
    from fluorovla.evaluation.containment_verdict import CalibrationError, bin_tolerance_fn, build_thresholds, load_locked_calibration, primary_reference_name
    from fluorovla.physics.patient_config import PatientConfig
    try:
        cal = load_locked_calibration(a.calibration)
    except CalibrationError as ex:
        die(f"calibration not usable: {ex}", 3)
    design, root = load_design(a.design), Path(a.configs_root)
    constants = ca.canonical_constants()
    primary = primary_reference_name(cal)
    thr = build_thresholds(cal)
    tolerance_of_radius = bin_tolerance_fn(thr)
    p99_of_radius = bin_tolerance_fn(thr, "tol_fullres_reference_p99_by_bin") if "tol_fullres_reference_p99_by_bin" in thr else None
    sign_bound = thr.get("engine_sign_guard_bound_mm", {}).get("value")
    out = Path(a.out)
    (out / "configs").mkdir(parents=True, exist_ok=True)
    (out / "traces").mkdir(exist_ok=True)
    traced = set(trace_case_names(design, ("primary", "supplementary")))
    names = all_config_names(root)
    geoms: dict = {}
    fid_cache: dict = {}
    rows = []
    for n in names:
        done = out / "configs" / f"{n}.json"
        if a.resume and done.is_file():
            res = json.load(open(done))
        else:
            cfg = PatientConfig.from_json(root / n / "config.json")
            key = (str(cfg.ct_dir), str(cfg.mesh_cache))
            if key not in geoms:
                geoms.clear()                                           # one patient volume in memory at a time
                geoms[key] = ca.load_patient_geometry(cfg, constants)
                fid_cache.clear()
            g = geoms[key]
            mesh_ref = None
            if n in traced:
                mv, mf = ca.collision_mesh_ct_mm(g.twin["mask"], g.twin["origin_xyz_mm"], g.twin["spacing_zyx_mm"], constants["mask_downsample"])
                mesh_ref = ca.mesh_reference_or_none(mv, mf) if ca.validate_mesh_reference_on_sphere()["available"] else None
            res = ca.config_analysis(cfg, g, constants, mesh_ref)
            if key not in fid_cache:
                fid_cache[key] = ca.mesh_fidelity(g, constants)
            res["mesh_fidelity"] = fid_cache[key]
            res["mesh_reference_status"] = "VALIDATED" if mesh_ref is not None else "UNAVAILABLE"
            if n in traced and (Path(a.trace_dir) / "traces" / n / "probe.npz").is_file():
                td = Path(a.trace_dir) / "traces" / n
                meta = json.load(open(td / "probe_meta.json"))
                tr = dict(np.load(td / "probe.npz"))
                radius_at = ca.local_radius_fn(*[np.load(cfg.path_npz_path)[k] for k in ("pts", "rad")])
                fields = {"fullres": g.primary(primary), "ds": g.ds, "alt": g.fullres_hc if primary == "fullres" else g.fullres}          # 'fullres' = the PRIMARY full-resolution variant
                ta = ca.analyze_trace(tr, meta, fields, constants, mesh_ref, tol_fn=lambda pts: tolerance_of_radius(radius_at(pts)), sign_bound_mm=sign_bound,
                                  tol_p99_fn=(lambda pts: p99_of_radius(radius_at(pts))) if p99_of_radius else None)
                ta["primary_reference"] = primary
                dump({"trace_analysis": ta, "trace_meta": meta}, out / "traces" / f"{n}.json")
            dump(res, done)
        s = res["straight_deployment"][primary]
        rows.append({"config": n, "reference": primary, "axis_violation_primary_mm": s["axis_violation_mm"], "axis_violation_ds_mm": res["straight_deployment"]["ds"]["axis_violation_mm"],
                     "capsule_penetration_primary_mm": s["capsule_penetration_mm"], "first_axis_exceedance_arclength_primary_mm": s["first_axis_exceedance_arclength_mm"],
                     "path_fraction_below_catheter_radius_primary": res["path_clearance"][primary]["fraction_wall_distance_below_catheter_radius"],
                     "min_wall_distance_primary_mm": res["path_clearance"][primary]["min_wall_distance_mm"], "twin_radius_range_mm": res["path_clearance"][primary]["twin_radius_range_mm"], "traced": n in traced})
        print(f"{n}: straight-deployment axis violation {s['axis_violation_mm']:.2f} mm ({primary}; quantitative only)", flush=True)
    with open(out / "straight_deployment_all_configurations.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_run_manifest(out / f"manifest_{time.strftime('%Y%m%d_%H%M%S')}", "containment audit CPU analysis", names, {"configs_root": a.configs_root, "trace_dir": a.trace_dir, "calibration_sha256": cal["calibration_sha256"]}, "COMPLETED")
    return 0


# --------------------------------------------------------------------------------------------------------------- verdict
def mode_verdict(a):
    require_linux("verdict")
    from fluorovla.evaluation.containment_verdict import CalibrationError, containment_verdict, load_locked_calibration
    try:
        cal = load_locked_calibration(a.calibration)
    except CalibrationError as ex:
        die(f"calibration not usable: {ex}", 3)
    design, ad = load_design(a.design), Path(a.analysis_dir)
    out = fresh_dir(a.out)
    allc = {p.stem: json.load(open(p)) for p in sorted((ad / "configs").glob("*.json"))}
    cases = []
    roles = {c["name"]: c["stage3a_role"] for c in design["trace_cases"]}                    # pre-declared roles; never derived from outcomes
    for n in trace_case_names(design, ("primary", "supplementary")):
        t = ad / "traces" / f"{n}.json"
        tj = json.load(open(t)) if t.is_file() else {}
        meta = tj.get("trace_meta") or {}
        cases.append({"name": n, "role": roles[n], "descriptive_outcome": meta.get("status"), "config_analysis": allc[n], "trace_analysis": tj.get("trace_analysis"), "trace_meta": meta})
    res = containment_verdict(cases, cal, allc)
    dump(res, out / "CONTAINMENT_FINDINGS.json")
    with open(out / "FINDINGS.md", "w", encoding="utf-8") as f:
        f.write("| case | pre-declared role | descriptive outcome (not an input) | label | reason codes |\n|---|---|---|---|---|\n")
        for c in res["traced_cases"]:
            f.write(f"| {c['name']} | {c['role']} | {c['descriptive_outcome']} | {c['label']} | {', '.join(c['reason_codes'])} |\n")
        f.write("\n| mechanism | positive controls (raw status) | non-discriminating | specificity established |\n|---|---|---|---|\n")
        for m, sp in res["specificity"].items():
            f.write(f"| {m} | {sp['positive_controls']} | {sp['non_discriminating']} | {sp['specificity_established']} |\n")
    write_run_manifest(out / "manifest", "containment audit findings", [c["name"] for c in cases], {"analysis_dir": a.analysis_dir, "calibration_sha256": cal["calibration_sha256"]}, "COMPLETED",
                       {"labels": {c["name"]: c["label"] for c in res["traced_cases"]}})
    for c in res["traced_cases"]:
        print(f"{c['name']}: {c['label']}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("selftest")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=mode_selftest)
    for name, fn in (("preflight", mode_preflight), ("calibrate", mode_calibrate), ("trace", mode_trace), ("analyze", mode_analyze), ("verdict", mode_verdict)):
        p = sub.add_parser(name)
        p.add_argument("--design", default=None, help=f"design configuration (default {DEFAULT_DESIGN})")
        p.add_argument("--out", required=True)
        if name in ("preflight", "calibrate", "trace", "analyze"):
            p.add_argument("--configs-root", required=True)
        if name in ("trace", "analyze", "verdict"):
            p.add_argument("--calibration", required=True)
        if name == "trace":
            p.add_argument("--reference-runs", default=None)
            p.add_argument("--tiers", nargs="+", default=["primary", "supplementary"])
            p.add_argument("--resume", action="store_true")
        if name == "analyze":
            p.add_argument("--trace-dir", required=True)
            p.add_argument("--resume", action="store_true")
        if name == "verdict":
            p.add_argument("--analysis-dir", required=True)
        if name == "calibrate":
            p.add_argument("--engineering-only", action="store_true", help="local engineering run: never locked, never usable by trace/analyze/verdict")
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
