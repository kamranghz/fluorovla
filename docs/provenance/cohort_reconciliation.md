# Cohort-count reconciliation (design stage vs twin build)

Status: audit of existing artifacts only (script `scripts/protocol/cohort_reconciliation.py`); **no scientific cohort has been changed**. The proposal for the authoritative definition (section 3) needs review.

## 1. How the counts arise

| step | patients | source |
|---|---|---|
| subjects in TotalSegmentator-small | 102 | `_pool_scan.json` |
| aorta + both iliac labels present | 73 | `_pool_scan.json` |
| design extractor (skimage skeleton) succeeded | 62 (11 failed, mostly 'no true bifurcation cluster') | `access_path_scan.json` |
| anatomical voxel/z-extent gate (both iliacs >=2000 vox and >=120 mm, aorta >=200 mm) | 61 | `_pool_gate_ids.json` (independent of the extractor) |
| design usable pool (access path >=150 mm, extractor ok) | 59 patients / 112 paths | `access_path_rows_feasible.json`; 5 of them are outside the 61-gate (s0454, s0573, s0815, s1082, s1401) |
| **twin build set** | **69** | union of the 62 patients the design extractor processed and the 61-gate. A build list I chose, **not a pool** |
| twin: >=1 valid 3-arm access path with S_contact >=150 mm | **62** patients / 119 paths | `stage1/twin_vs_design.json` |

## 2. Why 59 became 62 (and what 69 is)

- 69 is the number of patients attempted on the twin, not a cohort. 62 is the twin counterpart of the design 59.
- The 62 are **not a superset** of the 59 (59 - 3 + 6 = 62): 56 of the 59 remain, 3 left, 6 entered.
- **Entered (6)**: s0762, s0777, s0903, s0963, s1233, s1374. On all six the **design-stage extractor failed** ('no true bifurcation cluster'), so they were never evaluated for length. The twin graph gives a valid 3-arm bifurcation and S_contact >=150 mm. This is a design-stage extractor error, not a change of anatomy. All six are in the 61-patient anatomical gate.
- **Left (3)**: s0815, s1369, s1379. s0815 and s1369: the first twin cluster has 5 arms (my rule requires exactly 3). s1379: my twin-geometry code raised an IndexError. These are **limitations of my new twin extractor (strictness, or a bug), not demonstrated anatomy problems**; they have not been inspected visually and are 'unverified', not true exclusions.
- Twin S_contact is on median 13.6 mm longer than the design estimate (the design prototype under-counted skeleton length). No pool-59 path fell below 150 mm for that reason.
- Twin-built but not valid and not in the 59: s1395 (in gate, 4-arm cluster), s0970, s1123, s1384 (outside gate), and the other design-extractor failures that have no twin path >=150 mm (see table).

## 3. Authoritative cohort definition (proposal)

Authoritative = **twin-verified**: a patient is in the protocol pool iff the twin yields at least one valid 3-arm access path with S_contact >= 150 mm. Rationale: the protocol states that twin verification supersedes the design-stage extractor, and the twin reproduces the s0011 reference (183.58 mm) to 0.01 mm. Open items: (a) re-inspect the 3 patients that left, (b) 4 pool patients have quality-flagged descriptors (kept for the admission sweep, excluded from descriptor-based analyses), (c) the Stage 7 sweep universe is the pool minus the 13 development patients.

Counts by final_protocol_status: {"DEVELOPMENT": 12, "POOL_TWIN_VALID": 45, "POOL_TWIN_VALID_DESCRIPTOR_FLAGGED": 4, "EXCLUDED_TWIN_INVALID_OR_SHORT": 7, "DEVELOPMENT_SENTINEL": 1}

## 4. Table (69 twin-built patients; CSV: `outputs/cvpr_protocol_v1/reconciliation/cohort_reconciliation.csv`)

| patient_id | previous_status | twin_status | reason_for_change | final_protocol_status |
|---|---|---|---|---|
| s0011 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=184.7', 'right': 'valid S=183.6'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | DEVELOPMENT |
| s0058 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=200.5', 'right': 'valid S=200.5'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | DEVELOPMENT |
| s0250 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=216.1', 'right': 'valid S=218.4'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s0344 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=242.2', 'right': 'valid S=245.6'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | DEVELOPMENT |
| s0358 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=188.1', 'right': 'valid S=197.3'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s0454 | label73 / scan_ok / not_gate61 / pool59 | VALID>=150 {'left': 'valid S=175.7', 'right': 'valid S=191.8'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s0461 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'invalid:CLUSTER_ARMS_4', 'right': 'valid S=216.9'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s0477 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=184.2', 'right': 'valid S=202.6'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s0513 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=197.1', 'right': 'valid S=222.5'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | DEVELOPMENT |
| s0519 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=189.7', 'right': 'valid S=204.7'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s0573 | label73 / scan_ok / not_gate61 / pool59 | VALID>=150 {'left': 'valid S=199.2', 'right': 'valid S=202.6'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s0591 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=159.2', 'right': 'valid S=169.4'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s0629 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=201.5', 'right': 'valid S=222.6'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s0649 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=196.5', 'right': 'valid S=209.9'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s0703 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=234.2', 'right': 'valid S=236.4'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s0720 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=214.4', 'right': 'valid S=219.3'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s0726 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=186.6', 'right': 'valid S=206.6'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s0727 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=171.2', 'right': 'valid S=179.9'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s0762 | label73 / scan_FAILED:ValueError('no true bifurcation cluster' / gate61 / not_pool59 | VALID>=150 {'left': 'valid S=197.7', 'right': 'valid S=168.8'} | design extractor failed ('no true bifurcation cluster'); twin graph finds a valid cluster -> design-stage extractor error, anatomy valid | POOL_TWIN_VALID_DESCRIPTOR_FLAGGED |
| s0763 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=222.6', 'right': 'valid S=231.8'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s0777 | label73 / scan_FAILED:ValueError('no true bifurcation cluster' / gate61 / not_pool59 | VALID>=150 {'left': 'valid S=186.6', 'right': 'valid S=181.5'} | design extractor failed ('no true bifurcation cluster'); twin graph finds a valid cluster -> design-stage extractor error, anatomy valid | POOL_TWIN_VALID |
| s0812 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=179.6', 'right': 'valid S=184.1'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s0815 | label73 / scan_ok / not_gate61 / pool59 | NO_VALID_PATH>=150 {'left': 'invalid:CLUSTER_ARMS_5', 'right': 'invalid:CLUSTER_ARMS_5'} | twin cluster not a 3-arm bifurcation: ['CLUSTER_ARMS_5', 'CLUSTER_ARMS_5'] | EXCLUDED_TWIN_INVALID_OR_SHORT |
| s0859 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=191.6', 'right': 'valid S=220.0'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s0878 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=189.0', 'right': 'valid S=193.8'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s0885 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=247.0', 'right': 'valid S=246.8'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s0903 | label73 / scan_FAILED:ValueError('no true bifurcation cluster' / gate61 / not_pool59 | VALID>=150 {'left': 'valid S=189.0', 'right': 'valid S=200.3'} | design extractor failed ('no true bifurcation cluster'); twin graph finds a valid cluster -> design-stage extractor error, anatomy valid | POOL_TWIN_VALID |
| s0912 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=217.3', 'right': 'valid S=224.0'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s0915 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=219.1', 'right': 'valid S=211.1'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s0928 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=198.5', 'right': 'valid S=97.8'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s0950 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=77.7', 'right': 'valid S=243.3'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | DEVELOPMENT |
| s0963 | label73 / scan_FAILED:ValueError('no true bifurcation cluster' / gate61 / not_pool59 | VALID>=150 {'left': 'valid S=165.3', 'right': 'valid S=163.1'} | design extractor failed ('no true bifurcation cluster'); twin graph finds a valid cluster -> design-stage extractor error, anatomy valid | DEVELOPMENT |
| s0970 | label73 / scan_ok / not_gate61 / not_pool59 | NO_VALID_PATH>=150 {'left': 'valid S=111.6', 'right': 'valid S=129.7'} | not in design pool and no valid twin path >=150 mm | EXCLUDED_TWIN_INVALID_OR_SHORT |
| s1029 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=209.8', 'right': 'valid S=214.8'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s1082 | label73 / scan_ok / not_gate61 / pool59 | VALID>=150 {'left': 'valid S=143.6', 'right': 'valid S=156.2'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | DEVELOPMENT_SENTINEL |
| s1086 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=222.3', 'right': 'valid S=248.5'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s1123 | label73 / scan_ok / not_gate61 / not_pool59 | NO_VALID_PATH>=150 {'left': 'valid S=85.8', 'right': 'valid S=97.3'} | not in design pool and no valid twin path >=150 mm | EXCLUDED_TWIN_INVALID_OR_SHORT |
| s1124 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=215.7', 'right': 'valid S=223.6'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s1127 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=204.7', 'right': 'valid S=217.1'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s1206 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=156.3', 'right': 'valid S=164.0'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s1233 | label73 / scan_FAILED:ValueError('no true bifurcation cluster' / gate61 / not_pool59 | VALID>=150 {'left': 'valid S=186.6', 'right': 'valid S=192.1'} | design extractor failed ('no true bifurcation cluster'); twin graph finds a valid cluster -> design-stage extractor error, anatomy valid | POOL_TWIN_VALID |
| s1267 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=231.9', 'right': 'valid S=262.3'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s1287 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=239.9', 'right': 'valid S=254.7'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s1336 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=232.5', 'right': 'valid S=225.0'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s1348 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=178.3', 'right': 'valid S=203.2'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | DEVELOPMENT |
| s1350 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=174.8', 'right': 'valid S=195.6'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s1361 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=164.4', 'right': 'valid S=253.7'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID_DESCRIPTOR_FLAGGED |
| s1364 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=190.2', 'right': 'valid S=212.6'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | DEVELOPMENT |
| s1365 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=192.8', 'right': 'valid S=204.3'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s1366 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=200.6', 'right': 'valid S=204.8'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s1369 | label73 / scan_ok / gate61 / pool59 | NO_VALID_PATH>=150 {'left': 'invalid:CLUSTER_ARMS_5', 'right': 'invalid:CLUSTER_ARMS_5'} | twin cluster not a 3-arm bifurcation: ['CLUSTER_ARMS_5', 'CLUSTER_ARMS_5'] | EXCLUDED_TWIN_INVALID_OR_SHORT |
| s1371 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=226.2', 'right': 'valid S=243.3'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s1372 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=222.2', 'right': 'valid S=215.4'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s1373 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=214.7', 'right': 'valid S=262.2'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID_DESCRIPTOR_FLAGGED |
| s1374 | label73 / scan_FAILED:ValueError('no true bifurcation cluster' / gate61 / not_pool59 | VALID>=150 {'left': 'valid S=183.5', 'right': 'valid S=172.5'} | design extractor failed ('no true bifurcation cluster'); twin graph finds a valid cluster -> design-stage extractor error, anatomy valid | POOL_TWIN_VALID |
| s1379 | label73 / scan_ok / gate61 / pool59 | TWIN_GEOMETRY_ERROR IndexError('index -1 is out of bounds for axis 0 with size 0') | twin geometry error | EXCLUDED_TWIN_INVALID_OR_SHORT |
| s1383 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=212.9', 'right': 'valid S=149.4'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s1384 | label73 / scan_ok / not_gate61 / not_pool59 | NO_VALID_PATH>=150 {'left': 'valid S=125.5', 'right': 'valid S=129.0'} | not in design pool and no valid twin path >=150 mm | EXCLUDED_TWIN_INVALID_OR_SHORT |
| s1387 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=194.5', 'right': 'valid S=199.9'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s1388 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=173.7', 'right': 'valid S=181.2'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | DEVELOPMENT |
| s1390 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=228.0', 'right': 'valid S=236.1'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s1391 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=209.7', 'right': 'valid S=213.6'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | DEVELOPMENT |
| s1395 | label73 / scan_FAILED:ValueError('no true bifurcation cluster' / gate61 / not_pool59 | NO_VALID_PATH>=150 {'left': 'invalid:CLUSTER_ARMS_4', 'right': 'invalid:CLUSTER_ARMS_4'} | not in design pool and no valid twin path >=150 mm | EXCLUDED_TWIN_INVALID_OR_SHORT |
| s1397 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=216.9', 'right': 'valid S=233.8'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | DEVELOPMENT |
| s1400 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=279.9', 'right': 'valid S=276.8'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | DEVELOPMENT |
| s1401 | label73 / scan_ok / not_gate61 / pool59 | VALID>=150 {'left': 'valid S=203.6', 'right': 'valid S=204.8'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s1403 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=213.9', 'right': 'valid S=208.7'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID_DESCRIPTOR_FLAGGED |
| s1404 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=204.9', 'right': 'valid S=218.1'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
| s1405 | label73 / scan_ok / gate61 / pool59 | VALID>=150 {'left': 'valid S=181.2', 'right': 'valid S=187.3'} | unchanged membership; twin S longer than design S on median 13.6 mm; descriptors re-measured on the twin | POOL_TWIN_VALID |
