from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


REPO = Path(r"D:\Benjamin Data\Introduction to AI\Intro-to-AI")
OUT = REPO / "notebook_structure_anomaly_audit_outputs"
OUT.mkdir(parents=True, exist_ok=True)

EXCLUDED_PARTS = {
    ".ipynb_checkpoints",
    "audit_outputs",
    "notebook_structure_anomaly_audit_outputs",
}

FUNCTION_RE = re.compile(r"^[ \t]*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", re.M)
CLASS_RE = re.compile(r"^[ \t]*class\s+([A-Za-z_]\w*)\s*[\(:]", re.M)
HELPER_HEADING_RE = re.compile(
    r"\b(helper|helpers|utility|utilities|function|functions|common helpers|supporting functions)\b",
    re.I,
)
EXPECTED_LATER_RES = [
    re.compile(
        r"^(load_audio|read_audio|scan_audio|parse_|extract_|aggregate_|build_feature|build_mfcc|prepare_|make_|create_)",
        re.I,
    ),
    re.compile(
        r"^(evaluate|plot_|save_|load_|run_|train_|fit_|compute_|summarize_|collect_|build_|display_)",
        re.I,
    ),
]

STAGE_RES = [
    ("imports/setup", re.compile(r"^\s*(import|from)\s+|^\s*[%!]", re.I | re.M)),
    (
        "configuration/paths",
        re.compile(
            r"\b(RANDOM_STATE|SEED|PROJECT_ROOT|OUTPUT_|DATA|DIR|PATH|SAMPLE_RATE|BATCH_SIZE|EPOCHS|N_MFCC|CACHE_DIR)\b"
        ),
    ),
    (
        "dataset loading",
        re.compile(
            r"\b(read_csv|load_dataset|rglob|os\.listdir|manifest|train_df|validation_df|test_df|train_test_split|dataset|audio file)\b",
            re.I,
        ),
    ),
    (
        "feature preparation",
        re.compile(r"\b(mfcc|feature|preprocess|extract|scaler|standard|normalise|normalize|aggregate)\b", re.I),
    ),
    (
        "model definition",
        re.compile(r"\b(Sequential|Dense|Conv2D|SVC|RandomForest|LogisticRegression|XGB|model\s*=|classifier)\b", re.I),
    ),
    (
        "training",
        re.compile(r"\b(fit\(|train|training|epoch|early stopping|GridSearch|RandomizedSearch)\b", re.I),
    ),
    (
        "evaluation",
        re.compile(r"\b(predict|accuracy|precision|recall|f1|classification_report|confusion_matrix|evaluate|metrics)\b", re.I),
    ),
    (
        "outputs/saving",
        re.compile(r"\b(to_csv|savefig|joblib\.dump|pickle\.dump|save\(|OUTPUT|FIGURE|TABLE|METRICS|MODEL|saved outputs)\b", re.I),
    ),
]


def git(args: list[str], text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True,
        text=text,
        encoding="utf-8" if text else None,
        errors="replace" if text else None,
    )


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def source_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "".join(str(x) for x in value)
    return str(value)


def normalize_source(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip()


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def source_hash(text: str) -> str:
    return hashlib.sha256(normalize_source(text).encode("utf-8", errors="replace")).hexdigest()[:16]


def one_line(text: str, limit: int = 260) -> str:
    compact = re.sub(r"\s+", " ", str(text)).strip()
    if len(compact) > limit:
        return compact[: limit - 3].rstrip() + "..."
    return compact


def markdown_escape(text) -> str:
    return str(text).replace("|", r"\|").replace("\n", " ")


def cell_stage(text: str) -> str:
    stages = [name for name, rx in STAGE_RES if rx.search(text)]
    if FUNCTION_RE.search(text) or CLASS_RE.search(text):
        stages.append("function definitions")
    return "; ".join(stages) if stages else "unclassified"


def find_defs(text: str) -> tuple[list[dict], list[dict]]:
    functions: list[dict] = []
    classes: list[dict] = []
    try:
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append({"name": node.name, "line": getattr(node, "lineno", "")})
            elif isinstance(node, ast.ClassDef):
                classes.append({"name": node.name, "line": getattr(node, "lineno", "")})
        return functions, classes
    except Exception:
        pass

    for match in FUNCTION_RE.finditer(text):
        functions.append({"name": match.group(1), "line": text.count("\n", 0, match.start()) + 1})
    for match in CLASS_RE.finditer(text):
        classes.append({"name": match.group(1), "line": text.count("\n", 0, match.start()) + 1})
    return functions, classes


def heading_before(cells: list[dict], index: int) -> str:
    for i in range(min(index, len(cells) - 1), -1, -1):
        if cells[i].get("cell_type") != "markdown":
            continue
        text = source_text(cells[i].get("source"))
        for line in text.splitlines():
            if line.strip().startswith("#"):
                return line.strip()
    return ""


def first_stage_cell(cells: list[dict], stage_name: str) -> int | None:
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        if stage_name in cell_stage(source_text(cell.get("source"))):
            return i
    return None


def extract_profile(nb: dict) -> dict:
    cells = nb.get("cells", [])
    profile = {
        "cell_count": len(cells),
        "code_count": 0,
        "markdown_count": 0,
        "cells": [],
        "headings": [],
        "functions": [],
        "classes": [],
        "function_positions": defaultdict(list),
        "hash_positions": defaultdict(list),
        "first_dataset_cell": first_stage_cell(cells, "dataset loading"),
        "first_model_cell": first_stage_cell(cells, "model definition"),
        "first_eval_cell": first_stage_cell(cells, "evaluation"),
    }

    code_cells: list[tuple[int, str]] = []
    for i, cell in enumerate(cells):
        ctype = cell.get("cell_type", "unknown")
        text = source_text(cell.get("source"))
        norm = normalize_source(text)
        h = source_hash(text) if norm else ""
        if h:
            profile["hash_positions"][h].append(i)
        if ctype == "code":
            profile["code_count"] += 1
            code_cells.append((i, text))
        elif ctype == "markdown":
            profile["markdown_count"] += 1
            for line in text.splitlines():
                if line.strip().startswith("#"):
                    profile["headings"].append({"cell": i, "heading": line.strip()})
        profile["cells"].append(
            {
                "index": i,
                "type": ctype,
                "text": text,
                "hash": h,
                "stage": cell_stage(text) if ctype == "code" else "markdown",
                "line_count": text.count("\n") + 1 if text else 0,
            }
        )

    for i, text in code_cells:
        functions, classes = find_defs(text)
        heading = heading_before(cells, i)
        ratio = i / max(1, len(cells) - 1)
        for f in functions:
            record = {
                "name": f["name"],
                "cell": i,
                "line": f["line"],
                "ratio": ratio,
                "heading": heading,
                "stage": cell_stage(text),
                "cell_lines": text.count("\n") + 1 if text else 0,
                "first_use_after_def": "",
                "gap_to_first_use": "",
                "call_count": 0,
            }
            profile["functions"].append(record)
            profile["function_positions"][f["name"]].append(i)
        for c in classes:
            profile["classes"].append(
                {
                    "name": c["name"],
                    "cell": i,
                    "line": c["line"],
                    "ratio": ratio,
                    "heading": heading,
                    "stage": cell_stage(text),
                    "cell_lines": text.count("\n") + 1 if text else 0,
                }
            )

    for f in profile["functions"]:
        call_rx = re.compile(rf"(?<!def\s)\b{re.escape(f['name'])}\s*\(")
        call_count = 0
        first_use = None
        def_line_rx = re.compile(rf"^[ \t]*(?:async\s+)?def\s+{re.escape(f['name'])}\s*\([^\n]*", re.M)
        for i, text in code_cells:
            without_def_line = def_line_rx.sub("", text)
            matches = list(call_rx.finditer(without_def_line))
            if matches:
                call_count += len(matches)
                if i > f["cell"] and first_use is None:
                    first_use = i
        f["call_count"] = call_count
        if first_use is not None:
            f["first_use_after_def"] = first_use
            f["gap_to_first_use"] = first_use - f["cell"]

    return profile


def expected_later_function(name: str) -> bool:
    return any(rx.search(name) for rx in EXPECTED_LATER_RES)


def severity_key(value: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(value, 9)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_working_notebook(path: Path) -> tuple[dict | None, str]:
    try:
        return json.loads(path.read_bytes().decode("utf-8-sig")), ""
    except Exception as exc:
        return None, repr(exc)


def load_head_notebook(path_rel: str) -> tuple[dict | None, str]:
    result = git(["show", f"HEAD:{path_rel}"], text=False)
    if result.returncode != 0:
        return None, result.stderr.decode("utf-8", errors="replace")
    try:
        return json.loads(result.stdout.decode("utf-8-sig")), ""
    except Exception as exc:
        return None, repr(exc)


def get_tracked_notebooks() -> set[str]:
    result = git(["ls-files", "-z", "*.ipynb"], text=False)
    paths = result.stdout.decode("utf-8", errors="replace").split("\0")
    return {p for p in paths if p}


def get_modified_notebooks() -> set[str]:
    result = git(["status", "--porcelain=v1", "-z", "--", "*.ipynb"], text=False)
    records = result.stdout.decode("utf-8", errors="replace").split("\0")
    modified = set()
    for record in records:
        if not record:
            continue
        path = record[3:].strip().strip('"').replace("\\", "/")
        if path.endswith(".ipynb"):
            modified.add(path)
    return modified


def get_numstat() -> dict[str, dict[str, str]]:
    result = git(["diff", "--numstat", "--", "*.ipynb"])
    stats: dict[str, dict[str, str]] = {}
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            stats[parts[2].replace("\\", "/")] = {"insertions": parts[0], "deletions": parts[1]}
    return stats


def main() -> None:
    notebook_paths = sorted(
        path
        for path in REPO.rglob("*.ipynb")
        if not any(part in EXCLUDED_PARTS for part in path.parts)
    )
    tracked = get_tracked_notebooks()
    modified = get_modified_notebooks()
    numstat = get_numstat()

    profiles: dict[str, dict] = {}
    head_profiles: dict[str, dict] = {}
    before_hashes: dict[str, str] = {}
    parse_errors: list[dict] = []
    file_inventory: list[dict] = []

    for path in notebook_paths:
        path_rel = rel(path)
        raw = path.read_bytes()
        before_hashes[path_rel] = sha_bytes(raw)
        current_nb, current_error = load_working_notebook(path)
        head_nb = None
        head_error = ""

        if current_nb is not None:
            profiles[path_rel] = extract_profile(current_nb)
        else:
            parse_errors.append(
                {
                    "notebook_path": str(path),
                    "relative_notebook_path": path_rel,
                    "version": "working_tree",
                    "error": current_error,
                }
            )

        if path_rel in tracked:
            head_nb, head_error = load_head_notebook(path_rel)
            if head_nb is not None:
                head_profiles[path_rel] = extract_profile(head_nb)
            else:
                parse_errors.append(
                    {
                        "notebook_path": str(path),
                        "relative_notebook_path": path_rel,
                        "version": "HEAD",
                        "error": head_error,
                    }
                )

        current_profile = profiles.get(path_rel)
        head_profile = head_profiles.get(path_rel)
        file_inventory.append(
            {
                "notebook_path": str(path),
                "relative_notebook_path": path_rel,
                "tracked_in_git": path_rel in tracked,
                "modified_vs_head_before_audit": path_rel in modified,
                "file_size_bytes": len(raw),
                "sha256_before": before_hashes[path_rel],
                "working_tree_parse_error": current_error,
                "head_parse_error": head_error,
                "current_cell_count": current_profile["cell_count"] if current_profile else "",
                "current_code_cells": current_profile["code_count"] if current_profile else "",
                "current_markdown_cells": current_profile["markdown_count"] if current_profile else "",
                "current_function_defs": len(current_profile["functions"]) if current_profile else "",
                "head_cell_count": head_profile["cell_count"] if head_profile else "",
                "head_function_defs": len(head_profile["functions"]) if head_profile else "",
            }
        )

    findings: list[dict] = []
    function_inventory: list[dict] = []
    cell_order_diffs: list[dict] = []
    git_diff_summary: list[dict] = []

    for path_rel, profile in profiles.items():
        path = REPO / path_rel
        modified_here = path_rel in modified
        cell_count = profile["cell_count"]
        first_dataset = profile["first_dataset_cell"]

        for f in profile["functions"]:
            function_inventory.append(
                {
                    "notebook_path": str(path),
                    "relative_notebook_path": path_rel,
                    "function_name": f["name"],
                    "cell_number_1_based": f["cell"] + 1,
                    "cell_index_0_based": f["cell"],
                    "position_ratio": round(f["ratio"], 4),
                    "line_in_cell": f["line"],
                    "preceding_heading": f["heading"],
                    "cell_stage": f["stage"],
                    "cell_line_count": f["cell_lines"],
                    "first_use_cell_after_def_1_based": f["first_use_after_def"] + 1
                    if isinstance(f["first_use_after_def"], int)
                    else "",
                    "cell_gap_to_first_use": f["gap_to_first_use"],
                    "call_count_excluding_def_lines": f["call_count"],
                    "modified_vs_head_before_audit": modified_here,
                }
            )

        defs_by_cell: dict[int, list[dict]] = defaultdict(list)
        for f in profile["functions"]:
            defs_by_cell[f["cell"]].append(f)

        for cell_index, funcs in defs_by_cell.items():
            ratio = cell_index / max(1, cell_count - 1)
            early = ratio <= 0.25 or cell_index <= 8
            before_dataset = first_dataset is not None and cell_index < first_dataset
            names = [f["name"] for f in funcs]
            far_use = [
                f
                for f in funcs
                if isinstance(f["gap_to_first_use"], int) and f["gap_to_first_use"] >= 8
            ]
            later_named = [name for name in names if expected_later_function(name)]
            heading = profile["functions"][profile["functions"].index(funcs[0])]["heading"]
            stage = profile["cells"][cell_index]["stage"]

            if early and len(funcs) >= 3 and (before_dataset or far_use or later_named):
                findings.append(
                    {
                        "notebook_path": str(path),
                        "relative_notebook_path": path_rel,
                        "cell_number_1_based": cell_index + 1,
                        "cell_index_0_based": cell_index,
                        "finding_type": "early_dense_function_block",
                        "severity": "high" if modified_here and (before_dataset or far_use) else "medium",
                        "function_names": "; ".join(names),
                        "evidence": one_line(
                            f"{len(funcs)} function definitions appear early at cell {cell_index + 1}/{cell_count}; "
                            f"heading={heading or '(none)'}; stage={stage}; "
                            f"first dataset cell={(first_dataset + 1) if first_dataset is not None else 'unknown'}; "
                            f"far-use functions={[f['name'] for f in far_use]}"
                        ),
                        "recommended_action": "Review whether this helper block belongs near later data-preparation/evaluation cells. If it was moved for layout, restore top-to-bottom experimental order while preserving outputs.",
                    }
                )

            for f in funcs:
                if (
                    early
                    and expected_later_function(f["name"])
                    and isinstance(f["gap_to_first_use"], int)
                    and f["gap_to_first_use"] >= 10
                ):
                    findings.append(
                        {
                            "notebook_path": str(path),
                            "relative_notebook_path": path_rel,
                            "cell_number_1_based": cell_index + 1,
                            "cell_index_0_based": cell_index,
                            "finding_type": "function_hoisted_far_before_first_use",
                            "severity": "high" if modified_here else "medium",
                            "function_names": f["name"],
                            "evidence": one_line(
                                f"Function `{f['name']}` is defined at cell {cell_index + 1}, "
                                f"but the first later call appears at cell {f['first_use_after_def'] + 1}; "
                                f"gap={f['gap_to_first_use']} cells; heading={heading or '(none)'}."
                            ),
                            "recommended_action": "Verify whether this function was moved upward. If so, move it back near first use or restore the original cell order from Git.",
                        }
                    )

        for heading in profile["headings"]:
            index = heading["cell"]
            ratio = index / max(1, cell_count - 1)
            if ratio <= 0.25 and HELPER_HEADING_RE.search(heading["heading"]):
                findings.append(
                    {
                        "notebook_path": str(path),
                        "relative_notebook_path": path_rel,
                        "cell_number_1_based": index + 1,
                        "cell_index_0_based": index,
                        "finding_type": "early_utility_section_heading",
                        "severity": "medium" if modified_here else "low",
                        "function_names": "",
                        "evidence": one_line(
                            f"Heading `{heading['heading']}` appears in the first quarter of the notebook."
                        ),
                        "recommended_action": "Review nearby cells. Early utility sections can be legitimate, but they can also indicate aesthetic regrouping.",
                    }
                )

        if path_rel in head_profiles:
            head = head_profiles[path_rel]
            added_cells = 0
            removed_cells = 0
            moved_cells = 0
            function_moves: list[str] = []

            for h, current_indices in profile["hash_positions"].items():
                if h not in head["hash_positions"]:
                    added_cells += len(current_indices)
                    continue
                for current_index in current_indices:
                    old_index = min(head["hash_positions"][h], key=lambda i: abs(i - current_index))
                    if old_index != current_index:
                        moved_cells += 1
                        cell_order_diffs.append(
                            {
                                "notebook_path": str(path),
                                "relative_notebook_path": path_rel,
                                "cell_source_hash": h,
                                "head_cell_number_1_based": old_index + 1,
                                "current_cell_number_1_based": current_index + 1,
                                "delta_cells_current_minus_head": current_index - old_index,
                                "moved_direction": "earlier" if current_index < old_index else "later",
                                "current_cell_type": profile["cells"][current_index]["type"],
                                "current_stage": profile["cells"][current_index]["stage"],
                                "current_excerpt": one_line(profile["cells"][current_index]["text"]),
                            }
                        )

            for h, old_indices in head["hash_positions"].items():
                if h not in profile["hash_positions"]:
                    removed_cells += len(old_indices)

            for name, current_positions in profile["function_positions"].items():
                current_min = min(current_positions)
                if name in head["function_positions"]:
                    head_min = min(head["function_positions"][name])
                    delta = current_min - head_min
                    current_ratio = current_min / max(1, profile["cell_count"] - 1)
                    head_ratio = head_min / max(1, head["cell_count"] - 1)
                    ratio_delta = current_ratio - head_ratio
                    if delta <= -3 or ratio_delta <= -0.15:
                        severity = "high" if delta <= -8 or ratio_delta <= -0.30 else "medium"
                        function_moves.append(f"{name}:HEAD{head_min + 1}->CUR{current_min + 1}")
                        findings.append(
                            {
                                "notebook_path": str(path),
                                "relative_notebook_path": path_rel,
                                "cell_number_1_based": current_min + 1,
                                "cell_index_0_based": current_min,
                                "finding_type": "function_moved_earlier_vs_head",
                                "severity": severity,
                                "function_names": name,
                                "evidence": one_line(
                                    f"Function `{name}` appears at current cell {current_min + 1}; "
                                    f"in HEAD it appears at cell {head_min + 1}; moved {abs(delta)} cells earlier; "
                                    f"position-ratio delta={ratio_delta:.2f}."
                                ),
                                "recommended_action": "Inspect the Git diff for this notebook. If the movement was not intentional, restore the original cell order from HEAD or move the function back near first use.",
                            }
                        )
                elif modified_here and current_min / max(1, profile["cell_count"] - 1) <= 0.25:
                    if expected_later_function(name):
                        findings.append(
                            {
                                "notebook_path": str(path),
                                "relative_notebook_path": path_rel,
                                "cell_number_1_based": current_min + 1,
                                "cell_index_0_based": current_min,
                                "finding_type": "new_early_function_vs_head",
                                "severity": "medium",
                                "function_names": name,
                                "evidence": one_line(
                                    f"Function `{name}` is present early in the working tree but was not found in HEAD for this notebook."
                                ),
                                "recommended_action": "Check whether this new function is necessary here; if it duplicates later logic or was hoisted for style, relocate or revert the structural change.",
                            }
                        )

            git_diff_summary.append(
                {
                    "notebook_path": str(path),
                    "relative_notebook_path": path_rel,
                    "tracked_in_git": path_rel in tracked,
                    "modified_vs_head_before_audit": modified_here,
                    "head_cell_count": head["cell_count"],
                    "current_cell_count": profile["cell_count"],
                    "cell_count_delta": profile["cell_count"] - head["cell_count"],
                    "head_function_defs": len(head["functions"]),
                    "current_function_defs": len(profile["functions"]),
                    "function_def_delta": len(profile["functions"]) - len(head["functions"]),
                    "exact_source_cells_added": added_cells,
                    "exact_source_cells_removed": removed_cells,
                    "exact_source_cells_moved": moved_cells,
                    "functions_moved_earlier_vs_head": "; ".join(function_moves),
                    "git_insertions": numstat.get(path_rel, {}).get("insertions", ""),
                    "git_deletions": numstat.get(path_rel, {}).get("deletions", ""),
                }
            )

    for error in parse_errors:
        if error["version"] != "working_tree":
            continue
        path = Path(error["notebook_path"])
        size = path.stat().st_size if path.exists() else ""
        findings.append(
            {
                "notebook_path": error["notebook_path"],
                "relative_notebook_path": error["relative_notebook_path"],
                "cell_number_1_based": "",
                "cell_index_0_based": "",
                "finding_type": "notebook_parse_error",
                "severity": "high" if isinstance(size, int) and size < 100 else "medium",
                "function_names": "",
                "evidence": one_line(
                    f"Working-tree notebook could not be parsed as JSON; size={size} bytes; error={error['error']}"
                ),
                "recommended_action": "Restore or replace this notebook before relying on it. If it is a placeholder, remove it from the submitted codebase or replace it with a valid notebook.",
            }
        )

    for summary in git_diff_summary:
        if not summary["modified_vs_head_before_audit"]:
            continue
        added = int(summary["git_insertions"]) if str(summary["git_insertions"]).isdigit() else 0
        deleted = int(summary["git_deletions"]) if str(summary["git_deletions"]).isdigit() else 0
        if added + deleted >= 800 and not summary["functions_moved_earlier_vs_head"]:
            findings.append(
                {
                    "notebook_path": summary["notebook_path"],
                    "relative_notebook_path": summary["relative_notebook_path"],
                    "cell_number_1_based": "",
                    "cell_index_0_based": "",
                    "finding_type": "large_notebook_diff_without_clear_function_move",
                    "severity": "medium",
                    "function_names": "",
                    "evidence": one_line(
                        f"Git diff is large ({added} insertions, {deleted} deletions), but exact function movement was not detected by source/name matching."
                    ),
                    "recommended_action": "Review with notebook diff tooling. Large JSON diffs can hide cell reorder, metadata churn, or output changes.",
                }
            )

    seen = set()
    deduped = []
    for item in sorted(
        findings,
        key=lambda r: (
            severity_key(r["severity"]),
            r["relative_notebook_path"],
            str(r["cell_index_0_based"]),
            r["finding_type"],
            r["function_names"],
        ),
    ):
        key = (
            item["relative_notebook_path"],
            str(item["cell_index_0_based"]),
            item["finding_type"],
            item["function_names"],
            item["evidence"],
        )
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    findings = deduped

    function_inventory.sort(
        key=lambda r: (
            r["relative_notebook_path"],
            int(r["cell_index_0_based"]),
            r["function_name"],
        )
    )
    cell_order_diffs.sort(
        key=lambda r: (
            r["relative_notebook_path"],
            int(r["current_cell_number_1_based"]),
            int(r["head_cell_number_1_based"]),
        )
    )
    git_diff_summary.sort(
        key=lambda r: (
            not r["modified_vs_head_before_audit"],
            r["relative_notebook_path"],
        )
    )

    findings_csv = OUT / "notebook_structure_anomaly_findings.csv"
    function_csv = OUT / "function_position_inventory.csv"
    cell_diff_csv = OUT / "notebook_git_cell_order_diff.csv"
    git_summary_csv = OUT / "notebook_git_diff_summary.csv"
    parse_csv = OUT / "notebook_parse_errors.csv"
    inventory_csv = OUT / "notebook_audit_file_inventory.csv"
    summary_md = OUT / "notebook_structure_anomaly_audit_summary.md"
    metadata_json = OUT / "audit_run_metadata.json"

    write_csv(
        findings_csv,
        findings,
        [
            "notebook_path",
            "relative_notebook_path",
            "cell_number_1_based",
            "cell_index_0_based",
            "finding_type",
            "severity",
            "function_names",
            "evidence",
            "recommended_action",
        ],
    )
    write_csv(
        function_csv,
        function_inventory,
        [
            "notebook_path",
            "relative_notebook_path",
            "function_name",
            "cell_number_1_based",
            "cell_index_0_based",
            "position_ratio",
            "line_in_cell",
            "preceding_heading",
            "cell_stage",
            "cell_line_count",
            "first_use_cell_after_def_1_based",
            "cell_gap_to_first_use",
            "call_count_excluding_def_lines",
            "modified_vs_head_before_audit",
        ],
    )
    write_csv(
        cell_diff_csv,
        cell_order_diffs,
        [
            "notebook_path",
            "relative_notebook_path",
            "cell_source_hash",
            "head_cell_number_1_based",
            "current_cell_number_1_based",
            "delta_cells_current_minus_head",
            "moved_direction",
            "current_cell_type",
            "current_stage",
            "current_excerpt",
        ],
    )
    write_csv(
        git_summary_csv,
        git_diff_summary,
        [
            "notebook_path",
            "relative_notebook_path",
            "tracked_in_git",
            "modified_vs_head_before_audit",
            "head_cell_count",
            "current_cell_count",
            "cell_count_delta",
            "head_function_defs",
            "current_function_defs",
            "function_def_delta",
            "exact_source_cells_added",
            "exact_source_cells_removed",
            "exact_source_cells_moved",
            "functions_moved_earlier_vs_head",
            "git_insertions",
            "git_deletions",
        ],
    )
    write_csv(parse_csv, parse_errors, ["notebook_path", "relative_notebook_path", "version", "error"])
    write_csv(
        inventory_csv,
        file_inventory,
        [
            "notebook_path",
            "relative_notebook_path",
            "tracked_in_git",
            "modified_vs_head_before_audit",
            "file_size_bytes",
            "sha256_before",
            "working_tree_parse_error",
            "head_parse_error",
            "current_cell_count",
            "current_code_cells",
            "current_markdown_cells",
            "current_function_defs",
            "head_cell_count",
            "head_function_defs",
        ],
    )

    changed_by_audit = []
    for path in notebook_paths:
        path_rel = rel(path)
        if sha_bytes(path.read_bytes()) != before_hashes[path_rel]:
            changed_by_audit.append(path_rel)

    severity_counts = Counter(item["severity"] for item in findings)
    type_counts = Counter(item["finding_type"] for item in findings)
    affected_notebooks = sorted({item["relative_notebook_path"] for item in findings})
    modified_notebooks = sorted(modified)

    metadata = {
        "run_timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO),
        "audit_scope": "All .ipynb files under the repo, excluding audit output folders and .ipynb_checkpoints.",
        "notebooks_checked": len(notebook_paths),
        "tracked_notebooks_checked": len([p for p in notebook_paths if rel(p) in tracked]),
        "modified_notebooks_before_audit": modified_notebooks,
        "modified_notebook_count_before_audit": len(modified_notebooks),
        "findings_total": len(findings),
        "affected_notebook_count": len(affected_notebooks),
        "severity_counts": dict(severity_counts),
        "finding_type_counts": dict(type_counts),
        "parse_error_count": len(parse_errors),
        "changed_notebooks_by_audit": changed_by_audit,
        "notebook_outputs_executed_or_modified": False,
        "outputs_created": [
            str(summary_md),
            str(findings_csv),
            str(function_csv),
            str(cell_diff_csv),
            str(git_summary_csv),
            str(parse_csv),
            str(inventory_csv),
            str(metadata_json),
        ],
        "limitations": [
            "The audit detects structural anomalies and Git-diff evidence, but cannot prove who made a change or why.",
            "Notebook JSON diffs can be noisy when outputs, metadata, or line endings change.",
            "Early helper functions are not automatically wrong; flagged items require manual confirmation before editing.",
        ],
    }

    with metadata_json.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    lines = [
        "# Notebook Structure Anomaly Audit",
        "",
        f"Repository: `{REPO}`",
        f"Run time: `{metadata['run_timestamp_local']}`",
        "",
        "## What This Audit Can And Cannot Say",
        "",
        "This audit looks for structural signs consistent with notebook rearrangement, especially helper/function blocks moved earlier than their natural point of use. It does not identify who made a change or prove intent.",
        "",
        "## Scope And Integrity",
        "",
        f"- Notebooks checked: {len(notebook_paths)}",
        f"- Tracked notebooks checked: {metadata['tracked_notebooks_checked']}",
        f"- Modified notebooks before audit: {len(modified_notebooks)}",
        f"- Findings total: {len(findings)}",
        f"- Affected notebooks: {len(affected_notebooks)}",
        f"- High severity: {severity_counts.get('high', 0)}",
        f"- Medium severity: {severity_counts.get('medium', 0)}",
        f"- Low severity: {severity_counts.get('low', 0)}",
        f"- Parse errors: {len(parse_errors)}",
        f"- Notebook files changed by this audit: {len(changed_by_audit)}",
        "- Notebooks were not executed. Existing notebook outputs were not cleared or modified.",
        "",
        "## Output Files",
        "",
    ]
    for path in [findings_csv, function_csv, cell_diff_csv, git_summary_csv, parse_csv, inventory_csv, metadata_json]:
        lines.append(f"- `{path}`")

    lines += ["", "## Modified Notebooks Before Audit", ""]
    if modified_notebooks:
        diff_by_path = {item["relative_notebook_path"]: item for item in git_diff_summary}
        lines.append("| Notebook | Git insertions | Git deletions | Cell delta | Function delta | Functions moved earlier vs HEAD |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for path_rel in modified_notebooks:
            item = diff_by_path.get(path_rel, {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        markdown_escape(path_rel),
                        markdown_escape(item.get("git_insertions", "")),
                        markdown_escape(item.get("git_deletions", "")),
                        markdown_escape(item.get("cell_count_delta", "")),
                        markdown_escape(item.get("function_def_delta", "")),
                        markdown_escape(item.get("functions_moved_earlier_vs_head", "")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No tracked notebooks were modified before this audit.")

    priority = [item for item in findings if item["severity"] in {"high", "medium"}][:100]
    lines += ["", "## Highest Priority Findings", ""]
    if priority:
        lines.append("| Severity | Type | Notebook | Cell | Functions | Evidence | Recommended action |")
        lines.append("|---|---|---|---:|---|---|---|")
        for item in priority:
            lines.append(
                "| "
                + " | ".join(
                    [
                        markdown_escape(item["severity"]),
                        markdown_escape(item["finding_type"]),
                        markdown_escape(item["relative_notebook_path"]),
                        markdown_escape(item["cell_number_1_based"]),
                        markdown_escape(item["function_names"]),
                        markdown_escape(item["evidence"]),
                        markdown_escape(item["recommended_action"]),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No high/medium priority structural anomalies were found.")

    lines += ["", "## Affected Notebook Summary", ""]
    if findings:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for item in findings:
            grouped[item["relative_notebook_path"]].append(item)
        lines.append("| Notebook | High | Medium | Low | Main finding types |")
        lines.append("|---|---:|---:|---:|---|")
        for path_rel, items in sorted(
            grouped.items(),
            key=lambda pair: (
                min(severity_key(item["severity"]) for item in pair[1]),
                pair[0],
            ),
        ):
            sev = Counter(item["severity"] for item in items)
            types = Counter(item["finding_type"] for item in items).most_common(5)
            lines.append(
                "| "
                + " | ".join(
                    [
                        markdown_escape(path_rel),
                        str(sev.get("high", 0)),
                        str(sev.get("medium", 0)),
                        str(sev.get("low", 0)),
                        markdown_escape("; ".join(f"{name}={count}" for name, count in types)),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No affected notebooks were found.")

    lines += ["", "## Parse Errors", ""]
    if parse_errors:
        lines.append("| Notebook | Version | Error |")
        lines.append("|---|---|---|")
        for item in parse_errors:
            lines.append(
                "| "
                + " | ".join(
                    [
                        markdown_escape(item["relative_notebook_path"]),
                        markdown_escape(item["version"]),
                        markdown_escape(item["error"]),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No notebook parse errors were found.")

    lines += [
        "",
        "## Suggested Review Order",
        "",
        "1. Start with high-severity rows in `notebook_structure_anomaly_findings.csv`.",
        "2. Review the tracked notebooks that were already modified versus Git `HEAD`, because they are the strongest objective signal of recent structural changes.",
        "3. Use `function_position_inventory.csv` to inspect functions with large `cell_gap_to_first_use` values.",
        "4. Confirm with a visual notebook diff before moving any cells; early helper sections can be legitimate in some notebook styles.",
        "",
        "## Integrity Confirmation",
        "",
    ]
    if changed_by_audit:
        lines.append("WARNING: one or more notebook hashes changed during the audit. Review `audit_run_metadata.json` before continuing.")
    else:
        lines.append("All notebook SHA-256 hashes matched before and after the audit. The audit created only output files and did not alter notebook contents or outputs.")

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
