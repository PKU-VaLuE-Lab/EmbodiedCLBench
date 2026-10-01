#!/usr/bin/env python3
"""Audit completed analysis runs and combine them with the canonical main run."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def task_paths(job, task_id):
    output = Path(job["output_dir"])
    mode = job.get("mode")
    if mode is None:
        run = output / "task_runs" / task_id
        return run, run, [run]
    run = output / task_id
    app = run / "skill_application" if mode == "skill" else run
    scored = app / "task_runs" / ("01_" + task_id)
    phases = [app, run / "skill_learning"] if mode == "skill" else [app]
    return scored, app, phases


def read_metrics(score_path):
    score = read_json(score_path)
    grade = read_json(score_path.parent / "task_output" / "grade.json")
    reached = grade["reached_goal"]
    check(isinstance(reached, bool), f"Invalid reached_goal: {score_path}")
    check(bool(score["SR"]) == reached, f"SR/grade disagreement: {score_path}")
    shortest = int(grade["shortest_success_trace_length"])
    total = int(grade["valid_action_count"]) + int(grade["invalid_action_count"])
    check(shortest > 0, f"Invalid shortest path: {score_path}")
    check(total == int(grade["total_step_count"]), f"Step count disagreement: {score_path}")
    extra = total - shortest if reached else None
    if reached:
        check(extra >= 0, f"Success shorter than reference: {score_path}")
        check(extra == grade["extra_steps"] == score["ER"], f"ER disagreement: {score_path}")
    return {"reached_goal": reached, "shortest_steps": shortest,
            "total_steps": total, "actual_extra_steps": extra}


def validate_learning(prepared, main_bundle):
    main = {r["target_task_id"]: r["learning_task_ids"] for r in
            read_json(main_bundle / "manifest/portable_learning_reuse/hermesagent.json")["entries"]}
    plans = {}
    for k in (1, 3, 4):
        data = read_json(prepared / f"n_sample/learning_plan_k{k}.json")
        plans[k] = {r["target_task_id"]: r["learning_task_ids"] for r in data["targets"]}
    check(set(plans[1]) == set(plans[3]) == set(plans[4]), "N-sample target mismatch")
    for tid in plans[1]:
        ordered = plans[4][tid]
        check(len(ordered) == len(set(ordered)) == 4, f"Invalid k4: {tid}")
        check(plans[1][tid] == ordered[:1] and plans[3][tid] == ordered[:3]
              and main[tid] == ordered[:2], f"Learning sets are not nested: {tid}")
    ln = read_json(prepared / "l_n/learning_plan.json")["targets"]
    for row in ln:
        check(row["learning_task_ids"] == main[row["source_parent_task_id"]]
              and len(row["learning_task_ids"]) == 2,
              f"L_N learning mismatch: {row['target_task_id']}")
    l4 = {r["task_id"]: r for r in read_json(prepared / "n_sample/ready_l4_tasks.json")}
    parents = {r["target_task_id"]: r["source_parent_task_id"] for r in ln}
    levels = {k: {r["task_id"].rsplit("_", 1)[1]: r for r in
                  read_json(prepared / f"l_n/ready_l{k}_tasks.json")} for k in (3, 5, 6)}
    chains = []
    for unit, l3 in levels[3].items():
        parent = parents[l3["task_id"]]
        components = {4: {lid + "__scene_" + l4[parent]["scene_id"]
                          for lid in l4[parent]["source_l1_task_ids"]}}
        for level in (3, 5, 6):
            row = levels[level][unit]
            check(parents[row["task_id"]] == parent, f"Inconsistent chain parent: {unit}")
            components[level] = set(row["task"]["composition"]["source_l1_task_ids"])
        check(all(len(components[k]) == k for k in (3, 4, 5, 6)) and
              all(components[k] < components[k + 1] for k in (3, 4, 5)),
              f"Non-nested action chain: {unit}")
        chains.append({"analysis_unit_id": unit, "parent_l4_task_id": parent,
                       "l3_task_id": l3["task_id"]})
    return {"n_sample_nested_targets": len(plans[1]), "l_n_fixed_k2_targets": len(ln),
            "l_n_nested_chains": len(chains),
            "l_n_unique_parent_l4_count": len({r["parent_l4_task_id"] for r in chains}),
            "l_n_parent_weighting": "One matched L4 result per existing chain; repeated parents are reused observations, not independent runs.",
            "l_n_chains": chains}


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["experiment"], row["axis"], row["mode"])].append(row)
    result = []
    for (experiment, axis, mode), items in sorted(groups.items()):
        check(len(items) == len({r["analysis_unit_id"] for r in items}) == 50,
              f"Incomplete/duplicate group: {experiment}/{axis}/{mode}")
        for budget in range(7):
            successes = [r for r in items if r["reached_goal"]
                         and r["actual_extra_steps"] <= budget]
            result.append({"experiment": experiment, "axis": axis, "mode": mode,
                           "extra_budget": budget, "n": len(items), "successes": len(successes),
                           "sr_percent": 100 * len(successes) / len(items),
                           "er": sum(r["actual_extra_steps"] for r in successes) / len(successes)
                           if successes else None})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--main-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    manifest = read_json(args.run_root / "run_manifest.json")
    check(manifest["model"] == "qwen3.8-flash" and manifest["context_mode"] == "task_compact",
          "Unexpected model/context setting")
    check(manifest["extra_steps"] == 6 and len(manifest["jobs"]) == 15, "Unexpected run scale")
    learning_audit = validate_learning(args.prepared_root, args.main_bundle)
    rows, usage_rows, behavior_rows, errors, jobs = [], [], [], [], []
    targets = set()
    for job in manifest["jobs"]:
        output = Path(job["output_dir"])
        record = read_json(output / "job.json") if (output / "job.json").exists() else {}
        count = 0
        mode = job.get("mode") or "zero_shot"
        axis = int(output.parent.name[1:])
        for tid in job["selected_task_ids"]:
            scored, app, phases = task_paths(job, tid)
            score_path = scored / "score.json"
            if not score_path.exists():
                continue
            try:
                metric = read_metrics(score_path)
                step_limit = metric["shortest_steps"] + int(manifest["extra_steps"])
                behavior_rows.append({
                    "job": job["label"],
                    "task_id": tid,
                    "outcome": (
                        "success"
                        if metric["reached_goal"]
                        else "stopped_before_budget"
                        if metric["total_steps"] < step_limit
                        else "failed_at_budget"
                    ),
                    "total_steps": metric["total_steps"],
                    "step_limit": step_limit,
                    "unused_steps": max(0, step_limit - metric["total_steps"]),
                    "score_path": str(score_path),
                })
                for phase in phases:
                    primary = read_json(phase / "primary_result.json")
                    check(not primary.get("error"), f"API error: {primary.get('error')}")
                    usage = read_json(phase / "usage.json")
                    check(usage.get("usage_complete") is True, f"Incomplete usage: {phase}")
                    check(usage["input_tokens"] + usage["output_tokens"] == usage["total_tokens"],
                          f"Usage total disagreement: {phase}")
                    usage_rows.append({"job": job["label"], "task_id": tid,
                                       "phase": "skill_summary" if phase.name == "skill_learning" else "application",
                                       **{k: usage.get(k, 0) for k in ("input_tokens", "output_tokens",
                                           "cache_read_tokens", "uncached_input_tokens", "request_count")},
                                       "source_path": str(phase / "usage.json")})
                unit = tid.rsplit("_", 1)[1] if job["experiment"] == "l_n" else tid
                rows.append({"experiment": job["experiment"], "axis": axis, "mode": mode,
                             "task_id": tid, "analysis_unit_id": unit,
                             "source": "new_analysis_eval", "score_path": str(score_path), **metric})
                count += 1
                if job["experiment"] == "n_sample":
                    targets.add(tid)
            except (ValueError, KeyError, FileNotFoundError) as exc:
                errors.append({"job": job["label"], "task_id": tid, "error": str(exc)})
        jobs.append({"job": job["label"], "valid_results": count, "expected": 50,
                     "returncode": record.get("returncode")})
    audit = {"jobs": jobs, "new_result_count": len(rows), "expected_new_results": 750,
             "errors": errors, "learning": learning_audit,
             "complete": not errors and all(j["valid_results"] == 50 and j["returncode"] == 0 for j in jobs)}
    write_json(args.output_dir / "completion_audit.json", audit)
    if args.audit_only:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return
    check(audit["complete"], "Incomplete run; inspect completion_audit.json")
    with (args.main_bundle / "reports/full_100/task_metrics.csv").open(newline="", encoding="utf-8") as handle:
        main_rows = [r for r in csv.DictReader(handle) if r["harness"] == "hermesagent" and r["task_id"] in targets
                     and r["setting"] in ("zero_shot_l4", "in_context_l4", "skill_l4")]
    check(len(main_rows) == 150, "Main run does not cover all selected child L4 conditions")
    reused = {}
    for r in main_rows:
        score_path = args.repo_root / r["score_path"]
        metric = read_metrics(score_path)
        check(metric["reached_goal"] == (r["reached_goal"] == "True"), "Main SR report mismatch")
        check(metric["total_steps"] == int(r["total_steps"]), "Main step report mismatch")
        setting = r["setting"]
        mode = setting[:-3] if setting.endswith("_l4") else setting
        common = {"mode": mode, "task_id": r["task_id"], "source": "canonical_main_eval",
                  "score_path": str(score_path), **metric}
        reused[(r["task_id"], mode)] = common
        rows.append({"experiment": "n_sample", "axis": 0 if mode == "zero_shot" else 2,
                     "analysis_unit_id": r["task_id"], **common})
    for chain in learning_audit["l_n_chains"]:
        for mode in ("zero_shot", "in_context", "skill"):
            common = reused[(chain["parent_l4_task_id"], mode)]
            rows.append({"experiment": "l_n", "axis": 4,
                         "analysis_unit_id": chain["analysis_unit_id"], **common})
    check(len(rows) == 1050, "Unexpected result count including reused conditions")
    summary = aggregate(rows)
    write_csv(args.output_dir / "task_metrics.csv", rows)
    write_csv(args.output_dir / "budget_curves_extra0_to6.csv", summary)
    write_csv(args.output_dir / "extra3.csv", [r for r in summary if r["extra_budget"] == 3])
    write_csv(args.output_dir / "provider_usage_by_phase.csv", usage_rows)
    behavior_counts = defaultdict(int)
    for row in behavior_rows:
        behavior_counts[(row["job"], row["outcome"])] += 1
    write_json(args.output_dir / "behavioral_completion_audit.json", {
        "definition": "A stopped_before_budget result is an unsuccessful task whose recorded action count is below shortest_steps + configured extra_steps. It remains a model failure and is not retried.",
        "counts": [
            {"job": job, "outcome": outcome, "count": count}
            for (job, outcome), count in sorted(behavior_counts.items())
        ],
        "stopped_before_budget_count": sum(
            row["outcome"] == "stopped_before_budget" for row in behavior_rows
        ),
        "records": behavior_rows,
    })
    write_json(args.output_dir / "provider_usage.json", {
        "scope": "New formal analysis applications and skill summaries only; excludes smoke and reused main/L2 calls.",
        "usage_file_count": len(usage_rows),
        **{k: sum(r[k] for r in usage_rows) for k in ("input_tokens", "output_tokens", "cache_read_tokens", "uncached_input_tokens", "request_count")}})
    for budget in range(7):
        ns = [{"k": r["axis"], "mode": r["mode"], "successes": r["successes"], "n": r["n"],
               "sr_percent": r["sr_percent"]} for r in summary if r["experiment"] == "n_sample" and r["extra_budget"] == budget]
        write_csv(args.output_dir / "n_sample" / f"n_sample_extra_steps_{budget}.csv", ns)
        by_key = {(r["axis"], r["mode"]): r["sr_percent"] for r in summary
                  if r["experiment"] == "l_n" and r["extra_budget"] == budget}
        ln = [{"action_count": level, "zero_shot_sr": by_key[level, "zero_shot"],
               "icl_sr": by_key[level, "in_context"], "skill_sr": by_key[level, "skill"]} for level in (3, 4, 5, 6)]
        write_csv(args.output_dir / "L_N" / f"complexity_sr_extra{budget}.csv", ln)
    audit.update({"reused_main_unique_results": 150, "report_rows_including_reuse": len(rows)})
    write_json(args.output_dir / "completion_audit.json", audit)
    print(json.dumps({"complete": True, "new_results": 750, "reused_main_results": 150,
                      "report_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
