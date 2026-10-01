from __future__ import annotations

from typing import Any

from tongbench_eval.graph_env.io.predicate_normalizer import normalize_predicate, normalize_predicate_set


def _extract_args(predicate: str) -> list[str]:
    if "(" not in predicate or ")" not in predicate:
        return []
    return [item.strip() for item in predicate.split("(", 1)[1].rsplit(")", 1)[0].split(",") if item.strip()]


def _holding_predicates(predicates: set[str]) -> list[str]:
    return sorted(item for item in predicates if item.startswith("holding("))


def _with_reach_predicates(predicates: set[str]) -> list[str]:
    return sorted(item for item in predicates if item.startswith("with_reach("))


def _location_predicates(predicates: set[str], prefix: str, object_id: str) -> list[str]:
    output: list[str] = []
    for predicate in predicates:
        if not predicate.startswith(f"{prefix}("):
            continue
        args = _extract_args(predicate)
        if args and args[0] == object_id:
            output.append(predicate)
    return sorted(output)


def _record_rule(
    rules: list[dict[str, Any]],
    rule: str,
    reason: str,
    deleted: set[str],
    added: set[str],
) -> None:
    if not deleted and not added:
        return
    rules.append(
        {
            "rule": rule,
            "reason": reason,
            "deleted": normalize_predicate_set(sorted(deleted)),
            "added": normalize_predicate_set(sorted(added)),
        }
    )


def apply_state_consistency_rules(
    previous_state_predicates: list[str],
    add_effects: list[str],
    delete_effects: list[str],
    selected_action_candidate: dict,
    role_bindings: dict,
) -> dict:
    previous_state = set(normalize_predicate_set(previous_state_predicates))
    final_add = set(normalize_predicate_set(add_effects))
    final_delete = set(normalize_predicate_set(delete_effects))
    consistency_adds: set[str] = set()
    consistency_deletes: set[str] = set()
    rules_applied: list[dict[str, Any]] = []

    action_type = str(selected_action_candidate.get("action_type", ""))
    template_id = str(selected_action_candidate.get("template_id", ""))
    exclusive_in_view = bool(selected_action_candidate.get("_exclusive_in_view", True))

    def add_rule(rule: str, reason: str, deleted: set[str] | None = None, added: set[str] | None = None) -> None:
        deleted = set() if deleted is None else {normalize_predicate(item) for item in deleted}
        added = set() if added is None else {normalize_predicate(item) for item in added}
        consistency_deletes.update(deleted)
        consistency_adds.update(added)
        final_delete.update(deleted)
        final_add.update(added)
        _record_rule(rules_applied, rule, reason, deleted, added)

    holding_adds = [predicate for predicate in sorted(final_add) if predicate.startswith("holding(")]
    if holding_adds:
        kept_holding = holding_adds[-1]
        kept_args = _extract_args(kept_holding)
        kept_object = kept_args[0] if kept_args else None
        deleted: set[str] = set()
        if "empty" in previous_state or "empty" in final_add:
            deleted.add("empty")
        for predicate in _holding_predicates(previous_state.union(final_add)):
            args = _extract_args(predicate)
            if args and args[0] != kept_object:
                deleted.add(predicate)
        add_rule(
            "holding_vs_empty",
            "Holding a new object removes empty and any previous held object.",
            deleted=deleted,
        )
        if kept_object:
            location_deletes = set(_location_predicates(previous_state, "on", kept_object) + _location_predicates(previous_state, "in", kept_object))
            add_rule(
                "acquire_location_cleanup",
                "Holding an object removes prior on/in locations for that object.",
                deleted=location_deletes,
            )

    if "empty" in final_add:
        holding_deletes = set(_holding_predicates(previous_state.union(final_add)))
        add_rule(
            "empty_clears_holding",
            "An empty hand state removes all holding predicates.",
            deleted=holding_deletes,
        )

    for predicate in list(sorted(final_add)):
        args = _extract_args(predicate)
        if predicate.startswith("clean(") and args:
            add_rule(
                "clean_dirty_mutex",
                "A clean object cannot remain dirty.",
                deleted={f"dirty({args[0]})"},
            )
        if predicate.startswith("dirty(") and args:
            add_rule(
                "dirty_clean_mutex",
                "A dirty object cannot remain clean.",
                deleted={f"clean({args[0]})"},
            )
        if predicate.startswith("open(") and args:
            add_rule(
                "open_closed_mutex",
                "An open object cannot remain closed.",
                deleted={f"closed({args[0]})"},
            )
        if predicate.startswith("closed(") and args:
            add_rule(
                "closed_open_mutex",
                "A closed object cannot remain open.",
                deleted={f"open({args[0]})"},
            )
        if predicate.startswith("powered_on(") and args:
            add_rule(
                "power_on_off_mutex",
                "A powered-on object cannot remain powered off.",
                deleted={f"powered_off({args[0]})"},
            )
        if predicate.startswith("powered_off(") and args:
            add_rule(
                "power_off_on_mutex",
                "A powered-off object cannot remain powered on.",
                deleted={f"powered_on({args[0]})"},
            )
        if predicate.startswith("plugged_in(") and args:
            add_rule(
                "plugged_unplugged_mutex",
                "A plugged-in object cannot remain unplugged.",
                deleted={f"unplugged({args[0]})"},
            )
        if predicate.startswith("unplugged(") and args:
            add_rule(
                "unplugged_plugged_mutex",
                "An unplugged object cannot remain plugged in.",
                deleted={f"plugged_in({args[0]})"},
            )

    with_reach_adds = [predicate for predicate in sorted(final_add) if predicate.startswith("with_reach(")]
    if with_reach_adds:
        kept_with_reach = with_reach_adds[-1]
        kept_args = _extract_args(kept_with_reach)
        kept_target = kept_args[0] if kept_args else None
        deleted: set[str] = set()
        for predicate in _with_reach_predicates(previous_state.union(final_add)):
            args = _extract_args(predicate)
            if args and args[0] != kept_target:
                deleted.add(predicate)
        add_rule(
            "with_reach_exclusive",
            "with_reach tracks the currently active reachable target.",
            deleted=deleted,
        )

    if exclusive_in_view:
        in_view_adds = [predicate for predicate in sorted(final_add) if predicate.startswith("in_view(")]
        if in_view_adds:
            kept_in_view = in_view_adds[-1]
            kept_args = _extract_args(kept_in_view)
            kept_target = kept_args[0] if kept_args else None
            deleted: set[str] = set()
            for predicate in sorted(previous_state.union(final_add)):
                if not predicate.startswith("in_view("):
                    continue
                args = _extract_args(predicate)
                if args and args[0] != kept_target:
                    deleted.add(predicate)
            add_rule(
                "in_view_exclusive",
                "exclusive_in_view keeps only the newest in_view target.",
                deleted=deleted,
            )

    placement_targets_to_keep: set[str] = set()
    for predicate in list(sorted(final_add)):
        args = _extract_args(predicate)
        if predicate.startswith("on(") and len(args) == 2:
            obj, target = args
            placement_targets_to_keep.add(f"with_reach({target})")
            deleted = set(_location_predicates(previous_state, "on", obj) + _location_predicates(previous_state, "in", obj))
            deleted.add(f"holding({obj})")
            deleted.add(f"with_reach({obj})")
            add_rule(
                "place_on_location_consistency",
                "Placing an object on a target clears holding and previous locations for that object.",
                deleted=deleted,
                added={"empty", f"with_reach({target})"},
            )
        if predicate.startswith("in(") and len(args) == 2:
            obj, target = args
            placement_targets_to_keep.add(f"with_reach({target})")
            deleted = set(_location_predicates(previous_state, "in", obj) + _location_predicates(previous_state, "on", obj))
            deleted.add(f"holding({obj})")
            deleted.add(f"with_reach({obj})")
            add_rule(
                "place_in_location_consistency",
                "Putting an object in a target clears holding and previous locations for that object.",
                deleted=deleted,
                added={"empty", f"with_reach({target})"},
            )

    if placement_targets_to_keep:
        stale_with_reach = {
            predicate
            for predicate in _with_reach_predicates(previous_state.union(final_add))
            if predicate not in placement_targets_to_keep
        }
        add_rule(
            "placement_reach_focus",
            "Placement keeps the target reachable and removes stale reach predicates.",
            deleted=stale_with_reach,
            added=placement_targets_to_keep,
        )

    # Reach while holding: preserve holding and only move the reach focus.
    if action_type == "reach" or template_id == "walk_to_{object}":
        # No extra deletes needed for holding; rule documented through reach focus.
        held_predicates = {predicate for predicate in previous_state if predicate.startswith("holding(")}
        if held_predicates:
            add_rule(
                "reach_while_holding",
                "Reaching a new target preserves the currently held object.",
                added=set(),
            )

    final_delete -= {predicate for predicate in final_add if predicate != "holding(*)"}

    return {
        "final_add_effects": normalize_predicate_set(sorted(final_add)),
        "final_delete_effects": normalize_predicate_set(sorted(final_delete)),
        "consistency_deletes": normalize_predicate_set(sorted(consistency_deletes)),
        "consistency_adds": normalize_predicate_set(sorted(consistency_adds)),
        "consistency_rules_applied": rules_applied,
    }
