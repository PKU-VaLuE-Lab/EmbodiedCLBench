from __future__ import annotations

import argparse
from pathlib import Path

from tongbench_eval.visualization.adapter import VisualizationInputs, load_viz_graph, write_viz_graph
from tongbench_eval.visualization.dash_app import serve_viz_graph
from tongbench_eval.visualization.exporters import (
    export_interactive_html,
    export_pipeline_mermaid,
    export_task_dot,
    render_dot_to_svg,
    write_summary_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize TongBench task graphs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export static graph artifacts and viz_graph.json.")
    export_parser.add_argument("--graph", required=True, type=Path)
    export_parser.add_argument("--success-paths", type=Path, default=None)
    export_parser.add_argument("--atomic-transition-manifest", type=Path, default=None)
    export_parser.add_argument("--unique-states", type=Path, default=None)
    export_parser.add_argument("--full-branch-graph", type=Path, default=None)
    export_parser.add_argument("--image-manifest", type=Path, default=None)
    export_parser.add_argument("--trace", type=Path, default=None)
    export_parser.add_argument("--out-dir", required=True, type=Path)
    export_parser.add_argument("--render-svg", action="store_true")

    serve_parser = subparsers.add_parser("serve", help="Serve an interactive Dash Cytoscape graph browser.")
    serve_parser.add_argument("--viz-json", required=True, type=Path)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8060)
    serve_parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    if args.command == "export":
        _export(args)
    elif args.command == "serve":
        serve_viz_graph(args.viz_json, host=args.host, port=args.port, debug=bool(args.debug))


def _export(args: argparse.Namespace) -> None:
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_graph = load_viz_graph(
        VisualizationInputs(
            graph_path=args.graph,
            image_manifest_path=args.image_manifest,
            trace_path=args.trace,
            success_paths_path=args.success_paths,
            atomic_transition_manifest_path=args.atomic_transition_manifest,
            unique_states_path=args.unique_states,
            full_branch_graph_path=args.full_branch_graph,
        )
    )
    viz_json = out_dir / "viz_graph.json"
    write_viz_graph(viz_graph, viz_json)
    export_pipeline_mermaid(out_dir / "pipeline_overview.mmd")
    export_interactive_html(viz_graph, out_dir / "task_interactive.html")
    write_summary_markdown(viz_graph, out_dir / "README.md")

    dot_specs = [
        ("all", "all", False),
        ("trace_success", "trace_success", False),
        ("grouped_all", "all", True),
        ("grouped_trace_success", "trace_success", True),
    ]
    for name, mode, grouped_edges in dot_specs:
        dot_path = out_dir / f"task_static_{name}.dot"
        export_task_dot(viz_graph, dot_path, edge_mode=mode, grouped_edges=grouped_edges)
        if args.render_svg:
            svg_path = out_dir / f"task_static_{name}.svg"
            rendered = render_dot_to_svg(dot_path, svg_path)
            if not rendered:
                print("Graphviz 'dot' not found; skipped SVG rendering.")
    print(viz_json)


if __name__ == "__main__":
    main()
