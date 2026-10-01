from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def serve_viz_graph(viz_json: Path, *, host: str = "127.0.0.1", port: int = 8060, debug: bool = False) -> None:
    try:
        from dash import Dash, Input, Output, dcc, html
        import dash_cytoscape as cyto
    except ModuleNotFoundError as exc:
        missing = exc.name or "dash_cytoscape"
        raise SystemExit(
            f"Missing visualization dependency '{missing}'. "
            "Install with: python -m pip install -r requirements-visualization.txt"
        ) from exc

    data = json.loads(viz_json.read_text(encoding="utf-8"))
    elements = _cy_elements(data)
    stylesheet = _stylesheet()
    app = Dash(__name__)
    app.title = f"TongBench Graph - {data.get('task', {}).get('task_id', '')}"
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H2("TongBench Task Graph"),
                    html.Div(_summary_lines(data), className="summary"),
                    html.Label("Edge filter"),
                    dcc.Dropdown(
                        id="edge-filter",
                        options=[
                            {"label": "Trace + success edges", "value": "trace_success"},
                            {"label": "All edges", "value": "all"},
                        ],
                        value="trace_success",
                        clearable=False,
                    ),
                    html.Div(id="details", className="details"),
                ],
                className="sidebar",
            ),
            html.Div(
                [
                    cyto.Cytoscape(
                        id="task-graph",
                        elements=elements,
                        layout={"name": "breadthfirst", "directed": True, "spacingFactor": 1.4},
                        stylesheet=stylesheet,
                        style={"width": "100%", "height": "100vh"},
                        minZoom=0.08,
                        maxZoom=3,
                    )
                ],
                className="graph",
            ),
        ],
        className="app",
    )
    app.index_string = _index_string()

    @app.callback(Output("task-graph", "elements"), Input("edge-filter", "value"))
    def update_elements(edge_filter: str) -> list[dict[str, Any]]:
        if edge_filter == "all":
            return elements
        keep_edge_ids = {
            edge["data"]["id"]
            for edge in elements
            if "source" in edge.get("data", {})
            and (edge.get("classes", "").find("trace") >= 0 or edge.get("classes", "").find("success") >= 0)
        }
        keep_node_ids = {
            item["data"]["id"]
            for item in elements
            if "source" not in item.get("data", {})
        }
        filtered: list[dict[str, Any]] = []
        for item in elements:
            data_item = item.get("data", {})
            if "source" not in data_item:
                filtered.append(item)
            elif data_item.get("id") in keep_edge_ids and data_item.get("source") in keep_node_ids and data_item.get("target") in keep_node_ids:
                filtered.append(item)
        return filtered

    @app.callback(Output("details", "children"), Input("task-graph", "tapNodeData"), Input("task-graph", "tapEdgeData"))
    def show_details(node_data: dict[str, Any] | None, edge_data: dict[str, Any] | None) -> list[Any]:
        item = edge_data or node_data
        if not item:
            return [html.P("Click a node or edge to inspect details.")]
        children: list[Any] = [html.H3(item.get("id", "selected"))]
        image_path = item.get("image_path")
        if image_path:
            children.append(html.Img(src=_image_src(image_path), className="state-image"))
        for key in ["label", "atomic_action", "template_id", "action_type", "branch_status", "trace_steps", "state"]:
            if key in item and item[key] not in (None, "", []):
                value = item[key]
                children.append(html.H4(key))
                children.append(html.Pre(json.dumps(value, ensure_ascii=False, indent=2) if not isinstance(value, str) else value))
        return children

    app.run_server(host=host, port=port, debug=debug)


def _cy_elements(data: dict[str, Any]) -> list[dict[str, Any]]:
    elements: list[dict[str, Any]] = []
    for node in data.get("nodes", []) or []:
        classes = ["state"]
        if node.get("is_initial"):
            classes.append("initial")
        if node.get("is_goal"):
            classes.append("goal")
        if node.get("trace_steps"):
            classes.append("trace-node")
        elements.append(
            {
                "data": {
                    "id": node["id"],
                    "label": node["id"],
                    "unique_state_id": node.get("unique_state_id"),
                    "state": node.get("state", []),
                    "trace_steps": node.get("trace_steps", []),
                    "image_path": node.get("image_path"),
                },
                "classes": " ".join(classes),
            }
        )
    for edge in data.get("edges", []) or []:
        classes = ["edge"]
        if edge.get("is_success_edge"):
            classes.append("success")
        if edge.get("is_trace_edge"):
            classes.append("trace")
        elements.append(
            {
                "data": {
                    "id": edge["id"],
                    "source": edge["source"],
                    "target": edge["target"],
                    "label": edge.get("label"),
                    "atomic_action": edge.get("atomic_action"),
                    "template_id": edge.get("template_id"),
                    "action_type": edge.get("action_type"),
                    "branch_status": edge.get("branch_status"),
                    "trace_steps": edge.get("trace_steps", []),
                },
                "classes": " ".join(classes),
            }
        )
    return elements


def _stylesheet() -> list[dict[str, Any]]:
    return [
        {
            "selector": "node",
            "style": {
                "label": "data(label)",
                "font-size": 9,
                "text-valign": "center",
                "text-halign": "center",
                "background-color": "#f8fafc",
                "border-color": "#94a3b8",
                "border-width": 1,
                "width": 38,
                "height": 38,
            },
        },
        {"selector": "node.initial", "style": {"background-color": "#dcfce7", "border-color": "#16a34a", "border-width": 4}},
        {"selector": "node.goal", "style": {"background-color": "#dbeafe", "border-color": "#2563eb", "border-width": 4}},
        {"selector": "node.trace-node", "style": {"shape": "round-rectangle"}},
        {
            "selector": "edge",
            "style": {
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
                "line-color": "#cbd5e1",
                "target-arrow-color": "#cbd5e1",
                "width": 1,
                "label": "data(label)",
                "font-size": 7,
                "text-rotation": "autorotate",
                "text-background-color": "white",
                "text-background-opacity": 0.85,
            },
        },
        {"selector": "edge.success", "style": {"line-color": "#16a34a", "target-arrow-color": "#16a34a", "width": 2}},
        {"selector": "edge.trace", "style": {"line-color": "#f97316", "target-arrow-color": "#f97316", "width": 4, "z-index": 10}},
    ]


def _summary_lines(data: dict[str, Any]) -> list[Any]:
    task = data.get("task", {})
    stats = data.get("stats", {})
    return [
        html.P(f"Task: {task.get('task_id')}"),
        html.P(f"Description: {task.get('description')}"),
        html.P(
            f"Nodes {stats.get('node_count')} / Edges {stats.get('edge_count')} / "
            f"Trace steps {stats.get('trace_step_count')}"
        ),
    ]


def _image_src(path: str) -> str:
    image = Path(path)
    if not image.exists():
        return ""
    import base64

    ext = image.suffix.lower().lstrip(".") or "jpg"
    if ext == "jpg":
        ext = "jpeg"
    payload = base64.b64encode(image.read_bytes()).decode("ascii")
    return f"data:image/{ext};base64,{payload}"


def _index_string() -> str:
    return """<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      body { margin: 0; font-family: Inter, Arial, sans-serif; color: #111827; }
      .app { display: grid; grid-template-columns: 360px 1fr; min-height: 100vh; }
      .sidebar { border-right: 1px solid #e5e7eb; padding: 16px; overflow-y: auto; height: 100vh; box-sizing: border-box; }
      .sidebar h2 { font-size: 18px; margin: 0 0 12px; }
      .summary p { margin: 6px 0; font-size: 13px; }
      .details { margin-top: 18px; }
      .details h3 { font-size: 15px; margin: 0 0 10px; }
      .details h4 { font-size: 12px; margin: 12px 0 4px; color: #475569; }
      .details pre { background: #f8fafc; border: 1px solid #e2e8f0; padding: 8px; overflow-x: auto; font-size: 11px; white-space: pre-wrap; }
      .state-image { display: block; width: 100%; border: 1px solid #e2e8f0; margin: 8px 0 12px; }
      .graph { min-width: 0; }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>"""
