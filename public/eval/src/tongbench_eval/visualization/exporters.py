from __future__ import annotations

import json
import shutil
import subprocess
import base64
from pathlib import Path
from typing import Any


def export_pipeline_mermaid(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_pipeline_mermaid(), encoding="utf-8")


def export_task_dot(
    viz_graph: dict[str, Any],
    output_path: Path,
    *,
    edge_mode: str = "all",
    grouped_edges: bool = False,
) -> None:
    if edge_mode not in {"all", "trace_success"}:
        raise ValueError("edge_mode must be 'all' or 'trace_success'")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_task_dot(viz_graph, edge_mode=edge_mode, grouped_edges=grouped_edges), encoding="utf-8")


def render_dot_to_svg(dot_path: Path, svg_path: Path) -> bool:
    dot_bin = shutil.which("dot")
    if not dot_bin:
        return False
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([dot_bin, "-Tsvg", str(dot_path), "-o", str(svg_path)], check=True)
    return True


def write_summary_markdown(viz_graph: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    task = viz_graph.get("task", {})
    stats = viz_graph.get("stats", {})
    lines = [
        f"# {task.get('task_id')} 图可视化摘要",
        "",
        "## 任务",
        "",
        f"- Description: `{task.get('description')}`",
        f"- Source graph: `{task.get('source_graph')}`",
        f"- Expansion mode: `{task.get('expansion_mode')}`",
        f"- Nodes: `{stats.get('node_count')}`",
        f"- Edges: `{stats.get('edge_count')}`",
        f"- Aggregated edge groups: `{stats.get('edge_group_count')}`",
        f"- Parallel edges collapsed: `{stats.get('parallel_edge_reduction')}`",
        f"- Goal nodes: `{stats.get('goal_node_count')}`",
        f"- Success edges: `{stats.get('success_edge_count')}`",
        f"- Trace steps: `{stats.get('trace_step_count')}`",
        f"- Trace edges matched: `{stats.get('trace_edge_count')}`",
        f"- Full branch graph available: `{stats.get('full_branch_graph_available')}`",
        f"- Full branch nodes: `{stats.get('full_branch_node_count')}`",
        f"- Full branch raw edges: `{stats.get('full_branch_edge_count')}`",
        f"- Full branch statuses: `{stats.get('full_branch_status_counts')}`",
        "",
        "## 输出文件",
        "",
        "- `viz_graph.json`: 统一中间格式，给交互 app 使用。",
        "- `task_static_all.dot/svg`: raw 完整 atomic expanded graph，保留所有 parallel edge。",
        "- `task_static_trace_success.dot/svg`: raw success edge 和模型 trace edge。",
        "- `task_static_grouped_all.dot/svg`: 按 source/target/action 合并 parallel edge。",
        "- `task_static_grouped_trace_success.dot/svg`: 合并后的 success edge 和模型 trace edge。",
        "- `task_interactive.html`: standalone 交互式图浏览器。",
        "- `pipeline_overview.mmd`: 三段 pipeline 总览 Mermaid 图。",
        "",
        "## 交互图层",
        "",
        "- `Search Result`: 状态图叠加 search outcome，包括 shortest success、success、trace、pruned/loop/stopped。",
        "- `Expand Mapping`: 高层 parent action 展开到 atomic action chain。",
        "- `Agent Trace Overlay`: 模型实际 Eval 轨迹叠加在 atomic graph 上。",
        "- `Image State`: 状态节点挂载采集或复用图片。",
        "- `Atomic Graph`: Eval 实际使用的完整 atomic graph。",
        "- `Full Graph`: 如果提供 `*_full_branch_graph.json` 就画原始大图；否则显示降级摘要。",
        "",
        "## 图例",
        "",
        "- 绿色描边节点：initial state。",
        "- 蓝色描边节点：goal predicates 已满足。",
        "- 橙色粗边：模型实际 trace。",
        "- 蓝色边：shortest success path。",
        "- 绿色边：success edge。",
        "- 红色虚线边：pruned、loop 或 stopped branch。",
        "- 灰色边：普通 explored edge。",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_interactive_html(viz_graph: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _html_payload(viz_graph)
    output_path.write_text(_interactive_html(payload), encoding="utf-8")


def _pipeline_mermaid() -> str:
    return """flowchart LR
  scene[Scene objects / templates] --> taskgen[L1/L2/L3 task generation]
  taskgen --> fullbranch[Full-branch search graph]
  fullbranch --> success[Success paths]
  fullbranch --> atomic[Atomic expanded graph]
  atomic --> states[Unique state manifest]
  states --> capture[Image capture / L1 image reuse]
  capture --> evalinput[Eval input package]
  atomic --> evalinput
  evalinput --> graphenv[Offline graph_env]
  graphenv --> obs[Observation: text + image + available actions]
  obs --> agent[Hermes / Codex / Claude Code / OpenClaw]
  agent --> action[Selected action]
  action --> graphenv
  graphenv --> trace[trace.jsonl / decision_log.jsonl]
  graphenv --> grade[grade.json / score.json]

  classDef private fill:#f4f0ff,stroke:#7c3aed,color:#111;
  classDef public fill:#eef6ff,stroke:#2563eb,color:#111;
  classDef output fill:#fff7ed,stroke:#ea580c,color:#111;
  class scene,taskgen,fullbranch,success,atomic,states private;
  class capture,evalinput,graphenv,obs,agent,action public;
  class trace,grade output;
"""


def _task_dot(viz_graph: dict[str, Any], *, edge_mode: str, grouped_edges: bool = False) -> str:
    task = viz_graph.get("task", {})
    lines = [
        "digraph TongBenchTask {",
        '  graph [rankdir=LR, bgcolor="white", pad=0.2, nodesep=0.45, ranksep=0.85];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.08,0.06"];',
        '  edge [fontname="Helvetica", fontsize=8, arrowsize=0.7];',
        f'  label="{_dot_escape(str(task.get("task_id") or "TongBench task"))}\\n{_dot_escape(str(task.get("description") or ""))}";',
        '  labelloc="t";',
        "",
    ]
    for node in viz_graph.get("nodes", []) or []:
        lines.append(f"  {node['id']} [{_node_attrs(node)}];")
    lines.append("")
    edge_key = "edge_groups" if grouped_edges else "edges"
    for edge in viz_graph.get(edge_key, []) or []:
        if edge_mode == "trace_success" and not (edge.get("is_trace_edge") or edge.get("is_success_edge")):
            continue
        lines.append(f"  {edge['source']} -> {edge['target']} [{_edge_attrs(edge)}];")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _node_attrs(node: dict[str, Any]) -> str:
    predicates = list(node.get("state", []) or [])
    short_predicates = "\\n".join(_dot_escape(pred) for pred in predicates[:4])
    if len(predicates) > 4:
        short_predicates += f"\\n... +{len(predicates) - 4}"
    trace = node.get("trace_steps", [])
    label = f"{node.get('id')} / {node.get('unique_state_id')}"
    if trace:
        label += f"\\ntrace: {trace}"
    if short_predicates:
        label += f"\\n{short_predicates}"
    color = "#94a3b8"
    fill = "#f8fafc"
    penwidth = "1.2"
    if node.get("is_initial"):
        color = "#16a34a"
        fill = "#ecfdf5"
        penwidth = "2.4"
    if node.get("is_goal"):
        color = "#2563eb"
        fill = "#eff6ff"
        penwidth = "2.4"
    if node.get("is_initial") and node.get("is_goal"):
        color = "#7c3aed"
        fill = "#f5f3ff"
    return _attrs(label=label, color=color, fillcolor=fill, penwidth=penwidth)


def _edge_attrs(edge: dict[str, Any]) -> str:
    label = str(edge.get("atomic_action") or edge.get("action_type") or edge.get("id"))
    color = "#cbd5e1"
    penwidth = "0.8"
    style = "solid"
    if edge.get("is_success_edge"):
        color = "#16a34a"
        penwidth = "1.8"
    if edge.get("is_trace_edge"):
        color = "#f97316"
        penwidth = "3.0"
    if edge.get("branch_status") == "pruned":
        style = "dashed"
    return _attrs(label=label, color=color, penwidth=penwidth, style=style)


def _attrs(**items: str) -> str:
    return ", ".join(f'{key}="{_dot_escape(str(value))}"' for key, value in items.items())


def _dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def write_debug_json(viz_graph: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(viz_graph, ensure_ascii=False, indent=2), encoding="utf-8")


def _html_payload(viz_graph: dict[str, Any]) -> dict[str, Any]:
    nodes = []
    images_by_path: dict[str, str] = {}
    for node in viz_graph.get("nodes", []) or []:
        item = dict(node)
        image_path = item.get("image_path")
        if image_path and str(image_path) not in images_by_path:
            data_url = _image_data_url(str(image_path))
            if data_url:
                images_by_path[str(image_path)] = data_url
        nodes.append(item)
    layers = json.loads(json.dumps(viz_graph.get("layers", {}), ensure_ascii=False))
    for layer in layers.values():
        for node in layer.get("nodes", []) or []:
            image_path = node.get("image_path")
            if image_path and str(image_path) not in images_by_path:
                data_url = _image_data_url(str(image_path))
                if data_url:
                    images_by_path[str(image_path)] = data_url
    return {
        "task": viz_graph.get("task", {}),
        "stats": viz_graph.get("stats", {}),
        "nodes": nodes,
        "edges": viz_graph.get("edges", []),
        "edge_groups": viz_graph.get("edge_groups", []),
        "layers": layers,
        "images_by_path": images_by_path,
        "trace": viz_graph.get("trace", []),
    }


def _pipeline_elements(viz_graph: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    task = viz_graph.get("task", {}) or {}
    stats = viz_graph.get("stats", {}) or {}
    node_count = stats.get("node_count", 0)
    edge_count = stats.get("edge_count", 0)
    edge_group_count = stats.get("edge_group_count", edge_count)
    trace_step_count = stats.get("trace_step_count", 0)
    return {
        "nodes": [
            {"id": "scene", "label": "Scene\\nobjects/templates", "level": "private", "detail": "TongSim scene object set and action templates."},
            {"id": "taskgen", "label": "Task generation\\nL1/L2/L3", "level": "private", "detail": "Generate compositional task descriptions and predicates."},
            {"id": "fullbranch", "label": "Full-branch\\nsearch graph", "level": "private", "detail": f"Source graph: {task.get('source_graph')}."},
            {"id": "atomic", "label": f"Atomic graph\\n{node_count} nodes / {edge_count} raw edges", "level": "private", "detail": f"Expanded into atomic actions; {edge_group_count} visual edge groups after collapsing parallel edges."},
            {"id": "capture", "label": "Image capture\\nstate manifest", "level": "private", "detail": "Map unique states to captured/reused observation images."},
            {"id": "eval", "label": "Eval graph_env\\ntext + image + actions", "level": "public", "detail": "Offline environment exposes one observation and candidate actions each step."},
            {"id": "agent", "label": "Agent\\nHermes/Codex/etc.", "level": "public", "detail": "Agent chooses one action template and bound objects."},
            {"id": "trace", "label": f"Trace + score\\n{trace_step_count} steps", "level": "output", "detail": "Recorded trace.jsonl, decision logs, grade/score files."},
        ],
        "edges": [
            {"id": "p0", "source": "scene", "target": "taskgen", "label": "enumerate"},
            {"id": "p1", "source": "taskgen", "target": "fullbranch", "label": "search"},
            {"id": "p2", "source": "fullbranch", "target": "atomic", "label": "expand"},
            {"id": "p3", "source": "atomic", "target": "capture", "label": "state images"},
            {"id": "p4", "source": "capture", "target": "eval", "label": "package"},
            {"id": "p5", "source": "atomic", "target": "eval", "label": "graph"},
            {"id": "p6", "source": "eval", "target": "agent", "label": "observe"},
            {"id": "p7", "source": "agent", "target": "eval", "label": "choose_action"},
            {"id": "p8", "source": "eval", "target": "trace", "label": "record"},
        ],
    }


def _image_data_url(path: str) -> str | None:
    image = Path(path)
    if not image.exists():
        return None
    suffix = image.suffix.lower().lstrip(".") or "jpeg"
    if suffix == "jpg":
        suffix = "jpeg"
    return f"data:image/{suffix};base64,{base64.b64encode(image.read_bytes()).decode('ascii')}"


def _interactive_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    template = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TongBench Multi-Layer Graph</title>
  <script src="https://unpkg.com/cytoscape@3.31.2/dist/cytoscape.min.js"></script>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; color: #111827; }
    .app { display: grid; grid-template-columns: 410px 1fr; min-height: 100vh; }
    aside { box-sizing: border-box; height: 100vh; overflow-y: auto; border-right: 1px solid #e5e7eb; padding: 16px; background: #fff; }
    main { height: 100vh; min-width: 0; }
    h1 { font-size: 18px; margin: 0 0 10px; }
    h2 { font-size: 14px; margin: 18px 0 8px; color: #334155; }
    p { font-size: 13px; line-height: 1.38; }
    label { display: block; font-size: 12px; margin: 10px 0 4px; color: #475569; }
    select, input { width: 100%; box-sizing: border-box; border: 1px solid #cbd5e1; border-radius: 4px; padding: 7px; background: #fff; }
    pre { background: #f8fafc; border: 1px solid #e2e8f0; padding: 8px; overflow-x: auto; white-space: pre-wrap; font-size: 11px; max-height: 360px; }
    img { width: 100%; border: 1px solid #e2e8f0; margin: 8px 0; }
    #cy { width: 100%; height: 100%; background: #fbfdff; }
    .legend { display: grid; gap: 6px; font-size: 12px; }
    .swatch { display: inline-block; width: 18px; height: 8px; margin-right: 6px; border-radius: 2px; vertical-align: middle; }
    .muted { color: #64748b; }
    .warn { color: #b45309; }
  </style>
</head>
<body>
<div class="app">
  <aside>
    <h1>TongBench Multi-Layer Graph</h1>
    <div id="summary"></div>
    <label for="layer-mode">Graph layer</label>
    <select id="layer-mode">
      <option value="search" selected>Search Result</option>
      <option value="expand">Expand Mapping</option>
      <option value="trace">Agent Trace Overlay</option>
      <option value="image">Image State</option>
      <option value="atomic">Atomic Graph</option>
      <option value="full">Full Graph</option>
      <option value="pipeline">Pipeline Overview</option>
    </select>
    <label for="edge-source">Edge display</label>
    <select id="edge-source">
      <option value="grouped" selected>Collapsed/grouped edges</option>
      <option value="raw">Raw edges</option>
    </select>
    <label for="edge-filter">Edge filter</label>
    <select id="edge-filter">
      <option value="all">All edges</option>
      <option value="important" selected>Important edges</option>
      <option value="shortest">Shortest success</option>
      <option value="success">All success</option>
      <option value="trace">Agent trace</option>
      <option value="stopped">Pruned / loop / stopped</option>
    </select>
    <label for="label-mode">Edge labels</label>
    <select id="label-mode">
      <option value="important" selected>Important only</option>
      <option value="all">All labels</option>
      <option value="none">No edge labels</option>
    </select>
    <label for="layout-mode">Layout</label>
    <select id="layout-mode">
      <option value="auto" selected>Auto</option>
      <option value="breadthfirst">Breadth-first</option>
      <option value="cose">Force-directed</option>
      <option value="concentric">Concentric</option>
      <option value="grid">Grid</option>
    </select>
    <label for="search">Search node/action</label>
    <input id="search" placeholder="as4, switch_off, TV, path p0...">
    <h2>Legend</h2>
    <div class="legend">
      <div><span class="swatch" style="background:#dcfce7;border:2px solid #16a34a"></span>initial state</div>
      <div><span class="swatch" style="background:#dbeafe;border:2px solid #2563eb"></span>goal state</div>
      <div><span class="swatch" style="background:#f97316"></span>agent trace</div>
      <div><span class="swatch" style="background:#2563eb"></span>shortest success</div>
      <div><span class="swatch" style="background:#16a34a"></span>success edge</div>
      <div><span class="swatch" style="background:#a855f7"></span>mixed branch group</div>
      <div><span class="swatch" style="background:#ef4444"></span>pruned / loop / stopped</div>
    </div>
    <h2>Details</h2>
    <div id="details"><p>Click a node or edge.</p></div>
  </aside>
  <main><div id="cy"></div></main>
</div>
<script>
const DATA = __DATA_JSON__;
const layerSelect = document.getElementById('layer-mode');
const edgeSourceSelect = document.getElementById('edge-source');
const edgeFilterSelect = document.getElementById('edge-filter');
const labelModeSelect = document.getElementById('label-mode');
const layoutModeSelect = document.getElementById('layout-mode');
const searchInput = document.getElementById('search');

applyInitialParams();

function activeLayer() {
  return DATA.layers[layerSelect.value] || DATA.layers.search || Object.values(DATA.layers)[0];
}

function setSelectValue(select, value) {
  if (!value) return;
  for (const opt of select.options) {
    if (opt.value === value) {
      select.value = value;
      return;
    }
  }
}

function applyInitialParams() {
  const params = new URLSearchParams(window.location.search);
  setSelectValue(layerSelect, params.get('layer'));
  setSelectValue(edgeSourceSelect, params.get('edge'));
  setSelectValue(edgeFilterSelect, params.get('filter'));
  setSelectValue(labelModeSelect, params.get('labels'));
  setSelectValue(layoutModeSelect, params.get('layout'));
}

function updateUrlParams() {
  const params = new URLSearchParams();
  params.set('layer', layerSelect.value);
  params.set('edge', edgeSourceSelect.value);
  params.set('filter', edgeFilterSelect.value);
  params.set('labels', labelModeSelect.value);
  params.set('layout', layoutModeSelect.value);
  const url = `${window.location.pathname}?${params.toString()}`;
  window.history.replaceState(null, '', url);
}

function classesForNode(n) {
  const cls = ['node-kind-' + (n.type || 'state')];
  if (n.view) cls.push('view-' + n.view);
  if (n.level) cls.push('level-' + n.level);
  if (n.is_initial) cls.push('initial');
  if (n.is_goal) cls.push('goal');
  if (n.trace_steps && n.trace_steps.length) cls.push('trace-node');
  if (n.image_path) cls.push('has-image');
  if (n.dimmed) cls.push('dimmed');
  if (n.is_dead_end) cls.push('dead-end');
  return cls.join(' ');
}

function classesForEdge(e) {
  const cls = ['edge'];
  if (e.view) cls.push('view-' + e.view);
  if (e.search_status) cls.push('status-' + e.search_status);
  if (e.is_success_edge) cls.push('success');
  if (e.is_shortest_success_edge) cls.push('shortest');
  if (e.is_trace_edge) cls.push('trace');
  return cls.join(' ');
}

function currentEdges(layer) {
  if (edgeSourceSelect.value === 'raw' && layer.raw_edges) return layer.raw_edges;
  return layer.edges || [];
}

function edgeLabel(edge) {
  const mode = labelModeSelect.value;
  if (mode === 'none') return '';
  const label = edge.display_label || edge.label || edge.atomic_action || edge.id || '';
  if (mode === 'all') return label;
  const layer = activeLayer();
  const edgeCount = (currentEdges(layer) || []).length;
  const dense = edgeCount > 70;
  const status = edge.search_status || '';
  if (edge.is_trace_edge || edge.is_shortest_success_edge || status === 'expand') return label;
  if (status === 'mixed' && edgeCount <= 140) return label;
  if (!dense && (edge.is_success_edge || ['stopped', 'pruned', 'loop'].includes(status))) return label;
  return '';
}

function layerStatsText(layer) {
  const stats = layer.stats || {};
  const parts = [];
  if (stats.node_count !== undefined) parts.push(`${stats.node_count} nodes`);
  if (stats.raw_edge_count !== undefined) parts.push(`${stats.raw_edge_count} raw edges`);
  if (stats.edge_count !== undefined) parts.push(`${stats.edge_count} displayed edges`);
  if (stats.trace_step_count !== undefined) parts.push(`${stats.trace_step_count} trace steps`);
  if (stats.status_counts) {
    parts.push(Object.entries(stats.status_counts).map(([k, v]) => `${k}:${v}`).join(' / '));
  }
  return parts.join(' / ') || `${DATA.stats.node_count} atomic states / ${DATA.stats.edge_count} raw edges`;
}

function filteredElements() {
  const layer = activeLayer();
  const mode = edgeFilterSelect.value;
  const q = searchInput.value.trim().toLowerCase();
  let nodes = layer.nodes || [];
  let edges = currentEdges(layer);
  if (mode === 'important' && ['pipeline', 'full'].includes(layer.id)) {
    edges = edges;
  } else if (mode === 'important') {
    edges = edges.filter(e => e.is_trace_edge || e.is_shortest_success_edge || e.is_success_edge || ['expand', 'stopped', 'pruned', 'loop'].includes(e.search_status));
  } else if (mode === 'shortest') {
    edges = edges.filter(e => e.is_shortest_success_edge || e.search_status === 'shortest');
  } else if (mode === 'success') {
    edges = edges.filter(e => e.is_success_edge || e.is_shortest_success_edge || e.search_status === 'success' || e.search_status === 'shortest');
  } else if (mode === 'trace') {
    edges = edges.filter(e => e.is_trace_edge || e.search_status === 'trace');
  } else if (mode === 'stopped') {
    edges = edges.filter(e => ['stopped', 'pruned', 'loop'].includes(e.search_status));
  }
  if (q) {
    const nodeHits = new Set(nodes.filter(n => JSON.stringify(n).toLowerCase().includes(q)).map(n => n.id));
    const edgeHits = edges.filter(e => JSON.stringify(e).toLowerCase().includes(q));
    for (const e of edgeHits) { nodeHits.add(e.source); nodeHits.add(e.target); }
    nodes = nodes.filter(n => nodeHits.has(n.id));
    edges = edges.filter(e => nodeHits.has(e.source) && nodeHits.has(e.target));
  }
  return nodes.map(n => ({ data: {...n, label: n.display_label || n.label || n.id}, classes: classesForNode(n) }))
    .concat(edges.map(e => ({ data: {...e, label: edgeLabel(e)}, classes: classesForEdge(e) })));
}

function renderSummary() {
  const layer = activeLayer();
  const warning = layer.available === false ? '<p class="warn"><b>Note:</b> full branch graph file is not available for this input; this layer is a best-effort summary.</p>' : '';
  document.getElementById('summary').innerHTML = `
    <p><b>Task:</b> ${DATA.task.task_id}</p>
    <p><b>Description:</b> ${DATA.task.description || ''}</p>
    <p><b>Layer:</b> ${layer.title}</p>
    <p class="muted">${layer.description || ''}</p>
    <p><b>Stats:</b> ${layerStatsText(layer)}</p>
    <p class="muted">Tip: use <b>Full Graph</b> for BFS/pruning, <b>Search Result</b> for Eval atomic graph, and <b>Agent Trace Overlay</b> for model actions.</p>
    ${warning}
  `;
}

const cy = cytoscape({
  container: document.getElementById('cy'),
  elements: filteredElements(),
  layout: { name: 'breadthfirst', directed: true, spacingFactor: 1.75, padding: 60 },
  style: [
    { selector: 'node', style: { label: 'data(label)', 'font-size': 9, 'text-valign': 'center', 'text-halign': 'center', 'text-wrap': 'wrap', 'text-max-width': 100, 'background-color': '#f8fafc', 'border-color': '#94a3b8', 'border-width': 1, width: 46, height: 46 } },
    { selector: 'node.initial', style: { 'background-color': '#dcfce7', 'border-color': '#16a34a', 'border-width': 4 } },
    { selector: 'node.goal', style: { 'background-color': '#dbeafe', 'border-color': '#2563eb', 'border-width': 4 } },
    { selector: 'node.dead-end', style: { 'background-color': '#fee2e2', 'border-color': '#ef4444', 'border-width': 2 } },
    { selector: 'node.trace-node', style: { shape: 'round-rectangle' } },
    { selector: 'node.has-image', style: { 'border-style': 'double' } },
    { selector: 'node.dimmed', style: { opacity: 0.28 } },
    { selector: 'node.node-kind-parent_action', style: { shape: 'round-rectangle', width: 132, height: 62, 'background-color': '#fef3c7', 'border-color': '#d97706', 'border-width': 2, 'font-size': 10 } },
    { selector: 'node.node-kind-atomic_step', style: { shape: 'round-rectangle', width: 128, height: 58, 'background-color': '#ecfeff', 'border-color': '#0891b2', 'border-width': 2, 'font-size': 9 } },
    { selector: 'node.node-kind-full_state', style: { shape: 'round-rectangle', width: 82, height: 54, 'background-color': '#f8fafc', 'border-color': '#64748b', 'border-width': 1.5, 'font-size': 8, 'text-max-width': 76 } },
    { selector: 'node.node-kind-missing', style: { shape: 'round-rectangle', width: 150, height: 62, 'background-color': '#fff7ed', 'border-color': '#f97316', 'border-width': 2 } },
    { selector: 'node.node-kind-summary', style: { shape: 'round-rectangle', width: 150, height: 62, 'background-color': '#f8fafc', 'border-color': '#64748b', 'border-width': 2 } },
    { selector: 'node.node-kind-pipeline', style: { shape: 'round-rectangle', width: 140, height: 62, 'background-color': '#f8fafc', 'border-width': 2, 'border-color': '#64748b', 'font-size': 10 } },
    { selector: 'node.level-private', style: { 'background-color': '#f4f0ff', 'border-color': '#7c3aed' } },
    { selector: 'node.level-public', style: { 'background-color': '#eef6ff', 'border-color': '#2563eb' } },
    { selector: 'node.level-output', style: { 'background-color': '#fff7ed', 'border-color': '#ea580c' } },
    { selector: 'edge', style: { label: 'data(label)', 'font-size': 8, 'curve-style': 'bezier', 'target-arrow-shape': 'triangle', 'line-color': '#cbd5e1', 'target-arrow-color': '#cbd5e1', width: 1, 'text-rotation': 'autorotate', 'text-margin-y': -8, 'text-background-color': '#fff', 'text-background-opacity': 0.92, 'text-background-padding': 2 } },
    { selector: 'edge.success', style: { 'line-color': '#16a34a', 'target-arrow-color': '#16a34a', width: 2 } },
    { selector: 'edge.shortest', style: { 'line-color': '#2563eb', 'target-arrow-color': '#2563eb', width: 3 } },
    { selector: 'edge.trace', style: { 'line-color': '#f97316', 'target-arrow-color': '#f97316', width: 4, 'z-index': 10 } },
    { selector: 'edge.status-pruned, edge.status-loop, edge.status-stopped', style: { 'line-color': '#ef4444', 'target-arrow-color': '#ef4444', width: 2, 'line-style': 'dashed' } },
    { selector: 'edge.status-mixed', style: { 'line-color': '#a855f7', 'target-arrow-color': '#a855f7', width: 2.4, 'line-style': 'dashed' } },
    { selector: 'edge.status-expand', style: { 'line-color': '#0891b2', 'target-arrow-color': '#0891b2', width: 2 } },
    { selector: 'edge.view-pipeline', style: { 'line-color': '#64748b', 'target-arrow-color': '#64748b', width: 2 } }
  ]
});

function relayout() {
  updateUrlParams();
  renderSummary();
  const layer = activeLayer();
  edgeSourceSelect.disabled = !(layer.raw_edges && layer.raw_edges.length && layer.raw_edges.length !== (layer.edges || []).length);
  cy.elements().remove();
  cy.add(filteredElements());
  const edgeCount = currentEdges(layer).length;
  const isDense = edgeCount > 70 || ['search', 'atomic', 'image', 'full'].includes(layer.id);
  const requestedLayout = layoutModeSelect.value === 'auto' ? (layer.layout || 'breadthfirst') : layoutModeSelect.value;
  const layoutOptions = {
    name: requestedLayout,
    directed: true,
    spacingFactor: isDense ? 2.35 : 1.55,
    padding: 110,
    animate: false,
  };
  if (requestedLayout === 'cose') {
    Object.assign(layoutOptions, {
      idealEdgeLength: isDense ? 130 : 90,
      nodeRepulsion: isDense ? 9000 : 5000,
      gravity: 0.25,
      numIter: 1400,
    });
  }
  cy.layout(layoutOptions).run();
  cy.fit(undefined, 40);
}

layerSelect.addEventListener('change', relayout);
edgeSourceSelect.addEventListener('change', relayout);
edgeFilterSelect.addEventListener('change', relayout);
labelModeSelect.addEventListener('change', relayout);
layoutModeSelect.addEventListener('change', relayout);
searchInput.addEventListener('input', relayout);

function showDetails(data) {
  const details = document.getElementById('details');
  const imageData = data.image_path ? DATA.images_by_path[data.image_path] : null;
  const image = imageData ? `<img src="${imageData}">` : '';
  const printable = {...data};
  const body = JSON.stringify(printable, null, 2);
  details.innerHTML = `<h2>${escapeHtml(String(data.id || data.label))}</h2>${image}<pre>${escapeHtml(body)}</pre>`;
}

function escapeHtml(value) {
  return value.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
}

cy.on('tap', 'node', evt => showDetails(evt.target.data()));
cy.on('tap', 'edge', evt => showDetails(evt.target.data()));
renderSummary();
relayout();
</script>
</body>
</html>
"""
    return template.replace("__DATA_JSON__", data_json)
