import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
JSONL_PATH = BASE_DIR / "layer_metrics.jsonl"
SVG_PATH = BASE_DIR / "layer_ndcg5_hit5_by_layer.svg"


def load_rows(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    rows.sort(key=lambda row: row["layer"])
    return rows


def scale(value, src_min, src_max, dst_min, dst_max):
    if src_max == src_min:
        return (dst_min + dst_max) / 2
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def polyline_points(values, layers, chart_left, chart_top, chart_width, chart_height):
    y_min = 0.0
    y_max = max(values) * 1.12 if values else 1.0
    layer_min = min(layers) if layers else 0
    layer_max = max(layers) if layers else 1
    x_positions = []
    for layer in layers:
        x = scale(layer, layer_min, layer_max, chart_left, chart_left + chart_width)
        x_positions.append(x)
    points = []
    for x, value in zip(x_positions, values):
        y = scale(value, y_min, y_max, chart_top + chart_height, chart_top)
        points.append((x, y))
    return points, y_min, y_max


def build_svg(rows):
    layers = [row["layer"] for row in rows]
    ndcg5 = [row["metrics"]["NDCG@5"] for row in rows]
    hit5 = [row["metrics"]["Hit@5"] for row in rows]

    width = 980
    height = 560
    chart_left = 90
    chart_top = 70
    chart_width = 820
    chart_height = 360
    chart_bottom = chart_top + chart_height

    _, _, ndcg_y_max = polyline_points(ndcg5, layers, chart_left, chart_top, chart_width, chart_height)
    _, _, hit_y_max = polyline_points(hit5, layers, chart_left, chart_top, chart_width, chart_height)
    y_max = max(ndcg_y_max, hit_y_max)

    def to_points(values):
        points = []
        layer_min = min(layers) if layers else 0
        layer_max = max(layers) if layers else 1
        for layer, value in zip(layers, values):
            x = scale(layer, layer_min, layer_max, chart_left, chart_left + chart_width)
            y = scale(value, 0.0, y_max, chart_bottom, chart_top)
            points.append((x, y))
        return points

    ndcg_points = to_points(ndcg5)
    hit_points = to_points(hit5)

    y_ticks = 6
    grid_lines = []
    y_labels = []
    for i in range(y_ticks + 1):
        tick_value = y_max * i / y_ticks
        y = scale(tick_value, 0.0, y_max, chart_bottom, chart_top)
        grid_lines.append(
            f'<line x1="{chart_left}" y1="{y:.2f}" x2="{chart_left + chart_width}" y2="{y:.2f}" '
            'stroke="#d9d9d9" stroke-dasharray="4 4" stroke-width="1"/>'
        )
        y_labels.append(
            f'<text x="{chart_left - 12}" y="{y + 4:.2f}" font-size="12" text-anchor="end" fill="#444">{tick_value:.3f}</text>'
        )

    layer_min = min(layers) if layers else 0
    layer_max = max(layers) if layers else 1
    x_ticks = []
    x_grid_lines = []
    for layer in range(layer_min, layer_max + 1):
        x = scale(layer, layer_min, layer_max, chart_left, chart_left + chart_width)
        x_grid_lines.append(
            f'<line x1="{x:.2f}" y1="{chart_top}" x2="{x:.2f}" y2="{chart_bottom}" stroke="#efefef" stroke-width="1"/>'
        )
        x_ticks.append(
            f'<line x1="{x:.2f}" y1="{chart_bottom}" x2="{x:.2f}" y2="{chart_bottom + 6}" stroke="#444" stroke-width="1"/>'
        )
        x_ticks.append(
            f'<text x="{x:.2f}" y="{chart_bottom + 24}" font-size="12" text-anchor="middle" fill="#444">{layer}</text>'
        )

    def draw_series(points, values, color, marker_tag, label_dy):
        polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        circles = []
        labels = []
        for (x, y), value in zip(points, values):
            circles.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color}" stroke="white" stroke-width="1.2"/>'
            )
            labels.append(
                f'<text x="{x:.2f}" y="{y + label_dy:.2f}" font-size="11" text-anchor="middle" fill="{color}">{value:.3f}</text>'
            )
        return (
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{polyline}"/>',
            "\n".join(circles),
            "\n".join(labels),
        )

    ndcg_polyline, ndcg_markers, ndcg_labels = draw_series(ndcg_points, ndcg5, "#1f77b4", "circle", -10)
    hit_polyline, hit_markers, hit_labels = draw_series(hit_points, hit5, "#d62728", "square", 18)

    legend = """
    <g>
      <line x1="640" y1="40" x2="675" y2="40" stroke="#1f77b4" stroke-width="2.5"/>
      <circle cx="657.5" cy="40" r="4.5" fill="#1f77b4" stroke="white" stroke-width="1.2"/>
      <text x="685" y="44" font-size="13" fill="#222">NDCG@5</text>
      <line x1="770" y1="40" x2="805" y2="40" stroke="#d62728" stroke-width="2.5"/>
      <circle cx="787.5" cy="40" r="4.5" fill="#d62728" stroke="white" stroke-width="1.2"/>
      <text x="815" y="44" font-size="13" fill="#222">Hit@5</text>
    </g>
    """

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2}" y="34" font-size="22" font-weight="bold" text-anchor="middle" fill="#222">Layer-wise NDCG@5 and Hit@5</text>
  <text x="{width / 2}" y="{height - 18}" font-size="14" text-anchor="middle" fill="#333">Layer</text>
  <text x="24" y="{chart_top + chart_height / 2}" font-size="14" text-anchor="middle" fill="#333" transform="rotate(-90 24 {chart_top + chart_height / 2})">Metric Value</text>
  <rect x="{chart_left}" y="{chart_top}" width="{chart_width}" height="{chart_height}" fill="none" stroke="#444" stroke-width="1.2"/>
  {"".join(x_grid_lines)}
  {"".join(grid_lines)}
  {"".join(y_labels)}
  {"".join(x_ticks)}
  {ndcg_polyline}
  {hit_polyline}
  {ndcg_markers}
  {hit_markers}
  {ndcg_labels}
  {hit_labels}
  {legend}
</svg>
"""


def main():
    rows = load_rows(JSONL_PATH)
    svg = build_svg(rows)
    SVG_PATH.write_text(svg)
    print(f"Saved SVG to {SVG_PATH}")


if __name__ == "__main__":
    main()
