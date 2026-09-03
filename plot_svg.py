

PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2"]

def _scale(points, width, height, pad=40):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = (max_x - min_x) or 1.0
    span_y = (max_y - min_y) or 1.0

    def transform(p):
        sx = pad + (p[0] - min_x) / span_x * (width - 2 * pad)
        # flip y so data-up is svg-up
        sy = height - (pad + (p[1] - min_y) / span_y * (height - 2 * pad))
        return sx, sy

    return transform


def scatter_svg(path, groups, title, width=520, height=420, star_points=None):
    """groups: dict label -> list of (x, y) points.
    star_points: optional list of (x, y) drawn as larger stars (e.g. centroids).
    """
    all_points = [p for pts in groups.values() for p in pts] + (star_points or [])
    transform = _scale(all_points, width, height)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica,Arial,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="16" '
        f'font-weight="bold" fill="#111">{title}</text>',
    ]

    for i, (label, pts) in enumerate(groups.items()):
        color = PALETTE[i % len(PALETTE)]
        for p in pts:
            sx, sy = transform(p)
            svg.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="5" fill="{color}" '
                       f'fill-opacity="0.85" stroke="white" stroke-width="0.5"/>')
        # legend
        ly = 50 + i * 20
        svg.append(f'<circle cx="{width-110}" cy="{ly}" r="5" fill="{color}"/>')
        svg.append(f'<text x="{width-98}" y="{ly+4}" font-size="12" fill="#333">{label}</text>')

    if star_points:
        for p in star_points:
            sx, sy = transform(p)
            svg.append(
                f'<path d="M{sx:.1f},{sy-9:.1f} l2.6,7.9 8.3,0 -6.7,4.9 2.6,7.9 '
                f'-6.8,-4.9 -6.8,4.9 2.6,-7.9 -6.7,-4.9 8.3,0 z" '
                f'fill="black" stroke="white" stroke-width="1"/>'
            )

    svg.append("</svg>")

    with open(path, "w") as f:
        f.write("\n".join(svg))


def animated_kmeans_html(path, points, frames, title="K-Means simulation",
                         width=620, height=480):
    """Write a self-contained animation that also renders in Google Colab."""
    all_points = points + [c for frame in frames for c in frame["centroids"]]
    transform = _scale(all_points, width, height, pad=55)
    point_pixels = [transform(p) for p in points]
    frame_data = []
    for frame in frames:
        frame_data.append({
            "centroids": [transform(c) for c in frame["centroids"]],
            "labels": frame["labels"],
        })

    circles = []
    for index, (sx, sy) in enumerate(point_pixels):
        circles.append(
            f'<circle id="point-{index}" cx="{sx:.1f}" cy="{sy:.1f}" '
            'r="5" stroke="white" stroke-width="0.7"/>')
    stars = [f'<text id="centroid-{i}" text-anchor="middle" '
             'font-size="28" font-weight="bold">*</text>'
             for i in range(len(frames[0]["centroids"]))]
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
           f'height="{height}" viewBox="0 0 {width} {height}">'
           f'<rect width="100%" height="100%" fill="white"/>'
           f'<text x="{width / 2:.0f}" y="28" text-anchor="middle" '
           'font-family="sans-serif" font-size="18" font-weight="bold">'
           f'{title}</text>'
           f'<text id="iteration" x="{width / 2:.0f}" y="52" '
           'text-anchor="middle" font-family="sans-serif" font-size="13" '
           'fill="#555"></text>'
           + "".join(circles) + "".join(stars) + '</svg>')
    import json
    html = f'''<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title></head>
<body style="margin:0;background:#f7f7f5">{svg}
<script>
const frames = {json.dumps(frame_data)};
const colors = {json.dumps(PALETTE)};
const duration = 850;
let frameIndex = 0;
function render() {{
    const frame = frames[frameIndex];
    frame.labels.forEach((label, index) => {{
        document.getElementById(`point-${{index}}`).setAttribute("fill", colors[label % colors.length]);
    }});
    frame.centroids.forEach((center, index) => {{
        const star = document.getElementById(`centroid-${{index}}`);
        star.setAttribute("x", center[0]);
        star.setAttribute("y", center[1] + 9);
        star.setAttribute("fill", colors[index % colors.length]);
    }});
    document.getElementById("iteration").textContent =
        `Iteration ${{frameIndex + 1}} / ${{frames.length}}`;
    frameIndex = (frameIndex + 1) % frames.length;
}}
render();
setInterval(render, duration);
</script></body></html>'''
    with open(path, "w") as output:
        output.write(html)
