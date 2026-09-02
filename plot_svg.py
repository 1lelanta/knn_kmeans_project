

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
