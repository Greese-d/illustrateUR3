#!/usr/bin/env python3
import json
import math
import re
import sys
import xml.etree.ElementTree as ET

Q_SAMPLES = 20
C_SAMPLES = 24
CIRCLE_SAMPLES = 32
ARC_SAMPLES = 24

number = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
token_re = re.compile(r"[MLQZCASHVmlqzcashv]|[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?")
COMMANDS = set("MLQZCASHVmlqzcashv")

def sample_quadratic(p0, p1, p2, n):
    pts = []
    for i in range(1, n + 1):
        t = i / n
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
        pts.append([x, y])
    return pts

def sample_cubic(p0, p1, p2, p3, n):
    pts = []
    for i in range(1, n + 1):
        t = i / n
        mt = 1 - t
        x = (
            (mt ** 3) * p0[0]
            + 3 * (mt ** 2) * t * p1[0]
            + 3 * mt * (t ** 2) * p2[0]
            + (t ** 3) * p3[0]
        )
        y = (
            (mt ** 3) * p0[1]
            + 3 * (mt ** 2) * t * p1[1]
            + 3 * mt * (t ** 2) * p2[1]
            + (t ** 3) * p3[1]
        )
        pts.append([x, y])
    return pts

def sample_arc(x1, y1, rx, ry, x_rot, large_arc, sweep, x2, y2, n):
    """
    Convert SVG elliptical arc to sampled points using the
    endpoint-to-centre parameterisation from the SVG spec.
    Returns a list of [x, y] points not including the start point.
    """
    if rx == 0 or ry == 0:
        return [[x2, y2]]

    phi = math.radians(x_rot)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    # Step 1 — compute (x1', y1')
    dx = (x1 - x2) / 2
    dy = (y1 - y2) / 2
    x1p =  cos_phi * dx + sin_phi * dy
    y1p = -sin_phi * dx + cos_phi * dy

    # Step 2 — compute (cx', cy')
    rx = abs(rx)
    ry = abs(ry)
    x1p2 = x1p ** 2
    y1p2 = y1p ** 2
    rx2 = rx ** 2
    ry2 = ry ** 2

    # Ensure radii are large enough
    lam = x1p2 / rx2 + y1p2 / ry2
    if lam > 1:
        s = math.sqrt(lam)
        rx *= s
        ry *= s
        rx2 = rx ** 2
        ry2 = ry ** 2

    num = max(0.0, rx2 * ry2 - rx2 * y1p2 - ry2 * x1p2)
    den = rx2 * y1p2 + ry2 * x1p2
    sq = math.sqrt(num / den) if den != 0 else 0.0
    if large_arc == sweep:
        sq = -sq

    cxp =  sq * rx * y1p / ry
    cyp = -sq * ry * x1p / rx

    # Step 3 — compute (cx, cy)
    cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2
    cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2

    # Step 4 — compute angles
    def angle(ux, uy, vx, vy):
        dot = ux * vx + uy * vy
        mag = math.sqrt((ux**2 + uy**2) * (vx**2 + vy**2))
        if mag == 0:
            return 0.0
        a = math.acos(max(-1.0, min(1.0, dot / mag)))
        if ux * vy - uy * vx < 0:
            a = -a
        return a

    theta1 = angle(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dtheta = angle(
        (x1p - cxp) / rx,  (y1p - cyp) / ry,
        (-x1p - cxp) / rx, (-y1p - cyp) / ry
    )

    if not sweep and dtheta > 0:
        dtheta -= 2 * math.pi
    elif sweep and dtheta < 0:
        dtheta += 2 * math.pi

    pts = []
    for i in range(1, n + 1):
        t = i / n
        angle_t = theta1 + t * dtheta
        x = cos_phi * rx * math.cos(angle_t) - sin_phi * ry * math.sin(angle_t) + cx
        y = sin_phi * rx * math.cos(angle_t) + cos_phi * ry * math.sin(angle_t) + cy
        pts.append([x, y])
    return pts

def parse_path(d):
    tokens = token_re.findall(d.replace(",", " "))
    i = 0
    strokes = []
    stroke = []
    cmd = None
    current = (0.0, 0.0)
    start = None

    def get_float():
        nonlocal i
        val = float(tokens[i])
        i += 1
        return val

    def at_coords():
        """True if the next token is a number, not a command letter."""
        return i < len(tokens) and tokens[i] not in COMMANDS

    last_cp2 = None   # tracks last cubic control point 2 for S/s reflection

    while i < len(tokens):
        if tokens[i] in COMMANDS:
            cmd = tokens[i]
            i += 1
            if cmd in "Zz":
                if stroke and start is not None:
                    stroke.append([start[0], start[1]])
                    strokes.append(stroke)
                stroke = []
                current = start if start is not None else current
                start = None
                last_cp2 = None
                continue
        if cmd not in "CcSs":
            last_cp2 = None

        if cmd is None:
            raise ValueError("Path data missing command")

        if cmd in "Mm":
            x = get_float()
            y = get_float()
            if cmd == "m":
                x += current[0]
                y += current[1]
            current = (x, y)
            start = current
            if stroke:
                strokes.append(stroke)
            stroke = [[x, y]]
            cmd = "l" if cmd == "m" else "L"

        elif cmd in "Ll":
            while at_coords():
                x = get_float()
                y = get_float()
                if cmd == "l":
                    x += current[0]
                    y += current[1]
                current = (x, y)
                stroke.append([x, y])

        elif cmd in "Qq":
            while at_coords():
                x1 = get_float()
                y1 = get_float()
                x2 = get_float()
                y2 = get_float()
                if cmd == "q":
                    x1 += current[0]
                    y1 += current[1]
                    x2 += current[0]
                    y2 += current[1]
                pts = sample_quadratic(current, [x1, y1], [x2, y2], Q_SAMPLES)
                stroke.extend(pts)
                current = (x2, y2)

        elif cmd in "Cc":
            while at_coords():
                x1 = get_float()
                y1 = get_float()
                x2 = get_float()
                y2 = get_float()
                x3 = get_float()
                y3 = get_float()
                if cmd == "c":
                    x1 += current[0]
                    y1 += current[1]
                    x2 += current[0]
                    y2 += current[1]
                    x3 += current[0]
                    y3 += current[1]
                pts = sample_cubic(current, [x1, y1], [x2, y2], [x3, y3], C_SAMPLES)
                stroke.extend(pts)
                last_cp2 = (x2, y2)
                current = (x3, y3)

        elif cmd in "Ss":
            # Smooth cubic — control point 1 is reflection of last C/c/S/s cp2
            while at_coords():
                x2 = get_float()
                y2 = get_float()
                x3 = get_float()
                y3 = get_float()
                if cmd == "s":
                    x2 += current[0]
                    y2 += current[1]
                    x3 += current[0]
                    y3 += current[1]
                # Reflect last_cp2 through current; fall back to current if none
                ref = last_cp2 if last_cp2 is not None else current
                x1 = 2 * current[0] - ref[0]
                y1 = 2 * current[1] - ref[1]
                pts = sample_cubic(current, [x1, y1], [x2, y2], [x3, y3], C_SAMPLES)
                stroke.extend(pts)
                last_cp2 = (x2, y2)
                current = (x3, y3)


        elif cmd in "Hh":
            while at_coords():
                x = get_float()
                if cmd == "h":
                    x += current[0]
                current = (x, current[1])
                stroke.append([x, current[1]])

        elif cmd in "Vv":
            while at_coords():
                y = get_float()
                if cmd == "v":
                    y += current[1]
                current = (current[0], y)
                stroke.append([current[0], y])

        elif cmd in "Aa":
            while at_coords():
                rx     = get_float()
                ry     = get_float()
                x_rot  = get_float()
                large  = int(get_float())
                sweep  = int(get_float())
                x2     = get_float()
                y2     = get_float()
                if cmd == "a":
                    x2 += current[0]
                    y2 += current[1]
                pts = sample_arc(
                    current[0], current[1],
                    rx, ry, x_rot, large, sweep,
                    x2, y2, ARC_SAMPLES
                )
                stroke.extend(pts)
                current = (x2, y2)

        else:
            raise ValueError(f"Unsupported command: {cmd}")

    if stroke:
        strokes.append(stroke)
    return strokes

def parse_circle(el):
    cx = float(el.get("cx"))
    cy = float(el.get("cy"))
    r = float(el.get("r"))
    pts = []
    for i in range(CIRCLE_SAMPLES):
        a = 2 * math.pi * i / CIRCLE_SAMPLES
        pts.append([cx + r * math.cos(a), cy + r * math.sin(a)])
    pts.append(pts[0])
    return [pts]

def parse_polyline(points_text):
    nums = re.findall(number, points_text.replace(",", " "))
    if len(nums) < 4 or len(nums) % 2 != 0:
        return []
    pts = []
    for i in range(0, len(nums), 2):
        pts.append([float(nums[i]), float(nums[i + 1])])
    return [pts] if len(pts) >= 2 else []

def parse_transform(transform_text):
    if not transform_text:
        return []
    ops = []
    for name, args in re.findall(r"(\w+)\(([^)]*)\)", transform_text):
        values = [float(v) for v in re.findall(number, args)]
        ops.append((name, values))
    return ops

def apply_transform_to_point(x, y, ops):
    for name, values in ops:
        if name == "translate":
            tx = values[0] if len(values) > 0 else 0.0
            ty = values[1] if len(values) > 1 else 0.0
            x += tx
            y += ty
        elif name == "scale":
            sx = values[0] if len(values) > 0 else 1.0
            sy = values[1] if len(values) > 1 else sx
            x *= sx
            y *= sy
        elif name == "matrix" and len(values) >= 6:
            a, b, c, d, e, f = values[:6]
            x, y = (a * x + c * y + e, b * x + d * y + f)
    return x, y

def apply_transform(strokes, ops):
    if not ops:
        return strokes
    transformed = []
    for stroke in strokes:
        pts = []
        for x, y in stroke:
            tx, ty = apply_transform_to_point(x, y, ops)
            pts.append([tx, ty])
        transformed.append(pts)
    return transformed

def extract_svg_text(text):
    text = text.lstrip("\ufeff")
    match = re.search(r"<svg\b[^>]*>.*?</svg>", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError("No <svg>...</svg> block found in input")
    return match.group(0)

def svg_to_strokes(svg_text):
    root = ET.fromstring(svg_text)
    strokes = []

    def walk(el, inherited_ops):
        ops = inherited_ops + parse_transform(el.get("transform"))
        tag = el.tag.split("}")[-1]

        if tag == "path":
            d = el.get("d")
            if d:
                strokes.extend(apply_transform(parse_path(d), ops))

        elif tag == "circle":
            strokes.extend(apply_transform(parse_circle(el), ops))

        elif tag == "line":
            x1 = float(el.get("x1", "0"))
            y1 = float(el.get("y1", "0"))
            x2 = float(el.get("x2", "0"))
            y2 = float(el.get("y2", "0"))
            strokes.extend(apply_transform([[[x1, y1], [x2, y2]]], ops))

        elif tag == "polyline":
            pts = el.get("points")
            if pts:
                strokes.extend(apply_transform(parse_polyline(pts), ops))

        for child in el:
            walk(child, ops)

    walk(root, [])
    return strokes

def main():
    if len(sys.argv) != 3:
        print("Usage: svg_to_json.py INPUT_SVG OUTPUT_JSON")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        svg_text = f.read()

    try:
        svg_text = extract_svg_text(svg_text)
    except ValueError:
        pass

    strokes = svg_to_strokes(svg_text)

    payload = {"strokes": strokes}

    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(payload, f)

if __name__ == "__main__":
    main()