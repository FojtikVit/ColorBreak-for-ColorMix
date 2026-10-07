"""Dependency-free previews: orthographic point-splat PNG and binary PLY with face colours."""

from __future__ import annotations

import struct
import zlib
from typing import Sequence

import numpy as np


def _hex_to_rgb8(h: str) -> np.ndarray:
    h = h.lstrip("#")
    if len(h) < 6:
        return np.array([128, 128, 128], dtype=np.uint8)
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)


def colours_from_hex(hex_list: Sequence[str]) -> np.ndarray:
    return np.stack([_hex_to_rgb8(h) for h in hex_list]).astype(np.float64)


def write_png(path: str, rgb: np.ndarray) -> None:
    """Write an (H, W, 3) uint8 array as PNG using only the standard library."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def _splat_view(
    pts: np.ndarray,
    depth: np.ndarray,
    col: np.ndarray,
    px_per_mm: float,
    radius: int,
    margin_px: int = 10,
) -> np.ndarray:
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    size = np.ceil((hi - lo) * px_per_mm).astype(int) + 2 * margin_px
    w, h = int(size[0]), int(size[1])
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    zbuf = np.full((h, w), -np.inf)
    order = np.argsort(depth)  # far first; nearer points overwrite
    xs = ((pts[order, 0] - lo[0]) * px_per_mm).astype(int) + margin_px
    ys = h - 1 - (((pts[order, 1] - lo[1]) * px_per_mm).astype(int) + margin_px)
    d = depth[order]
    c = col[order]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius + 0.5:
                continue
            x = np.clip(xs + dx, 0, w - 1)
            y = np.clip(ys + dy, 0, h - 1)
            # later (nearer) entries win thanks to the sort order of fancy assignment
            img[y, x] = c
            zbuf[y, x] = d
    return img


def densify(
    vertices: np.ndarray, triangles: np.ndarray, face_colours: np.ndarray, pts_per_mm2: float, rng: np.random.Generator
):
    """Sample points over every triangle (count ∝ area) so coarse meshes render without gaps.
    Returns (points (M,3), normals (M,3), colours (M,3))."""
    a, b, c = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    n = np.cross(b - a, c - a)
    nl = np.linalg.norm(n, axis=1)
    areas = 0.5 * nl
    normals = n / np.where(nl > 0, nl, 1.0)[:, None]
    counts = np.maximum(1, np.ceil(areas * pts_per_mm2)).astype(int)
    tri = np.repeat(np.arange(len(triangles)), counts)
    u = rng.random(len(tri))
    v = rng.random(len(tri))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = a[tri] + u[:, None] * (b[tri] - a[tri]) + v[:, None] * (c[tri] - a[tri])
    return pts, normals[tri], face_colours[tri]


def render_views(
    vertices: np.ndarray,
    triangles: np.ndarray,
    face_colours: np.ndarray,
    path: str,
    px_per_mm: float = 14.0,
    views: Sequence[str] = ("front", "side", "top"),
    seed: int = 0,
) -> None:
    """Three orthographic views side by side.  ``face_colours``: (T, 3) in 0..255."""
    rng = np.random.default_rng(seed)
    spacing_px = 1.5
    centroids, normals, colours = densify(vertices, triangles, face_colours, (px_per_mm / spacing_px) ** 2, rng)
    light = np.array([0.3, -0.5, 0.8])
    light /= np.linalg.norm(light)
    shade = 0.7 + 0.3 * np.abs(normals @ light)
    col = np.clip(colours * shade[:, None], 0, 255).astype(np.uint8)
    radius = 1
    panels = []
    for v in views:
        if v == "front":  # look along +y: x right, z up, nearer = smaller y
            pts, depth = centroids[:, [0, 2]], -centroids[:, 1]
        elif v == "back":
            pts, depth = centroids[:, [0, 2]] * np.array([-1, 1]), centroids[:, 1]
        elif v == "side":  # look along -x: y right, z up
            pts, depth = centroids[:, [1, 2]], centroids[:, 0]
        else:  # top
            pts, depth = centroids[:, [0, 1]], centroids[:, 2]
        panels.append(_splat_view(pts, depth, col, px_per_mm, radius))
    h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        pad = np.full((h, p.shape[1], 3), 255, dtype=np.uint8)
        pad[: p.shape[0]] = p
        padded.append(pad)
        padded.append(np.full((h, 8, 3), 230, dtype=np.uint8))
    write_png(path, np.concatenate(padded[:-1], axis=1))


def write_ply(path: str, vertices: np.ndarray, triangles: np.ndarray, face_colours: np.ndarray) -> None:
    """Binary little-endian PLY with per-face RGB (viewable in MeshLab, Blender, ...)."""
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n"
        f"element face {len(triangles)}\nproperty list uchar int vertex_indices\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    face_dt = np.dtype([("n", "u1"), ("v", "<i4", 3), ("rgb", "u1", 3)])
    faces = np.empty(len(triangles), dtype=face_dt)
    faces["n"] = 3
    faces["v"] = triangles.astype(np.int32)
    faces["rgb"] = np.clip(face_colours, 0, 255).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertices.astype("<f4").tobytes())
        f.write(faces.tobytes())
