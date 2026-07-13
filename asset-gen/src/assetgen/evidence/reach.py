"""Exact plan-geometry preparation for unbranched flowing-water evidence."""
from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import split

_TAU_M = 1e-6


def _line_parts(geometry) -> tuple[LineString, ...]:
    if geometry.is_empty:
        return ()
    if geometry.geom_type == "LineString":
        return (geometry,)
    if geometry.geom_type in ("MultiLineString", "GeometryCollection"):
        return tuple(
            line
            for child in geometry.geoms
            for line in _line_parts(child)
        )
    return ()


def _endpoint_tangent(line: LineString, *, start: bool) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(line.coords, dtype=np.float64)
    endpoint = coordinates[0 if start else -1]
    candidates = coordinates[1:] - endpoint if start else endpoint - coordinates[:-1][::-1]
    norms = np.linalg.norm(candidates, axis=1)
    nonzero = np.flatnonzero(norms > 0.0)
    if nonzero.size == 0:
        raise ValueError("centerline endpoint lacks a non-zero segment")
    tangent = candidates[nonzero[0]] / norms[nonzero[0]]
    return endpoint, tangent


def _endpoint_cut(polygon, line: LineString, *, start: bool):
    endpoint, tangent = _endpoint_tangent(line, start=start)
    normal = np.array((-tangent[1], tangent[0]), dtype=np.float64)
    min_x, min_y, max_x, max_y = polygon.bounds
    radius = float(np.hypot(max_x - min_x, max_y - min_y) + 1.0)
    cross_section = LineString((endpoint - radius * normal, endpoint + radius * normal))
    chords = tuple(
        chord
        for chord in _line_parts(polygon.intersection(cross_section))
        if chord.length > 0.0 and chord.distance(Point(endpoint)) <= _TAU_M
    )
    if len(chords) != 1:
        raise ValueError("endpoint cross-section does not identify one local water chord")

    chord = chords[0]
    first = np.asarray(chord.coords[0], dtype=np.float64)
    last = np.asarray(chord.coords[-1], dtype=np.float64)
    direction = last - first
    chord_length = float(np.linalg.norm(direction))
    if not np.isfinite(chord_length) or chord_length <= 0.0:
        raise ValueError("endpoint cross-section chord is degenerate")
    direction /= chord_length
    epsilon = max(1e-6, 1e-9 * radius)
    cutter = LineString((first - epsilon * direction, last + epsilon * direction))
    pieces = tuple(split(polygon, cutter).geoms)
    if len(pieces) == 1 and polygon.boundary.covers(chord):
        return polygon
    midpoint = line.interpolate(line.length * 0.5)
    owners = tuple(piece for piece in pieces if piece.covers(midpoint))
    if len(owners) != 1:
        raise ValueError("endpoint cut does not identify one centreline component")
    return owners[0]


def cap_water_polygon_to_centerline(water_polygon, centerline: LineString):
    """Retain only water between the endpoint cross-sections of one graph reach."""
    polygon = shapely.force_2d(water_polygon)
    line = shapely.force_2d(centerline)
    if (
        polygon.geom_type not in ("Polygon", "MultiPolygon")
        or polygon.is_empty
        or not polygon.is_valid
        or line.geom_type != "LineString"
        or line.is_empty
        or not line.is_valid
        or not line.is_simple
        or line.length < 2.0
        or not polygon.covers(line)
    ):
        raise ValueError("capping requires valid water geometry covering one simple reach")
    capped = _endpoint_cut(polygon, line, start=True)
    capped = _endpoint_cut(capped, line, start=False)
    endpoints = (Point(line.coords[0]), Point(line.coords[-1]))
    if (
        capped.geom_type != "Polygon"
        or not capped.is_valid
        or line.difference(capped).length > _TAU_M
        or any(capped.distance(endpoint) > _TAU_M for endpoint in endpoints)
    ):
        raise ValueError("capped water polygon does not preserve the complete centreline")
    return shapely.normalize(capped)
