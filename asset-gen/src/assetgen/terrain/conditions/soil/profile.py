"""Conservative official-legend grammar for Mullastikukaart profile fields."""
from __future__ import annotations

import re
from typing import Any

PROFILE_GRAMMAR_ID = "laas.mullastikukaart-profile-grammar/1"

_TYPOGRAPHY = str.maketrans(
    {
        "₀": "_0",
        "₁": "_1",
        "₂": "_2",
        "₃": "_3",
        "₄": "_4",
        "₅": "_5",
        "₆": "_6",
        "₇": "_7",
        "₈": "_8",
        "₉": "_9",
        "⁰": "°",
        "¹": "_1",
        "²": "_2",
        "³": "_3",
        "⁴": "_4",
        "⁵": "_5",
        "⁶": "_6",
        "⁷": "_7",
        "⁸": "_8",
        "⁹": "_9",
        "–": "-",
        "—": "-",
        "−": "-",
    }
)
_TEXTURES = {
    "l": ("sand", None),
    "pl": ("fine_sand", None),
    "tpl": ("silty_fine_sand", None),
    "tl": ("silty_sand", None),
    "sl": ("sandy_loam", None),
    "plsl": ("fine_sandy_loam", None),
    "tsl": ("silty_sandy_loam", None),
    "ls": ("loam", None),
    "ls_1": ("loam", 1),
    "ls_2": ("loam", 2),
    "ls_3": ("loam", 3),
    "plls": ("fine_sandy_loam_to_loam", None),
    "tls": ("silty_loam", None),
    "s": ("clay", None),
}
_SKELETON = {
    "kr": "gravel",
    "dk": "sandstone_fragments",
    "r": "limestone_fragments",
    "v": "limestone_pebbles",
    "v°": "crystalline_pebbles",
    "kb": "shingle",
    "ck": "oil_shale_fragments",
    "k": "limestone_stones",
    "k°": "crystalline_stones",
    "pk": "limestone_slabs",
    "p": "limestone_bedrock",
    "d": "sandstone_bedrock",
    "lu": "calcareous_sediment",
}
_UNGRADED_SKELETON = {"kr", "dk", "pk", "p", "d", "lu"}
_SKELETON_PREFIXES = tuple(sorted(_SKELETON, key=len, reverse=True))
_TEXTURE_SUFFIXES = tuple(sorted(_TEXTURES, key=len, reverse=True))
_DEPTH = re.compile(r"^(\d+)(?:-(\d+))?$")
_FINAL_TOP_DEPTH = re.compile(r"^(.*)\((\d+(?:-\d+)?)\)$")


def _canonical(raw: str) -> str:
    folded = raw.translate(_TYPOGRAPHY).strip()
    return re.sub(r"\s*([/;+])\s*", r"\1", folded)


def _split_top_level(raw: str) -> list[str]:
    pieces: list[str] = []
    token: list[str] = []
    depth = 0
    for character in raw:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                return [raw]
        is_separator = depth == 0 and (character == ";" or character.isspace())
        if is_separator:
            if token:
                pieces.append("".join(token))
                token = []
            if character == ";" or not pieces or not pieces[-1].isspace():
                pieces.append(character)
        else:
            token.append(character)
    if token:
        pieces.append("".join(token))
    return pieces if depth == 0 else [raw]


def _interval(raw: str) -> dict[str, int] | None:
    match = _DEPTH.fullmatch(raw)
    if match is None:
        return None
    low = int(match.group(1))
    high = int(match.group(2) or low)
    if low > high:
        return None
    return {"min_cm": low, "max_cm": high}


def _texture(code: str) -> dict[str, Any] | None:
    spec = _TEXTURES.get(code)
    if spec is None:
        return None
    family, grade = spec
    result: dict[str, Any] = {"code": code, "family": family}
    if grade is not None:
        result["loam_grade"] = grade
    return result


def _skeleton_components(raw: str) -> list[dict[str, Any]] | None:
    components: list[dict[str, Any]] = []
    remainder = raw
    while remainder:
        family_code = next(
            (code for code in _SKELETON_PREFIXES if remainder.startswith(code)),
            None,
        )
        if family_code is None:
            return None
        remainder = remainder[len(family_code) :]
        grade: int | None = None
        grade_match = re.match(r"^_([1-5])", remainder)
        if grade_match is not None:
            grade = int(grade_match.group(1))
            remainder = remainder[grade_match.end() :]
        if grade is not None and family_code in _UNGRADED_SKELETON:
            return None
        component: dict[str, Any] = {
            "code": family_code,
            "family": _SKELETON[family_code],
        }
        if grade is None:
            component["content_class"] = "ungraded_or_over_70_percent"
        else:
            component["content_class"] = grade
            component["volume_percent_range"] = {
                1: [2, 10],
                2: [10, 20],
                3: [20, 30],
                4: [30, 50],
                5: [50, 70],
            }[grade]
        components.append(component)
    return components or None


def _material_exact(raw: str) -> dict[str, Any] | None:
    carbonate = raw.startswith("+")
    token = raw[1:] if carbonate else raw
    cemented = token.endswith("m")
    token = token[:-1] if cemented else token
    if not token:
        return None

    direct_texture = _texture(token)
    if direct_texture is not None:
        return {
            "kind": "mineral_fine_earth",
            "texture": direct_texture,
            "skeleton": None,
            "carbonate": carbonate,
            "cemented": cemented,
        }
    peat = re.fullmatch(r"t(?:_([123]))?", token)
    if peat is not None:
        result: dict[str, Any] = {
            "kind": "peat",
            "carbonate": carbonate,
            "cemented": cemented,
        }
        if peat.group(1) is not None:
            result["decomposition_class"] = int(peat.group(1))
        return result

    options: list[tuple[list[dict[str, Any]], dict[str, Any] | None]] = []
    standalone = _skeleton_components(token)
    if standalone is not None:
        options.append((standalone, None))
    for texture_code in _TEXTURE_SUFFIXES:
        if not token.endswith(texture_code) or token == texture_code:
            continue
        skeleton = _skeleton_components(token[: -len(texture_code)])
        if skeleton is not None:
            options.append((skeleton, _texture(texture_code)))
    if options:
        skeleton_components, fine_earth = max(
            options,
            key=lambda option: len(option[0]),
        )
        result: dict[str, Any] = {
            "kind": "mineral_skeletal_material",
            "texture": fine_earth,
            "skeleton": skeleton_components[0] if len(skeleton_components) == 1 else None,
            "carbonate": carbonate
            or any(
                component["code"] in {"r", "v", "kb", "k", "pk", "p", "lu"}
                for component in skeleton_components
            ),
            "cemented": cemented,
        }
        if len(skeleton_components) > 1:
            result["skeleton_components"] = skeleton_components
            result["skeleton_relation"] = "recorded_combination_not_collapsed"
        return result
    return None


def _material_alternatives(raw: str) -> list[dict[str, Any]] | None:
    shorthand = re.fullmatch(r"(.*)_([0-9])(?:,|-)_([0-9])(.*)", raw)
    candidates: list[str]
    if shorthand is not None:
        candidates = [
            f"{shorthand.group(1)}_{grade}{shorthand.group(4)}"
            for grade in (shorthand.group(2), shorthand.group(3))
        ]
    elif "," in raw:
        candidates = raw.split(",")
    else:
        return None
    parsed = [_material_exact(candidate) for candidate in candidates]
    if any(item is None for item in parsed):
        return None
    return [
        {"source_code": candidate, "material": item}
        for candidate, item in zip(candidates, parsed, strict=True)
    ]


def _material_transition(raw: str) -> dict[str, Any] | None:
    if raw.count("-") != 1:
        return None
    start, end = raw.split("-", 1)
    start_material = _material_exact(start)
    end_material = _material_exact(end)
    if start_material is None or end_material is None:
        return None
    return {
        "from": {"source_code": start, "material": start_material},
        "to": {"source_code": end, "material": end_material},
    }


def _material_with_depth(
    raw: str,
) -> tuple[str, dict[str, Any], dict[str, int]] | None:
    candidates: list[tuple[str, dict[str, Any], dict[str, int]]] = []
    for split_at in range(1, len(raw)):
        depth = _interval(raw[split_at:])
        material = _material_exact(raw[:split_at])
        if depth is not None and material is not None:
            candidates.append((raw[:split_at], material, depth))
    return max(candidates, key=lambda item: len(item[0])) if candidates else None


def _qualified_material_with_depth(
    raw: str,
) -> tuple[
    str,
    list[dict[str, Any]] | dict[str, Any],
    dict[str, int],
    str,
] | None:
    candidates = []
    for split_at in range(1, len(raw)):
        depth = _interval(raw[split_at:])
        if depth is None:
            continue
        prefix = raw[:split_at]
        alternatives = _material_alternatives(prefix)
        if alternatives is not None:
            candidates.append((prefix, alternatives, depth, "alternatives"))
            continue
        transition = _material_transition(prefix)
        if transition is not None:
            candidates.append((prefix, transition, depth, "transition"))
    return max(candidates, key=lambda item: len(item[0])) if candidates else None


def _parse_layer_token(raw: str) -> dict[str, Any]:
    token = raw
    carbonate_onset: dict[str, int] | None = None
    carbonate_match = re.fullmatch(r"(.+)\+(\d+(?:-\d+)?)", token)
    if carbonate_match is not None:
        candidate_material = _material_exact(carbonate_match.group(1))
        candidate_depth = _interval(carbonate_match.group(2))
        if candidate_material is not None and candidate_depth is not None:
            token = carbonate_match.group(1)
            carbonate_onset = candidate_depth
    parenthesized_top: dict[str, int] | None = None
    top_match = _FINAL_TOP_DEPTH.fullmatch(token)
    if top_match is not None:
        candidate = _interval(top_match.group(2))
        if candidate is not None:
            parenthesized_top = candidate
            token = top_match.group(1)

    material = _material_exact(token)
    thickness: dict[str, int] | None = None
    if material is None:
        candidate = _material_with_depth(token)
        if candidate is not None:
            token, material, thickness = candidate
    alternatives = None if material is not None else _material_alternatives(token)
    transition = None
    if material is None and alternatives is None:
        transition = _material_transition(token)
    if material is None and alternatives is None and transition is None:
        qualified = _qualified_material_with_depth(token)
        if qualified is not None:
            token, qualified_material, thickness, qualified_kind = qualified
            if qualified_kind == "alternatives":
                alternatives = qualified_material
            else:
                transition = qualified_material
    mixture = None
    if material is None and alternatives is None and transition is None:
        mixture_match = re.fullmatch(r"(.+)\(([^()]+)\)", token)
        if mixture_match is not None:
            primary = _material_exact(mixture_match.group(1))
            minor = _material_exact(mixture_match.group(2))
            if primary is not None and minor is not None:
                mixture = {
                    "primary": {
                        "source_code": mixture_match.group(1),
                        "material": primary,
                    },
                    "minor": {
                        "source_code": mixture_match.group(2),
                        "material": minor,
                        "share_percent_range": [10, 20],
                    },
                }
    if material is None and alternatives is None and transition is None and mixture is None:
        return {
            "status": "unparseable_preserved",
            "raw": raw,
            "residual": raw,
        }
    result: dict[str, Any] = {
        "status": "parsed_complete_official_grammar",
        "raw": raw,
        "source_material_code": token,
    }
    if material is not None:
        result["material"] = material
    elif alternatives is not None:
        result["material_alternatives"] = alternatives
        result["qualifier"] = "explicit_material_alternatives_not_collapsed"
    elif transition is not None:
        result["material_transition"] = transition
        result["qualifier"] = "explicit_texture_transition_not_collapsed"
    else:
        result["material_mixture"] = mixture
        result["qualifier"] = "parenthesized_minor_component_10_to_20_percent"
    if thickness is not None:
        result["numeric_suffix_cm"] = thickness
    if parenthesized_top is not None:
        result["parenthesized_top_depth_cm"] = parenthesized_top
    if carbonate_onset is not None:
        result["carbonate_onset_depth_cm"] = carbonate_onset
        result["carbonate_qualifier"] = "effervescence_begins_at_recorded_depth"
    return result


def _add_intervals(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {
        "min_cm": left["min_cm"] + right["min_cm"],
        "max_cm": left["max_cm"] + right["max_cm"],
    }


def _parse_profile_expression(raw: str) -> dict[str, Any]:
    minor_component = raw.startswith("(") and raw.endswith(")")
    expression = raw[1:-1] if minor_component else raw
    layer_tokens = expression.split("/")
    if not expression or any(not token for token in layer_tokens):
        return {"status": "unparseable_preserved", "raw": raw, "residual": raw}
    layers = [_parse_layer_token(token) for token in layer_tokens]
    if any(layer["status"] != "parsed_complete_official_grammar" for layer in layers):
        return {
            "status": "unparseable_preserved",
            "raw": raw,
            "layers": layers,
            "residuals": [
                layer["residual"]
                for layer in layers
                if layer["status"] == "unparseable_preserved"
            ],
        }

    all_peat = all(
        layer.get("material", {}).get("kind") == "peat" for layer in layers
    )
    final_numeric = layers[-1].pop("numeric_suffix_cm", None)
    final_peat_thickness = final_numeric if all_peat and len(layers) > 1 else None
    research_depth = (
        None
        if final_peat_thickness is not None
        else final_numeric or {"min_cm": 100, "max_cm": 100}
    )
    research_depth_source = (
        "sum_recorded_peat_layer_thicknesses"
        if final_peat_thickness is not None
        else "explicit_final_layer_suffix"
        if final_numeric
        else "implicit_1m"
    )
    current_top: dict[str, int] | None = {"min_cm": 0, "max_cm": 0}
    for index, layer in enumerate(layers):
        is_final = index == len(layers) - 1
        explicit_top = layer.pop("parenthesized_top_depth_cm", None)
        if explicit_top is not None:
            if not is_final or len(layers) < 3:
                return {
                    "status": "unparseable_preserved",
                    "raw": raw,
                    "layers": layers,
                    "residuals": [layer["raw"]],
                    "reason": "parenthesized_depth_is_only_defined_for_final_layer_of_3plus_profile",
                }
            current_top = explicit_top
            layer["top_depth_source"] = "explicit_parenthesized_final_layer_top"
        layer["top_depth_cm"] = current_top
        if is_final:
            if final_peat_thickness is not None:
                layer["thickness_cm"] = final_peat_thickness
                research_depth = (
                    _add_intervals(current_top, final_peat_thickness)
                    if current_top is not None
                    else None
                )
                layer["bottom_depth_cm"] = research_depth
                layer["bottom_depth_source"] = research_depth_source
            elif (
                research_depth_source == "implicit_1m"
                and current_top is not None
                and current_top["min_cm"] > 100
            ):
                research_depth = None
                research_depth_source = (
                    "unspecified_beyond_recorded_layer_boundary_over_1m"
                )
                layer["bottom_depth_cm"] = None
                layer["bottom_depth_source"] = research_depth_source
            elif (
                current_top is not None
                and research_depth is not None
                and current_top["min_cm"] > research_depth["max_cm"]
            ):
                return {
                    "status": "unparseable_preserved",
                    "raw": raw,
                    "layers": layers,
                    "residuals": [layer["raw"]],
                    "reason": "final_layer_starts_below_recorded_research_depth",
                }
            else:
                layer["bottom_depth_cm"] = research_depth
                layer["bottom_depth_source"] = research_depth_source
            continue
        thickness = layer.pop("numeric_suffix_cm", None)
        layer["thickness_cm"] = thickness
        if thickness is None or current_top is None:
            layer["bottom_depth_cm"] = None
            layer["bottom_depth_source"] = "unspecified_in_source"
            current_top = None
        else:
            current_top = _add_intervals(current_top, thickness)
            layer["bottom_depth_cm"] = current_top
            layer["bottom_depth_source"] = "cumulative_recorded_layer_thickness"

    result: dict[str, Any] = {
        "status": "parsed_complete_official_grammar",
        "raw": raw,
        "layers": layers,
        "research_depth_cm": research_depth,
        "research_depth_source": research_depth_source,
    }
    if minor_component:
        result["component_qualifier"] = "parenthesized_minor_component_10_to_20_percent"
    return result


def parse_texture_profile(raw: Any) -> dict[str, Any]:
    """Parse Loimis1/Loimis2 without inventing unresolved component relationships."""
    if raw is None:
        return {"status": "missing", "grammar": PROFILE_GRAMMAR_ID}
    source = str(raw)
    canonical = _canonical(source)
    pieces = _split_top_level(canonical)
    expressions: list[dict[str, Any]] = []
    separators: list[dict[str, str]] = []
    for piece in pieces:
        if not piece:
            continue
        if piece == ";":
            separators.append(
                {"raw": piece, "semantics": "complex_component_boundary"}
            )
        elif piece.isspace():
            separators.append(
                {
                    "raw": piece,
                    "semantics": "recorded_variant_boundary_source_semantics_not_explicit",
                }
            )
        else:
            expressions.append(_parse_profile_expression(piece))
    complete = bool(expressions) and all(
        expression["status"] == "parsed_complete_official_grammar"
        for expression in expressions
    )
    return {
        "status": (
            "parsed_complete_official_grammar" if complete else "unparseable_preserved"
        ),
        "grammar": PROFILE_GRAMMAR_ID,
        "raw": source,
        "canonical": canonical,
        "expressions": expressions,
        "separators": separators,
        "residuals": [
            residual
            for expression in expressions
            if expression["status"] == "unparseable_preserved"
            for residual in expression.get("residuals", [expression.get("residual", expression["raw"])])
        ],
    }


def _parse_horizon_atom(raw: str) -> dict[str, Any] | None:
    discontinuous = raw.startswith("(") and raw.endswith(")")
    token = raw[1:-1] if discontinuous else raw
    if token == "0":
        result: dict[str, Any] = {"kind": "absent_horizon"}
    else:
        match = re.fullmatch(
            r"(th|t(?:_([123])(?:,_([123]))?)?)?(\d+(?:-\d+)?)?",
            token,
        )
        if match is None or (match.group(1) is None and match.group(4) is None):
            return None
        thickness = _interval(match.group(4)) if match.group(4) else None
        if match.group(4) and thickness is None:
            return None
        prefix = match.group(1)
        if prefix == "th":
            kind = "raw_humus_horizon"
        elif prefix and prefix.startswith("t"):
            kind = "peat_horizon"
        else:
            kind = "humus_horizon"
        result = {"kind": kind, "thickness_cm": thickness}
        if match.group(2) is not None:
            result["decomposition_class"] = int(match.group(2))
        if match.group(3) is not None:
            result["decomposition_classes"] = [
                int(match.group(2)),
                int(match.group(3)),
            ]
            result.pop("decomposition_class", None)
            result["qualifier"] = "explicit_decomposition_alternatives_not_collapsed"
    if discontinuous:
        result["qualifier"] = "parenthesized_discontinuous_or_weakly_developed"
    return result


def _parse_litter_term(raw: str) -> dict[str, Any] | None:
    match = re.fullmatch(r"(\d+(?:-\d+)?)_([123])", raw)
    if match is None:
        return None
    thickness = _interval(match.group(1))
    if thickness is None:
        return None
    return {
        "kind": "forest_litter_layer",
        "thickness_cm": thickness,
        "decomposition_class": int(match.group(2)),
    }


def _parse_horizon_sequence(raw: str) -> dict[str, Any] | None:
    terms = raw.split("+")
    parsed: list[dict[str, Any]] = []
    for term in terms:
        item = _parse_litter_term(term) or _parse_horizon_atom(term)
        if item is None:
            return None
        parsed.append({"raw": term, **item})
    return {"raw": raw, "terms": parsed}


def _parse_humus_expression(raw: str) -> dict[str, Any]:
    sides = raw.split("/")
    if len(sides) > 2 or any(not side for side in sides):
        return {"status": "unparseable_preserved", "raw": raw, "residual": raw}
    parsed = [_parse_horizon_sequence(side) for side in sides]
    if any(side is None for side in parsed):
        return {"status": "unparseable_preserved", "raw": raw, "residual": raw}
    sequences = [side for side in parsed if side is not None]
    result: dict[str, Any] = {
        "status": "parsed_complete_official_grammar",
        "raw": raw,
        "sequences": sequences,
    }
    if len(sequences) == 2:
        left_terms = sequences[0]["terms"]
        right_terms = sequences[1]["terms"]
        if any(term["kind"] == "forest_litter_layer" for term in left_terms):
            relation = "forest_litter_over_humus_or_raw_humus_horizon"
        elif any(term["kind"] == "peat_horizon" for term in left_terms):
            relation = "peat_over_humus_or_raw_humus_horizon"
        elif all(
            term["kind"] in {"humus_horizon", "absent_horizon"}
            for term in left_terms
        ) and all(
            term["kind"] in {"humus_horizon", "raw_humus_horizon", "absent_horizon"}
            for term in right_terms
        ):
            relation = "cultivated_then_natural_land_horizon_alternation"
        else:
            relation = "recorded_horizon_pair_semantics_not_further_inferred"
        result["sequence_relation"] = relation
    return result


def parse_humus_profile(raw: Any) -> dict[str, Any]:
    """Parse Huumus horizon formulae while preserving every recorded variant."""
    if raw is None:
        return {"status": "missing", "grammar": PROFILE_GRAMMAR_ID}
    source = str(raw)
    canonical = _canonical(source)
    pieces = _split_top_level(canonical)
    expressions: list[dict[str, Any]] = []
    separators: list[dict[str, str]] = []
    for piece in pieces:
        if not piece:
            continue
        if piece == ";":
            separators.append({"raw": piece, "semantics": "complex_component_boundary"})
        elif piece.isspace():
            separators.append(
                {
                    "raw": piece,
                    "semantics": "recorded_variant_boundary_source_semantics_not_explicit",
                }
            )
        else:
            expressions.append(_parse_humus_expression(piece))
    complete = bool(expressions) and all(
        expression["status"] == "parsed_complete_official_grammar"
        for expression in expressions
    )
    return {
        "status": (
            "parsed_complete_official_grammar" if complete else "unparseable_preserved"
        ),
        "grammar": PROFILE_GRAMMAR_ID,
        "raw": source,
        "canonical": canonical,
        "expressions": expressions,
        "separators": separators,
        "residuals": [
            expression.get("residual", expression["raw"])
            for expression in expressions
            if expression["status"] == "unparseable_preserved"
        ],
    }
