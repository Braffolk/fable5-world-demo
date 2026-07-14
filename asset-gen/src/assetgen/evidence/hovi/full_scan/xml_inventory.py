"""Namespace-aware, point-free inventory of ASTM E57 XML metadata."""
from __future__ import annotations

import io
import math
import re
import struct
import xml.etree.ElementTree as ET
from decimal import Decimal, InvalidOperation


E57_NAMESPACE = "http://www.astm.org/COMMIT/E57/2010-e57-v1.0"
MAX_XML_BYTES = 50 * 1024 * 1024
MAX_XML_DEPTH = 256
MAX_XML_ELEMENTS = 1_000_000
_FORBIDDEN_DECLARATION = re.compile(
    rb"<!\s*(?:DOCTYPE|ENTITY|ELEMENT|ATTLIST|NOTATION)\b|"
    rb"\b(?:SYSTEM|PUBLIC)\s*['\"]",
    re.IGNORECASE,
)
_NUMERIC_TYPES = {"Integer", "ScaledInteger", "Float"}
_PROTOTYPE_TYPES = _NUMERIC_TYPES | {"String", "Structure", "Vector"}
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_UINT64_MAX = (1 << 64) - 1
_FLOAT_MAX = {
    "single": Decimal("3.4028234663852886E+38"),
    "double": Decimal("1.7976931348623157E+308"),
}
_XSD_FLOAT = re.compile(
    r"[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)"
)
_STANDARD_POINT_FIELDS = {
    "cartesianX",
    "cartesianY",
    "cartesianZ",
    "cartesianInvalidState",
    "sphericalRange",
    "sphericalAzimuth",
    "sphericalElevation",
    "sphericalInvalidState",
    "intensity",
    "isIntensityInvalid",
    "colorRed",
    "colorGreen",
    "colorBlue",
    "isColorInvalid",
    "rowIndex",
    "columnIndex",
    "returnCount",
    "returnIndex",
    "timeStamp",
    "isTimeStampInvalid",
}
_INVALID_STATE_FIELDS = (
    "cartesianInvalidState",
    "sphericalInvalidState",
    "isIntensityInvalid",
    "isColorInvalid",
    "isTimeStampInvalid",
)
_INDEX_FIELDS = ("rowIndex", "columnIndex", "returnCount", "returnIndex")
_STANDARD_LINE_GROUP_FIELDS = {
    "idElementValue",
    "startPointIndex",
    "pointCount",
}


def _qname(name: str) -> tuple[str | None, str]:
    if name.startswith("{"):
        namespace, local = name[1:].split("}", 1)
        return namespace, local
    return None, name


def _attribute_list(element: ET.Element) -> list[dict]:
    attributes = []
    for name, value in element.attrib.items():
        namespace, local = _qname(name)
        attributes.append(
            {"namespaceUri": namespace, "localName": local, "value": value}
        )
    return sorted(
        attributes,
        key=lambda value: (value["namespaceUri"] or "", value["localName"]),
    )


def _node_identity(element: ET.Element) -> dict:
    namespace, local = _qname(element.tag)
    return {
        "namespaceUri": namespace,
        "localName": local,
        "attributes": _attribute_list(element),
    }


def _presence(value: dict | None) -> dict:
    return {"present": False} if value is None else {"present": True, **value}


def _declared_attribute(element: ET.Element, name: str) -> dict:
    if name not in element.attrib:
        return {"present": False}
    return {"present": True, "value": element.attrib[name]}


def _parse_integer(value: str, label: str) -> int:
    lexical = value.strip()
    if not lexical or re.fullmatch(r"[+-]?\d+", lexical) is None:
        raise ValueError(f"{label} is not a valid E57 Integer")
    parsed = int(lexical, 10)
    if parsed < _INT64_MIN or parsed > _INT64_MAX:
        raise ValueError(f"{label} is outside the signed 64-bit E57 Integer range")
    return parsed


def _parse_uint64(value: str, label: str) -> int:
    lexical = value.strip()
    if not lexical or re.fullmatch(r"\+?\d+", lexical) is None:
        raise ValueError(f"{label} is not a valid nonnegative 64-bit integer")
    parsed = int(lexical, 10)
    if parsed > _UINT64_MAX:
        raise ValueError(f"{label} is outside the unsigned 64-bit range")
    return parsed


def _parse_float(value: str, label: str, precision: str) -> Decimal:
    lexical = value.strip()
    if _XSD_FLOAT.fullmatch(lexical) is None:
        raise ValueError(f"{label} is not an XSD float/double lexical value")
    try:
        parsed = Decimal(lexical)
    except InvalidOperation as error:
        raise ValueError(f"{label} is not a valid E57 Float") from error
    if not parsed.is_finite():
        raise ValueError(f"{label} must be finite")
    if parsed.is_zero() and parsed.is_signed():
        raise ValueError(f"{label} may not encode negative zero")
    if abs(parsed) > _FLOAT_MAX[precision]:
        raise ValueError(f"{label} is outside E57 {precision}-precision range")
    converted = float(parsed)
    if not math.isfinite(converted):
        raise ValueError(f"{label} is not finite as binary floating point")
    if precision == "single":
        try:
            converted = struct.unpack(">f", struct.pack(">f", converted))[0]
        except OverflowError as error:
            raise ValueError(f"{label} is not representable as binary32") from error
    if parsed and converted == 0.0:
        raise ValueError(f"{label} underflows its declared {precision} precision")
    return parsed


def _numeric_attribute(
    element: ET.Element,
    name: str,
    parser,
    label: str,
    default,
    default_source: str,
) -> tuple[dict, object]:
    declared = _declared_attribute(element, name)
    if declared["present"]:
        lexical = declared["value"]
        parsed = parser(lexical, f"{label}.{name}")
        return (
            {
                "present": True,
                "lexical": lexical,
                "value": str(parsed) if isinstance(parsed, Decimal) else parsed,
                "effectiveValue": str(parsed) if isinstance(parsed, Decimal) else parsed,
                "effectiveSource": "publisher_attribute",
            },
            parsed,
        )
    return (
        {
            "present": False,
            "effectiveValue": str(default) if isinstance(default, Decimal) else default,
            "effectiveSource": default_source,
        },
        default,
    )


class _InventoryContext:
    def __init__(self) -> None:
        self.used: set[int] = set()

    def mark(self, element: ET.Element) -> ET.Element:
        self.used.add(id(element))
        return element


def _require_type(element: ET.Element, expected: str | set[str], label: str) -> str:
    actual = element.attrib.get("type")
    allowed = {expected} if isinstance(expected, str) else expected
    if actual not in allowed:
        raise ValueError(f"{label} has type {actual!r}; expected {sorted(allowed)}")
    return actual


def _direct_children(parent: ET.Element, local: str) -> list[ET.Element]:
    tag = f"{{{E57_NAMESPACE}}}{local}"
    return [child for child in parent if child.tag == tag]


def _single(
    parent: ET.Element,
    local: str,
    expected_type: str | set[str],
    context: _InventoryContext,
    *,
    required: bool = False,
) -> ET.Element | None:
    matches = _direct_children(parent, local)
    if len(matches) > 1:
        raise ValueError(f"E57 element {local!r} is duplicated")
    if not matches:
        if required:
            raise ValueError(f"required E57 element {local!r} is absent")
        return None
    element = context.mark(matches[0])
    _require_type(element, expected_type, local)
    return element


def _scalar(
    element: ET.Element,
    allowed_types: set[str],
    label: str,
    *,
    allow_empty_value: bool = False,
) -> dict:
    data_type = _require_type(element, allowed_types, label)
    text = element.text or ""
    result = {**_node_identity(element), "type": data_type, "text": text}
    if data_type == "String":
        result["value"] = text
        result["valueSource"] = "publisher_terminal"
        return result
    empty_value = not text.strip()
    if empty_value and not allow_empty_value:
        raise ValueError(f"{label} lacks its required terminal value")
    result["valuePresent"] = not empty_value
    result["valueSource"] = (
        "pinned_e57_0_11_13_xml_omitted_terminal_zero"
        if empty_value
        else "publisher_terminal"
    )
    if data_type == "Integer":
        value = 0 if empty_value else _parse_integer(text, label)
        minimum, minimum_value = _numeric_attribute(
            element,
            "minimum",
            _parse_integer,
            label,
            _INT64_MIN,
            "pinned_reader_integer_default_int64_minimum",
        )
        maximum, maximum_value = _numeric_attribute(
            element,
            "maximum",
            _parse_integer,
            label,
            _INT64_MAX,
            "pinned_reader_integer_default_int64_maximum",
        )
        if minimum_value > maximum_value or not minimum_value <= value <= maximum_value:
            raise ValueError(f"{label} Integer value is outside its declared range")
        result.update(value=value, minimum=minimum, maximum=maximum)
        return result
    if data_type == "ScaledInteger":
        value = 0 if empty_value else _parse_integer(text, label)
        minimum, minimum_value = _numeric_attribute(
            element,
            "minimum",
            _parse_integer,
            label,
            _INT64_MIN,
            "pinned_reader_scaled_integer_default_int64_minimum",
        )
        maximum, maximum_value = _numeric_attribute(
            element,
            "maximum",
            _parse_integer,
            label,
            _INT64_MAX,
            "pinned_reader_scaled_integer_default_int64_maximum",
        )
        parse_double = lambda lexical, field: _parse_float(lexical, field, "double")
        scale, scale_value = _numeric_attribute(
            element,
            "scale",
            parse_double,
            label,
            Decimal(1),
            "pinned_reader_scaled_integer_default_scale_one",
        )
        offset, offset_value = _numeric_attribute(
            element,
            "offset",
            parse_double,
            label,
            Decimal(0),
            "pinned_reader_scaled_integer_default_offset_zero",
        )
        if minimum_value > maximum_value or not minimum_value <= value <= maximum_value:
            raise ValueError(f"{label} ScaledInteger raw value is outside its range")
        if scale_value <= 0:
            raise ValueError(f"{label}.scale must be positive")
        physical_value = Decimal(value) * scale_value + offset_value
        result.update(
            value=value,
            valueDomain="raw_signed_integer",
            effectivePhysicalValue=str(physical_value),
            effectivePhysicalFormula="raw_value * scale + offset",
            minimum=minimum,
            maximum=maximum,
            scale=scale,
            offset=offset,
        )
        return result
    if data_type == "Float":
        precision = _declared_attribute(element, "precision")
        if precision["present"] and precision["value"] not in {"single", "double"}:
            raise ValueError(f"{label}.precision is not single or double")
        effective_precision = precision.get("value", "double")
        precision = {
            **precision,
            "effectiveValue": effective_precision,
            "effectiveSource": (
                "publisher_attribute"
                if precision["present"]
                else "pinned_reader_float_default_double"
            ),
        }
        parse_precision = lambda lexical, field: _parse_float(
            lexical,
            field,
            effective_precision,
        )
        limit = _FLOAT_MAX[effective_precision]
        minimum, minimum_value = _numeric_attribute(
            element,
            "minimum",
            parse_precision,
            label,
            -limit,
            f"pinned_reader_float_default_{effective_precision}_minimum",
        )
        maximum, maximum_value = _numeric_attribute(
            element,
            "maximum",
            parse_precision,
            label,
            limit,
            f"pinned_reader_float_default_{effective_precision}_maximum",
        )
        value_decimal = Decimal(0) if empty_value else parse_precision(text, label)
        if minimum_value > maximum_value or not minimum_value <= value_decimal <= maximum_value:
            raise ValueError(f"{label} Float value is outside its declared range")
        result.update(
            value=0 if empty_value else text.strip(),
            semanticDecimal=str(value_decimal),
            precision=precision,
            minimum=minimum,
            maximum=maximum,
        )
        return result
    raise ValueError(f"unsupported scalar type at {label}: {data_type}")


def _optional_scalar(
    parent: ET.Element,
    local: str,
    allowed_types: set[str],
    context: _InventoryContext,
    *,
    required: bool = False,
    allow_empty_value: bool = True,
) -> dict:
    element = _single(parent, local, allowed_types, context, required=required)
    return _presence(
        None
        if element is None
        else _scalar(
            element,
            allowed_types,
            local,
            allow_empty_value=allow_empty_value,
        )
    )


def _prototype_node(
    prototype: ET.Element,
    context: _InventoryContext,
    label: str,
    *,
    record_kind: str,
    depth: int = 0,
) -> dict:
    declared_type = _require_type(prototype, _PROTOTYPE_TYPES, label)
    namespace, local = _qname(prototype.tag)
    standard_fields = (
        _STANDARD_POINT_FIELDS
        if record_kind == "point"
        else _STANDARD_LINE_GROUP_FIELDS
    )
    result = {
        **_node_identity(prototype),
        "type": declared_type,
        "standardized": (
            depth == 1
            and namespace == E57_NAMESPACE
            and local in standard_fields
        ),
        "standardRecordKind": record_kind,
    }
    if declared_type in _NUMERIC_TYPES | {"String"}:
        result.update(
            _scalar(
                prototype,
                _NUMERIC_TYPES | {"String"},
                label,
                allow_empty_value=True,
            )
        )
        return result
    children = []
    seen: set[tuple[str | None, str]] = set()
    for index, child in enumerate(prototype):
        if not isinstance(child.tag, str):
            continue
        context.mark(child)
        identity = _qname(child.tag)
        if declared_type == "Structure" and identity in seen:
            raise ValueError(f"duplicate prototype field {identity!r} in {label}")
        seen.add(identity)
        child_type = child.attrib.get("type")
        if child_type in {"Blob", "CompressedVector"} or child_type not in _PROTOTYPE_TYPES:
            raise ValueError(
                f"unsupported compressed-vector prototype node type {child_type!r}"
            )
        node = _prototype_node(
            child,
            context,
            f"{label}[{index}]",
            record_kind=record_kind,
            depth=depth + 1,
        )
        node["index"] = index
        children.append(node)
    result["children"] = children
    return result


def _component_structure(
    scan: ET.Element,
    local: str,
    components: tuple[str, ...],
    allowed_types: set[str],
    context: _InventoryContext,
    *,
    coordinate_frame: str,
    ordered_pairs: tuple[tuple[str, str], ...],
) -> dict:
    structure = _single(scan, local, "Structure", context)
    if structure is None:
        return {"present": False}
    values = {
        component: _optional_scalar(
            structure,
            component,
            allowed_types,
            context,
            allow_empty_value=True,
        )
        for component in components
    }
    result = {
        "present": True,
        **_node_identity(structure),
        "type": "Structure",
        "components": values,
        "provenance": "publisher_declared_xml_metadata",
        "coordinateFrame": coordinate_frame,
        "validatedAgainstCompressedVectorRecords": False,
    }
    for minimum_name, maximum_name in ordered_pairs:
        minimum = values[minimum_name]
        maximum = values[maximum_name]
        if minimum["present"] and maximum["present"]:
            minimum_value = Decimal(str(minimum["semanticDecimal"])) if minimum["type"] == "Float" else Decimal(minimum["value"])
            maximum_value = Decimal(str(maximum["semanticDecimal"])) if maximum["type"] == "Float" else Decimal(maximum["value"])
            if minimum_value > maximum_value:
                raise ValueError(f"{local}.{minimum_name} exceeds {maximum_name}")
    return result


def _pose(scan: ET.Element, context: _InventoryContext) -> dict:
    pose = _single(scan, "pose", "Structure", context)
    if pose is None:
        return {
            "present": False,
            "effectiveTransform": "identity",
            "effectiveSource": "absent_pose_structure",
        }
    rotation = _single(pose, "rotation", "Structure", context)
    translation = _single(pose, "translation", "Structure", context)

    def components(
        structure: ET.Element | None,
        names: tuple[str, ...],
        label: str,
    ) -> dict:
        if structure is None:
            return {
                "present": False,
                "effectiveValue": (
                    {"w": 1, "x": 0, "y": 0, "z": 0}
                    if label == "rotation"
                    else {"x": 0, "y": 0, "z": 0}
                ),
                "effectiveSource": f"absent_pose_{label}_structure",
            }
        return {
            "present": True,
            **_node_identity(structure),
            "type": "Structure",
            "components": {
                name: _optional_scalar(
                    structure,
                    name,
                    {"Float"},
                    context,
                    allow_empty_value=True,
                )
                for name in names
            },
            "publisherLabel": label,
        }

    return {
        "present": True,
        **_node_identity(pose),
        "type": "Structure",
        "rotation": components(rotation, ("w", "x", "y", "z"), "rotation"),
        "translation": components(translation, ("x", "y", "z"), "translation"),
        "provenance": "publisher_declared_xml_metadata",
        "validatedAgainstCompressedVectorRecords": False,
        "transformSemantics": {
            "direction": "scan_local_to_file_frame",
            "formula": "p_file = R(q) * p_scan_local + t",
            "quaternionComponentOrder": ["w", "x", "y", "z"],
            "translationUnit": "metre",
        },
    }


def _original_guids(scan: ET.Element, context: _InventoryContext) -> dict:
    vector = _single(scan, "originalGuids", "Vector", context)
    if vector is None:
        return {"present": False}
    values = []
    for index, child in enumerate(vector):
        if child.tag != f"{{{E57_NAMESPACE}}}vectorChild":
            continue
        context.mark(child)
        values.append(
            {
                "index": index,
                **_scalar(child, {"String"}, f"originalGuids[{index}]"),
            }
        )
    return {
        "present": True,
        **_node_identity(vector),
        "type": "Vector",
        "values": values,
    }


def _compressed_vector(
    parent: ET.Element,
    local: str,
    context: _InventoryContext,
    *,
    required: bool,
    record_kind: str,
) -> dict | None:
    vector = _single(parent, local, "CompressedVector", context, required=required)
    if vector is None:
        return None
    record_count = vector.attrib.get("recordCount")
    file_offset = vector.attrib.get("fileOffset")
    if record_count is None or file_offset is None:
        raise ValueError(f"{local} lacks recordCount or fileOffset")
    record_count_value = _parse_uint64(record_count, f"{local}.recordCount")
    file_offset_value = _parse_uint64(file_offset, f"{local}.fileOffset")
    prototype = _single(vector, "prototype", "Structure", context, required=True)
    assert prototype is not None
    return {
        **_node_identity(vector),
        "type": "CompressedVector",
        "publisherDeclaredRecordCount": {
            "lexical": record_count,
            "value": record_count_value,
            "validatedAgainstDecodedRecords": False,
        },
        "publisherDeclaredFileOffset": {
            "lexical": file_offset,
            "value": file_offset_value,
            "validatedByRecordDecode": False,
        },
        "prototype": _prototype_node(
            prototype,
            context,
            f"{local}.prototype",
            record_kind=record_kind,
        ),
        "recordsDecoded": 0,
    }


def _point_grouping(
    scan: ET.Element,
    context: _InventoryContext,
    point_prototype: dict,
    namespaces: dict[str, str],
) -> dict:
    schemes = _single(scan, "pointGroupingSchemes", "Structure", context)
    if schemes is None:
        return {"present": False}
    grouping = _single(schemes, "groupingByLine", "Structure", context)
    if grouping is None:
        return {
            "present": True,
            **_node_identity(schemes),
            "type": "Structure",
            "groupingByLine": {"present": False},
        }
    id_element = _optional_scalar(
        grouping,
        "idElementName",
        {"String"},
        context,
        required=True,
    )
    groups = _compressed_vector(
        grouping,
        "groups",
        context,
        required=True,
        record_kind="line_group",
    )
    assert groups is not None
    group_fields = groups["prototype"].get("children", [])
    core_group_fields = {
        field["localName"]: field
        for field in group_fields
        if field["namespaceUri"] == E57_NAMESPACE
        and field["localName"] in _STANDARD_LINE_GROUP_FIELDS
    }
    if set(core_group_fields) != _STANDARD_LINE_GROUP_FIELDS or any(
        field["type"] != "Integer" for field in core_group_fields.values()
    ):
        raise ValueError(
            "Hovi groupingByLine prototype lacks its three direct Integer core fields"
        )
    id_value = id_element["value"]
    if ":" in id_value:
        prefix, local_name = id_value.split(":", 1)
        namespace = namespaces.get(prefix)
        if namespace is None:
            raise ValueError("groupingByLine.idElementName uses an unknown prefix")
    else:
        local_name = id_value
        namespace = E57_NAMESPACE
    point_fields = point_prototype.get("children", [])
    matches = [
        field
        for field in point_fields
        if field["namespaceUri"] == namespace and field["localName"] == local_name
    ]
    if len(matches) != 1 or matches[0]["type"] != "Integer":
        raise ValueError(
            "groupingByLine.idElementName does not resolve to one direct Integer point field"
        )
    resolved = matches[0]
    return {
        "present": True,
        **_node_identity(schemes),
        "type": "Structure",
        "groupingByLine": {
            "present": True,
            **_node_identity(grouping),
            "type": "Structure",
            "idElementName": id_element,
            "idElementResolution": {
                "resolved": True,
                "namespaceUri": namespace,
                "localName": local_name,
                "pointPrototypeIndex": resolved["index"],
                "pointPrototypeType": resolved["type"],
                "publisherDeclared": True,
                "validatedAgainstDecodedGroups": False,
            },
            "groups": groups,
            "groupRecordsDecoded": 0,
        },
    }


def _field_declarations(prototype: dict, names: tuple[str, ...]) -> dict:
    fields = prototype.get("children", [])
    result = {}
    for name in names:
        matches = [
            field
            for field in fields
            if field["namespaceUri"] == E57_NAMESPACE
            and field["localName"] == name
        ]
        result[name] = (
            {"present": False}
            if not matches
            else {
                "present": True,
                "prototypeIndex": matches[0]["index"],
                "publisherDeclared": True,
            }
        )
    return result


def _scan(
    scan: ET.Element,
    index: int,
    context: _InventoryContext,
    namespaces: dict[str, str],
) -> dict:
    _require_type(scan, "Structure", f"data3D[{index}]")
    points = _compressed_vector(
        scan,
        "points",
        context,
        required=True,
        record_kind="point",
    )
    assert points is not None
    cartesian_bounds = _component_structure(
        scan,
        "cartesianBounds",
        ("xMinimum", "xMaximum", "yMinimum", "yMaximum", "zMinimum", "zMaximum"),
        {"Float"},
        context,
        coordinate_frame="scan_local_pre_pose",
        ordered_pairs=(("xMinimum", "xMaximum"), ("yMinimum", "yMaximum"), ("zMinimum", "zMaximum")),
    )
    spherical_bounds = _component_structure(
        scan,
        "sphericalBounds",
        (
            "rangeMinimum",
            "rangeMaximum",
            "elevationMinimum",
            "elevationMaximum",
            "azimuthStart",
            "azimuthEnd",
        ),
        {"Float"},
        context,
        coordinate_frame="scan_local_pre_pose",
        ordered_pairs=(("rangeMinimum", "rangeMaximum"), ("elevationMinimum", "elevationMaximum")),
    )
    index_bounds = _component_structure(
        scan,
        "indexBounds",
        (
            "rowMinimum",
            "rowMaximum",
            "columnMinimum",
            "columnMaximum",
            "returnMinimum",
            "returnMaximum",
        ),
        {"Integer"},
        context,
        coordinate_frame="scan_record_index_domain",
        ordered_pairs=(("rowMinimum", "rowMaximum"), ("columnMinimum", "columnMaximum"), ("returnMinimum", "returnMaximum")),
    )
    return {
        "index": index,
        **_node_identity(scan),
        "type": "Structure",
        "guid": _optional_scalar(scan, "guid", {"String"}, context),
        "name": _optional_scalar(scan, "name", {"String"}, context),
        "originalGuids": _original_guids(scan, context),
        "pose": _pose(scan, context),
        "points": {**points, "pointsRead": 0},
        "bounds": {
            "cartesian": cartesian_bounds,
            "spherical": spherical_bounds,
            "index": index_bounds,
        },
        "invalidStateDeclarations": _field_declarations(
            points["prototype"],
            _INVALID_STATE_FIELDS,
        ),
        "indexAndReturnDeclarations": _field_declarations(
            points["prototype"],
            _INDEX_FIELDS,
        ),
        "pointGroupingSchemes": _point_grouping(
            scan,
            context,
            points["prototype"],
            namespaces,
        ),
        "publisherMetadataValidationBoundary": {
            "counts": "publisher_declared_not_record_validated",
            "bounds": "publisher_declared_scan_local_not_record_validated",
            "pose": "publisher_declared_not_point_validated",
        },
    }


def _generic_node(element: ET.Element) -> dict:
    return {
        **_node_identity(element),
        "text": element.text,
        "children": [
            {"node": _generic_node(child), "tail": child.tail}
            for child in element
            if isinstance(child.tag, str)
        ],
    }


def _unknown_frontier(
    element: ET.Element,
    context: _InventoryContext,
    path: str,
) -> list[dict]:
    unknown = []
    counts: dict[tuple[str | None, str], int] = {}
    for child in element:
        if not isinstance(child.tag, str):
            continue
        namespace, local = _qname(child.tag)
        key = (namespace, local)
        ordinal = counts.get(key, 0)
        counts[key] = ordinal + 1
        child_path = f"{path}/{{{namespace or ''}}}{local}[{ordinal}]"
        if id(child) not in context.used:
            unknown.append(
                {
                    "path": child_path,
                    "classification": (
                        "extension_namespace"
                        if namespace != E57_NAMESPACE
                        else "unmodeled_standard_namespace"
                    ),
                    "node": _generic_node(child),
                }
            )
        else:
            unknown.extend(_unknown_frontier(child, context, child_path))
    return unknown


def _parse_document(xml_bytes: bytes) -> tuple[ET.Element, list[dict]]:
    if len(xml_bytes) > MAX_XML_BYTES:
        raise ValueError(f"E57 XML exceeds the {MAX_XML_BYTES}-byte hard cap")
    if _FORBIDDEN_DECLARATION.search(xml_bytes):
        raise ValueError("E57 XML contains a forbidden DTD/entity/SYSTEM/PUBLIC declaration")
    namespaces: list[dict] = []
    seen_namespaces: set[tuple[str, str]] = set()
    depth = 0
    elements = 0
    iterator = ET.iterparse(
        io.BytesIO(xml_bytes),
        events=("start-ns", "start", "end"),
    )
    for event, value in iterator:
        if event == "start-ns":
            prefix, uri = value
            pair = (prefix or "", uri)
            if pair not in seen_namespaces:
                seen_namespaces.add(pair)
                namespaces.append({"prefix": prefix or "", "uri": uri})
        elif event == "start":
            depth += 1
            elements += 1
            if depth > MAX_XML_DEPTH or elements > MAX_XML_ELEMENTS:
                raise ValueError("E57 XML exceeds parser depth or element-count limits")
        else:
            depth -= 1
    root = iterator.root
    if root is None or depth != 0:
        raise ValueError("E57 XML did not produce one complete root")
    return root, namespaces


def inventory_xml(xml_bytes: bytes) -> dict:
    """Parse publisher XML without accessing any compressed-vector records."""
    root, namespaces = _parse_document(xml_bytes)
    if root.tag != f"{{{E57_NAMESPACE}}}e57Root":
        raise ValueError("E57 XML root namespace or local name is unsupported")
    context = _InventoryContext()
    context.mark(root)
    _require_type(root, "Structure", "e57Root")
    data3d = _single(root, "data3D", "Vector", context, required=True)
    assert data3d is not None
    namespace_map = {entry["prefix"]: entry["uri"] for entry in namespaces}
    scans = []
    for child in data3d:
        if child.tag != f"{{{E57_NAMESPACE}}}vectorChild":
            continue
        context.mark(child)
        scans.append(_scan(child, len(scans), context, namespace_map))
    coordinate_metadata = _optional_scalar(
        root,
        "coordinateMetadata",
        {"String"},
        context,
    )
    root_inventory = {
        **_node_identity(root),
        "type": "Structure",
        "formatName": _optional_scalar(
            root,
            "formatName",
            {"String"},
            context,
            required=True,
        ),
        "guid": _optional_scalar(
            root,
            "guid",
            {"String"},
            context,
            required=True,
        ),
        "versionMajor": _optional_scalar(
            root,
            "versionMajor",
            {"Integer"},
            context,
            required=True,
        ),
        "versionMinor": _optional_scalar(
            root,
            "versionMinor",
            {"Integer"},
            context,
            required=True,
        ),
        "coordinateMetadata": coordinate_metadata,
        "coordinateReferenceInterpretation": {
            "publisherMetadataPresent": coordinate_metadata["present"],
            "publisherMetadataEmpty": (
                coordinate_metadata["present"]
                and coordinate_metadata["value"] == ""
            ),
            "crsInferred": False,
            "interpretation": (
                "no_coordinate_reference_system_declared"
                if coordinate_metadata["present"]
                and coordinate_metadata["value"] == ""
                else "publisher_coordinate_metadata_preserved_without_inference"
            ),
        },
        "e57LibraryVersion": _optional_scalar(
            root,
            "e57LibraryVersion",
            {"String"},
            context,
        ),
        "data3D": {
            **_node_identity(data3d),
            "type": "Vector",
            "scanCount": len(scans),
            "scans": scans,
        },
    }
    return {
        "namespaceDeclarations": namespaces,
        "root": root_inventory,
        "unknownPublisherElements": _unknown_frontier(root, context, "/e57Root"),
        "publisherDeclarationsPreserved": True,
        "pointsRead": 0,
        "compressedVectorRecordsDecoded": 0,
    }
