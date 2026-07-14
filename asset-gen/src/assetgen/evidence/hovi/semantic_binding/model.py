"""Fixed target ontology and publisher-to-target mappings for Hovi evidence."""

from __future__ import annotations

TARGET_CLASSES = (
    "mineral_soil",
    "litter",
    "moss_or_peat",
    "roots",
    "deadwood",
    "living_vegetation",
    "clasts",
    "water",
    "unknown",
)

# Only publisher-assigned fractional-cover fields are mapped. A missing source
# category is not evidence of zero cover.
PUBLISHER_BINDINGS = {
    "vasc": {
        "target_class": "living_vegetation",
        "source_definition": "vascular plants",
        "qualification": "direct_publisher_fraction",
    },
    "nonvasc": {
        "target_class": "moss_or_peat",
        "source_definition": "nonvascular plants, explicitly mosses",
        "qualification": "moss_only_no_peat_inference",
    },
    "intactlitt": {
        "target_class": "litter",
        "source_definition": "intact plant litter",
        "qualification": "direct_publisher_fraction",
    },
    "decomplitt": {
        "target_class": "litter",
        "source_definition": "decomposed plant litter",
        "qualification": "direct_publisher_fraction",
    },
    "lichen": {
        "target_class": "unknown",
        "source_definition": "lichen",
        "qualification": "preserved_unmapped_living_biota",
    },
}

UNMEASURED_TARGET_CLASSES = (
    "mineral_soil",
    "roots",
    "deadwood",
    "clasts",
    "water",
)

CLASS_COLORS = {
    "living_vegetation": (75, 135, 69),
    "moss_or_peat": (126, 160, 88),
    "litter": (145, 101, 64),
    "unknown": (112, 117, 113),
}
