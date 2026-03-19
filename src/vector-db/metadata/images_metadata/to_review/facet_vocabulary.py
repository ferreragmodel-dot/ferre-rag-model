#!/usr/bin/env python3
"""Ferré compact facet vocabulary v1.

This module freezes the controlled vocabulary used by the retrieval layer.
It is intentionally smaller and more archive-specific than a general fashion
ontology. The JSON emitted by this module is meant to be consumed later by:
- grounded -> compact derivation scripts
- multimodal generation prompts
- response_schema / enum definitions for structured output
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

SCHEMA_VERSION = "ferre_facet_v1"
DEFAULT_OUTPUT = Path("metadata/config/ferre_facet_vocabulary_v1.json")

VOCABULARY: Dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "description": "Ferré-specific compact vocabulary for retrieval facets.",
    "enum_fields": [
        "garments",
        "colors",
        "material_families",
        "patterns",
        "silhouette_tags",
        "length_tags",
        "neckline_tags",
        "sleeve_tags",
        "closure_tags",
        "embellishment_tags",
        "style_tags",
    ],
    "free_text_fields": [
        "description_raw",
        "materials_raw",
        "remark_raw",
        "description_short",
        "remark_short",
    ],
    "forbidden_in_facets": {
        "supplier_names": True,
        "article_codes": True,
        "color_codes": True,
        "percentages": True,
        "comma_joined_arrays": True,
        "prose_inside_arrays": True,
    },
    "facet_fields": {
        "garments": [
            "jacket", "coat", "dress", "evening_dress", "skirt", "shirt",
            "t_shirt", "tunic", "blouse", "top", "bustier", "bodice",
            "cape", "bolero", "shawl", "stole", "duster", "pants",
            "overskirt", "underskirt", "belt", "sash", "hat",
        ],
        "colors": [
            "black", "white", "gray", "dove_gray", "beige", "ecru", "ivory",
            "red", "coral", "pink", "lilac", "blue", "navy", "brown",
            "green", "yellow", "gold", "silver", "straw", "bois_de_rose",
        ],
        "material_families": [
            "silk", "wool", "cotton", "nylon", "velvet", "lace", "tulle",
            "organza", "georgette", "satin", "faille", "gazar", "cady",
            "chiffon", "jersey", "poplin", "taffeta", "mousseline", "crepe",
            "grosgrain", "raffia", "leather", "python", "crocodile", "fur",
            "fox_fur", "alpaca", "cashmere", "mohair", "crystal_beads",
            "jet_beads",
        ],
        "patterns": [
            "solid", "striped", "pinstripe", "chalkstripe", "plaid",
            "scotch_plaid", "polka_dot", "zigzag", "floral", "paisley",
            "geometric", "printed",
        ],
        "silhouette_tags": [
            "slim", "straight", "fitted", "boxy", "structured", "tubular",
            "tube", "circular", "a_line", "flared", "trumpet", "bell",
            "kimono", "voluminous", "fluid", "draped", "amphora",
        ],
        "length_tags": [
            "hip_length", "upper_hip_length", "mid_thigh_length", "knee_length",
            "ankle_length", "floor_length", "above_knee", "seven_eighth_length",
        ],
        "neckline_tags": [
            "round_neck", "crew_neck", "scoop_neck", "high_neck", "halter",
            "strapless", "peter_pan_collar", "mannish_collar", "shawl_collar",
            "hood",
        ],
        "sleeve_tags": [
            "sleeveless", "short_sleeve", "long_sleeve", "wide_sleeve",
            "kimono_sleeve", "puff_sleeve", "gigot_sleeve", "tubular_sleeve",
            "elbow_length", "three_quarter_sleeve",
        ],
        "closure_tags": [
            "single_breasted", "one_button", "back_zip", "side_zip",
            "button_front", "crossover", "no_fastening", "hook_and_eye",
        ],
        "embellishment_tags": [
            "embroidery", "cornely_embroidery", "beading", "jet_beading",
            "applique", "lace_applique", "topstitching", "pleating",
            "sunburst_pleating", "intarsia", "passementerie",
            "ribbon_application", "openwork", "ajour", "tassels", "ruffles",
        ],
        "style_tags": [
            "masculine_feminine", "power_suit", "dandy", "oriental",
            "hollywood_glamour", "romantic", "lingerie_inspired",
            "torero_inspired", "maghreb_inspired", "sculptural",
            "origami_like", "aristocratic", "daywear", "eveningwear",
        ],
    },
    "synonym_maps": {
        "garments": {
            "abito": "dress",
            "abito da sera": "evening_dress",
            "evening gown": "evening_dress",
            "gonna": "skirt",
            "giacca": "jacket",
            "camicia": "shirt",
            "t-shirt": "t_shirt",
            "t shirt": "t_shirt",
            "bolero jacket": "bolero",
            "camicia e gonna": ["shirt", "skirt"],
            "jacket/skirt/t-shirt outfit": ["jacket", "skirt", "t_shirt"],
            "tunic and skirt outfit": ["tunic", "skirt"],
            "duster and pants outfit": ["duster", "pants"],
            "fox coat and fox dress": ["coat", "dress"],
        },
        "colors": {
            "dark blue": "navy",
            "navy blue": "navy",
            "dove gray": "dove_gray",
            "natural hue of white": "white",
            "straw color": "straw",
            "bois de rose": "bois_de_rose",
        },
        "material_families": {
            "crêpe": "crepe",
            "crepe de chine": "crepe",
            "gros grain": "grosgrain",
            "cashmire": "cashmere",
            "cachemire": "cashmere",
            "cachmire": "cashmere",
            "wool/silk satin": ["wool", "silk", "satin"],
            "crystal jet": ["jet_beads"],
            "silver fox fur": ["fox_fur", "fur"],
            "genuine crocodile leather": ["crocodile", "leather"],
            "python leather": ["python", "leather"],
        },
        "patterns": {
            "scotch plaid": "scotch_plaid",
            "polkadot": "polka_dot",
            "dot": "polka_dot",
            "bouquet design": "floral",
            "flower decors": "floral",
            "white horizontal 8 cm parallel striping": "striped",
            "vichy": "geometric",
        },
        "silhouette_tags": {
            "a line": "a_line",
            "a-line": "a_line",
            "tube style": "tube",
            "tubular style": "tubular",
        },
        "embellishment_tags": {
            "applicazioni": "applique",
            "appliqué": "applique",
            "lace in appliqué form": ["lace_applique", "applique"],
            "all over glass tube beading": ["beading"],
            "cornely (machine) embroidery": ["cornely_embroidery", "embroidery"],
            "openwork (ajour embroidery) edging": ["openwork", "ajour"],
        },
        "style_tags": {
            "masculine/feminine": "masculine_feminine",
            "masculine and feminine": "masculine_feminine",
            "hollywood and cinecittà glamour": "hollywood_glamour",
            "romanticism": "romantic",
            "oriental flair": "oriental",
            "lingerie": "lingerie_inspired",
            "power suit": "power_suit",
            "dandyish": "dandy",
            "maghrebian burnus": "maghreb_inspired",
            "origami technique": "origami_like",
        },
    },
    "normalization_rules": {
        "object_labels_are_decomposed": True,
        "null_arrays_are_omitted": True,
        "array_values_are_unique": True,
        "array_values_are_lowercase": True,
    },
}


def build_vertex_response_schema() -> Dict[str, Any]:
    """Return a schema skeleton ready to be adapted for Vertex structured output."""
    enums = VOCABULARY["facet_fields"]
    array_field = lambda name: {
        "type": "array",
        "items": {"type": "string", "enum": enums[name]},
    }
    return {
        "type": "object",
        "properties": {
            "garments": array_field("garments"),
            "colors": array_field("colors"),
            "material_families": array_field("material_families"),
            "patterns": array_field("patterns"),
            "silhouette_tags": array_field("silhouette_tags"),
            "length_tags": array_field("length_tags"),
            "neckline_tags": array_field("neckline_tags"),
            "sleeve_tags": array_field("sleeve_tags"),
            "closure_tags": array_field("closure_tags"),
            "embellishment_tags": array_field("embellishment_tags"),
            "style_tags": array_field("style_tags"),
            "description_short": {"type": "string"},
            "remark_short": {"type": "string"},
            "facet_confidence": {"type": "number"},
        },
    }


def write_vocabulary(output_path: Path = DEFAULT_OUTPUT) -> Path:
    payload = dict(VOCABULARY)
    payload["vertex_response_schema_template"] = build_vertex_response_schema()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    path = write_vocabulary()
    print(f"Wrote {path}")
