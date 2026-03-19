#!/usr/bin/env python3
"""Derive Ferré compact retrieval facets from the normalized grounded archive.

Phase 03 in the compact-facets pipeline:
- read metadata/normalized_grounded/grounded_raw.jsonl
- read metadata/config/ferre_facet_vocabulary_v1.json
- derive compact retrieval facets with bilingual heuristic rules
- validate all facet arrays against the frozen vocabulary
- write metadata/derived_grounded_facets/grounded_compact.jsonl
- write metadata/derived_grounded_facets/grounded_compact_report.json

This phase is intentionally deterministic and lightweight: no LLM is used here.
The goal is to squeeze the grounded archive into a compact, filterable schema that
later phases can merge into Chroma metadata and use as exemplars for image-side generation.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

INPUT_JSONL = Path("metadata/normalized_grounded/grounded_raw.jsonl")
VOCAB_PATH = Path("metadata/config/ferre_facet_vocabulary_v1.json")
OUTPUT_DIR = Path("metadata/derived_grounded_facets")
OUTPUT_JSONL = OUTPUT_DIR / "grounded_compact.jsonl"
OUTPUT_REPORT = OUTPUT_DIR / "grounded_compact_report.json"
SCHEMA_VERSION = "ferre_facets_v1"

FACET_FIELDS = [
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
]

NON_FILTER_TEXT_FIELDS = ["description_raw", "materials_raw", "remark_raw"]

# Bilingual extraction rules. The vocabulary file already contains canonical values and a first synonym map.
# Here we add the archive-specific phrase patterns actually needed to derive facets from the Ferré corpus.
EXTRACTION_PATTERNS: Dict[str, Dict[str, Sequence[str]]] = {
    "garments": {
        "evening_dress": [
            r"\bevening dress\b",
            r"\bevening gown\b",
            r"\babito da sera\b",
        ],
        "dress": [
            r"\bdress\b",
            r"\bdrees\b",
            r"\babito\b",
            r"\bredingote\b",
        ],
        "jacket": [
            r"\bjacket\b",
            r"\bgiacca\b",
            r"\bgiacchino\b",
        ],
        "coat": [
            r"\bcoat\b",
            r"\bcappotto\b",
            r"\bsoprabito\b",
            r"\bcabane\b",
            r"\bcaban\b",
            r"\bredingote\b",
        ],
        "skirt": [
            r"\bskirt\b",
            r"\bgonna\b",
        ],
        "shirt": [
            r"(?<!t-)\bshirt\b",
            r"\bcamicia\b",
        ],
        "t_shirt": [
            r"\bt[ -]?shirt\b",
        ],
        "tunic": [
            r"\btunic\b",
            r"\btunica\b",
        ],
        "blouse": [r"\bblouse\b", r"\bblusa\b"],
        "top": [r"\btop\b"],
        "bustier": [r"\bbustier\b"],
        "bodice": [r"\bbodice\b"],
        "cape": [r"\bcape\b", r"\bcappa\b"],
        "bolero": [r"\bbolero\b"],
        "shawl": [r"\bshawl\b"],
        "stole": [r"\bstole\b", r"\bstola\b"],
        "duster": [r"\bduster\b"],
        "pants": [r"\bpants\b", r"\btrousers\b", r"\bpantaloni\b"],
        "overskirt": [r"\boverskirt\b"],
        "underskirt": [r"\bunderskirt\b"],
        "belt": [r"\bbelt\b", r"\bcintura\b", r"\bcinturetta\b"],
        "sash": [r"\bsash\b", r"\bfascia\b"],
        "hat": [r"\bhat\b", r"\bcappello\b"],
    },
    "colors": {
        "black": [r"\bblack\b", r"\bnero\b", r"\bnera\b"],
        "white": [r"\bwhite\b", r"\bbianco\b", r"\bbianca\b"],
        "gray": [r"\bgray\b", r"\bgrey\b", r"\bgrigio\b", r"\bgrigia\b"],
        "dove_gray": [r"\bdove gray\b", r"\bgrigio tortora\b"],
        "beige": [r"\bbeige\b"],
        "ecru": [r"\becru\b", r"\b\u00e9cru\b"],
        "ivory": [r"\bivory\b", r"\bavorio\b"],
        "red": [r"\bred\b", r"\brosso\b", r"\brossa\b"],
        "coral": [r"\bcoral\b", r"\bcorallo\b"],
        "pink": [r"\bpink\b", r"\brosa\b"],
        "lilac": [r"\blilac\b", r"\blilla\b"],
        "blue": [r"\bblue\b", r"\bblu\b"],
        "navy": [r"\bnavy\b", r"\bdark blue\b", r"\bblu scuro\b"],
        "brown": [r"\bbrown\b", r"\bmarrone\b"],
        "green": [r"\bgreen\b", r"\bverde\b"],
        "yellow": [r"\byellow\b", r"\bgiallo\b", r"\bgialla\b"],
        "gold": [r"\bgold\b", r"\boro\b", r"\bdorato\b", r"\bdorata\b"],
        "silver": [r"\bsilver\b", r"\bargento\b", r"\bargentato\b"],
        "straw": [r"\bstraw\b", r"\bpaglia\b"],
        "bois_de_rose": [r"\bbois de rose\b"],
    },
    "material_families": {
        "silk": [r"\bsilk\b", r"\bseta\b"],
        "wool": [r"\bwool\b", r"\blana\b"],
        "cotton": [r"\bcotton\b", r"\bcotone\b"],
        "nylon": [r"\bnylon\b"],
        "velvet": [r"\bvelvet\b", r"\bvelluto\b"],
        "lace": [r"\blace\b", r"\bpizzo\b"],
        "tulle": [r"\btulle\b"],
        "organza": [r"\borganza\b"],
        "georgette": [r"\bgeorgette\b"],
        "satin": [r"\bsatin\b", r"\bsatinato\b"],
        "faille": [r"\bfaille\b"],
        "gazar": [r"\bgazar\b"],
        "cady": [r"\bcady\b"],
        "chiffon": [r"\bchiffon\b"],
        "jersey": [r"\bjersey\b"],
        "poplin": [r"\bpoplin\b", r"\bpopeline\b"],
        "taffeta": [r"\btaffeta\b", r"\btaffet\u00e0\b"],
        "mousseline": [r"\bmousseline\b"],
        "crepe": [r"\bcrepe\b", r"\bcr\u00eape\b"],
        "grosgrain": [r"\bgros ?grain\b"],
        "raffia": [r"\braffia\b", r"\brafia\b"],
        "leather": [r"\bleather\b", r"\bpelle\b"],
        "python": [r"\bpython\b", r"\bpitone\b"],
        "crocodile": [r"\bcrocodile\b", r"\bcoccodrillo\b"],
        "fur": [r"\bfur\b", r"\bpelliccia\b"],
        "fox_fur": [r"\bfox fur\b", r"\bvolpe\b"],
        "alpaca": [r"\balpaca\b"],
        "cashmere": [r"\bcashmere\b", r"\bcachemire\b", r"\bcashmire\b", r"\bcachmire\b"],
        "mohair": [r"\bmohair\b"],
        "crystal_beads": [r"\bcrystal beads?\b", r"\bcrystals?\b"],
        "jet_beads": [r"\bjet beading\b", r"\bjet beads?\b", r"\bcrystal jet\b"],
    },
    "patterns": {
        "solid": [r"\bsolid\b", r"\bplain\b", r"\bunito\b"],
        "striped": [r"\bstriped\b", r"\bstriping\b", r"\bstripes\b", r"\brighe\b", r"\brigato\b"],
        "pinstripe": [r"\bpinstripe\b"],
        "chalkstripe": [r"\bchalkstripe\b"],
        "plaid": [r"\bplaid\b", r"\btartan\b"],
        "scotch_plaid": [r"\bscotch plaid\b"],
        "polka_dot": [r"\bpolka ?dot\b", r"\bpolkadot\b", r"\bpois\b"],
        "zigzag": [r"\bzigzag\b", r"\bzig-zag\b"],
        "floral": [r"\bfloral\b", r"\bflower\w*\b", r"\bbouquet\b", r"\bfiori\b"],
        "paisley": [r"\bpaisley\b"],
        "geometric": [r"\bgeometric\b", r"\bvichy\b", r"\bchecker(?:ed)?\b"],
        "printed": [r"\bprinted\b", r"\bprint\b", r"\bstampat\w*\b"],
    },
    "silhouette_tags": {
        "slim": [r"\bslim\b", r"\basciutt\w*\b"],
        "straight": [r"\bstraight\b", r"\brett\w*\b"],
        "fitted": [r"\bfitted\b", r"\baderent\w*\b"],
        "boxy": [r"\bboxy\b"],
        "structured": [r"\bstructured\b", r"\bstructure\b", r"\bstrutturat\w*\b"],
        "tubular": [r"\btubular\b"],
        "tube": [r"\btube\b"],
        "circular": [r"\bcircular\b", r"\bcircolare\b"],
        "a_line": [r"\ba-?line\b"],
        "flared": [r"\bflared\b", r"\bsvasat\w*\b"],
        "trumpet": [r"\btrumpet\b"],
        "bell": [r"\bbell\b"],
        "kimono": [r"\bkimono\b"],
        "voluminous": [r"\bvoluminous\b", r"\bcapacious\b", r"\bampi\w*\b"],
        "fluid": [r"\bfluid\b", r"\bfluido\b", r"\bfluida\b"],
        "draped": [r"\bdraped\b", r"\bdrappeggiat\w*\b"],
        "amphora": [r"\bamphora\b", r"\banfora\b"],
    },
    "length_tags": {
        "hip_length": [r"\bhip length\b", r"\blunghezza ai fianchi\b"],
        "upper_hip_length": [r"\bupper hip\b", r"\bsopra i fianchi\b"],
        "mid_thigh_length": [r"\bmid-?thigh\b"],
        "knee_length": [r"\bknee length\b", r"\blunghezza al ginocchio\b", r"\bal ginocchio\b"],
        "ankle_length": [r"\bankle length\b", r"\balla caviglia\b"],
        "floor_length": [r"\bfloor length\b", r"\ba terra\b"],
        "above_knee": [r"\babove knee\b", r"\bsopra il ginocchio\b"],
        "seven_eighth_length": [r"\b7/8\b", r"\bsette ottavi\b"],
    },
    "neckline_tags": {
        "round_neck": [r"\bround neck\b", r"\bwide round neck\b", r"\bgirocollo\b"],
        "crew_neck": [r"\bcrew neck\b"],
        "scoop_neck": [r"\bscoop neck\b"],
        "high_neck": [r"\bhigh neck\b", r"\bcollo alto\b"],
        "halter": [r"\bhalter\b"],
        "strapless": [r"\bstrapless\b"],
        "peter_pan_collar": [r"\bpeter pan collar\b"],
        "mannish_collar": [r"\bmannish collar\b", r"\brevers a uomo\b"],
        "shawl_collar": [r"\bshawl collar\b"],
        "hood": [r"\bhood\b", r"\bcappuccio\b"],
    },
    "sleeve_tags": {
        "sleeveless": [r"\bsleeveless\b", r"\bsenza maniche\b"],
        "short_sleeve": [r"\bshort sleeves?\b", r"\bmaniche corte\b"],
        "long_sleeve": [r"\blong sleeves?\b", r"\bmaniche lunghe\b"],
        "wide_sleeve": [r"\bwide sleeves?\b", r"\bcapacious sleeves?\b", r"\bmaniche ampie\b"],
        "kimono_sleeve": [r"\bkimono sleeves?\b", r"\bmaniche a kimono\b"],
        "puff_sleeve": [r"\bpuff sleeves?\b"],
        "gigot_sleeve": [r"\bgigot sleeves?\b"],
        "tubular_sleeve": [r"\btubular sleeves?\b"],
        "elbow_length": [r"\belbow length\b", r"\bal gomito\b"],
        "three_quarter_sleeve": [r"\b3/4 sleeves?\b", r"\bthree quarter sleeves?\b", r"\btre quarti\b"],
    },
    "closure_tags": {
        "single_breasted": [r"\bsingle-breasted\b", r"\bmonopetto\b"],
        "one_button": [r"\bone button\b", r"\bun bottone\b"],
        "back_zip": [r"\bback zip\b", r"\bzip sul dorso\b", r"\bzip on the back\b"],
        "side_zip": [r"\bside zip\b", r"\bzip al fianco\b", r"\bzip laterale\b"],
        "button_front": [r"\bbutton front\b", r"\bbuttoned front\b", r"\bbottoni frontali\b"],
        "crossover": [r"\bcrossover\b", r"\bincrociat\w*\b"],
        "no_fastening": [r"\bno fastening\b", r"\bsenza chiusura\b"],
        "hook_and_eye": [r"\bhook(?:-and-| and )eye\b", r"\bganci e occhielli\b"],
    },
    "embellishment_tags": {
        "embroidery": [r"\bembroidery\b", r"\bricam\w*\b"],
        "cornely_embroidery": [r"\bcornely\b"],
        "beading": [r"\bbeading\b", r"\bperlin\w*\b"],
        "jet_beading": [r"\bjet beading\b", r"\bperline jet\b"],
        "applique": [r"\bappliqu\w*\b", r"\bapplicazion\w*\b"],
        "lace_applique": [r"\blace appliqu\w*\b"],
        "topstitching": [r"\btopstitching\b", r"\bimpuntur\w*\b"],
        "pleating": [r"\bpleating\b", r"\bpliss\w*\b", r"\bpleat\w*\b"],
        "sunburst_pleating": [r"\bsunburst\b"],
        "intarsia": [r"\bintarsia\b"],
        "passementerie": [r"\bpassementerie\b"],
        "ribbon_application": [r"\bribbon application\b", r"\bnastr\w*\b"],
        "openwork": [r"\bopenwork\b"],
        "ajour": [r"\bajour\b"],
        "tassels": [r"\btassels?\b", r"\bnappine\b"],
        "ruffles": [r"\bruffles?\b", r"\brouches?\b", r"\bvolants?\b", r"\bbalze\b"],
    },
    "style_tags": {
        "masculine_feminine": [r"\bmasculine/feminine\b", r"\bmasculine and feminine\b", r"\bmasculine\b.*\bfeminine\b"],
        "power_suit": [r"\bpower suit\b"],
        "dandy": [r"\bdandy\w*\b"],
        "oriental": [r"\boriental\b"],
        "hollywood_glamour": [r"\bhollywood\b", r"\bcinecitt\u00e0 glamour\b", r"\bglamour\b"],
        "romantic": [r"\bromantic\w*\b"],
        "lingerie_inspired": [r"\blingerie\b"],
        "torero_inspired": [r"\btorero\b"],
        "maghreb_inspired": [r"\bmaghreb\w*\b", r"\bburnus\b"],
        "sculptural": [r"\bsculptur\w*\b"],
        "origami_like": [r"\borigami\b"],
        "aristocratic": [r"\baristocratic\b", r"\baristocratico\b", r"\baristocratica\b"],
        "daywear": [r"\bdaywear\b"],
        "eveningwear": [r"\beveningwear\b", r"\bevening\b", r"\bsera\b"],
    },
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")
    text = text.replace("\u2018", "'")
    text = text.replace("\u2019", "'")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def humanize(value: str) -> str:
    return value.replace("_", " ")


def join_english(values: Sequence[str]) -> str:
    values = [humanize(v) for v in values if v]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def trim_text(text: Optional[str], limit: int = 180) -> Optional[str]:
    if not text:
        return None
    cleaned = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    if len(cleaned) <= limit:
        return cleaned
    cut = cleaned[: limit - 1].rsplit(" ", 1)[0]
    return cut + "…"


def first_sentence(text: Optional[str], limit: int = 180) -> Optional[str]:
    if not text:
        return None
    cleaned = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    parts = re.split(r"(?<=[\.!?;])\s+", cleaned)
    candidate = parts[0].strip()
    return trim_text(candidate, limit=limit)


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_vocabulary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compile_patterns(vocab: Dict[str, Any]) -> Dict[str, Dict[str, List[re.Pattern[str]]]]:
    compiled: Dict[str, Dict[str, List[re.Pattern[str]]]] = {}
    synonym_maps = vocab.get("synonym_maps", {})
    for field, allowed_values in vocab["facet_fields"].items():
        field_patterns: Dict[str, List[str]] = {value: [] for value in allowed_values}

        # Canonical values as search hints, when they are meaningful literal phrases.
        for value in allowed_values:
            canonical_phrase = humanize(value)
            field_patterns[value].append(rf"\b{re.escape(canonical_phrase)}\b")

        # Synonym map from vocabulary JSON.
        for synonym, target in synonym_maps.get(field, {}).items():
            targets = target if isinstance(target, list) else [target]
            for canonical in targets:
                if canonical in field_patterns:
                    field_patterns[canonical].append(rf"\b{re.escape(normalize_text(synonym))}\b")

        # Archive-specific regex patterns.
        for canonical, patterns in EXTRACTION_PATTERNS.get(field, {}).items():
            if canonical in field_patterns:
                field_patterns[canonical].extend(patterns)

        compiled[field] = {
            canonical: [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]
            for canonical, patterns in field_patterns.items()
        }
    return compiled


def extract_from_text(text: str, field: str, compiled_patterns: Dict[str, Dict[str, List[re.Pattern[str]]]], allowed: Set[str]) -> List[str]:
    hits: List[str] = []
    for canonical, patterns in compiled_patterns[field].items():
        for pattern in patterns:
            if pattern.search(text):
                if canonical in allowed and canonical not in hits:
                    hits.append(canonical)
                break
    return hits


def dedupe_keep_order(values: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def remove_redundant_tags(facets: Dict[str, List[str]]) -> None:
    if "evening_dress" in facets["garments"] and "dress" in facets["garments"]:
        facets["garments"] = [v for v in facets["garments"] if v != "dress"]
    if "scotch_plaid" in facets["patterns"] and "plaid" in facets["patterns"]:
        facets["patterns"] = [v for v in facets["patterns"] if v != "plaid"]
    if "cornely_embroidery" in facets["embellishment_tags"] and "embroidery" not in facets["embellishment_tags"]:
        facets["embellishment_tags"].append("embroidery")
    if "lace_applique" in facets["embellishment_tags"] and "applique" not in facets["embellishment_tags"]:
        facets["embellishment_tags"].append("applique")
    if "fox_fur" in facets["material_families"] and "fur" not in facets["material_families"]:
        facets["material_families"].append("fur")
    if "python" in facets["material_families"] and "leather" not in facets["material_families"]:
        facets["material_families"].append("leather")
    if "crocodile" in facets["material_families"] and "leather" not in facets["material_families"]:
        facets["material_families"].append("leather")
    if "jet_beads" in facets["material_families"] and "crystal_beads" in facets["material_families"]:
        # keep both: they are different archive signals
        pass
    for field in FACET_FIELDS:
        facets[field] = dedupe_keep_order(facets[field])


def validate_facets(facets: Dict[str, List[str]], allowed_by_field: Dict[str, Set[str]]) -> Tuple[bool, Dict[str, List[str]]]:
    invalid: Dict[str, List[str]] = {}
    for field in FACET_FIELDS:
        bad = [value for value in facets[field] if value not in allowed_by_field[field]]
        if bad:
            invalid[field] = bad
        facets[field] = [value for value in facets[field] if value in allowed_by_field[field]]
    return (not invalid), invalid


def count_nonempty_groups(facets: Dict[str, List[str]]) -> int:
    return sum(1 for field in FACET_FIELDS if facets[field])


def compute_facet_confidence(record: Dict[str, Any], facets: Dict[str, List[str]]) -> float:
    norm = record.get("normalized", {})
    source_score = 0.0
    if norm.get("object"):
        source_score += 0.25
    if norm.get("description"):
        source_score += 0.30
    if norm.get("materials"):
        source_score += 0.20
    if norm.get("working_process"):
        source_score += 0.15
    if norm.get("remark"):
        source_score += 0.10

    extraction_score = min(0.35, 0.05 * count_nonempty_groups(facets))
    if facets["garments"]:
        extraction_score += 0.05

    confidence = min(1.0, 0.20 + source_score + extraction_score)

    # Penalize very generic object-only records, which will need image fill later.
    if record.get("raw_completeness") != "full" and count_nonempty_groups(facets) <= 1:
        confidence = min(confidence, 0.60)
    if not facets["garments"]:
        confidence = min(confidence, 0.55)
    return round(confidence, 3)


def build_description_short(facets: Dict[str, List[str]], description_raw: Optional[str]) -> Optional[str]:
    garments = facets["garments"][:3]
    colors = facets["colors"][:3]
    patterns = facets["patterns"][:1]
    materials = facets["material_families"][:4]

    if garments:
        pieces: List[str] = []
        if colors:
            color_text = "/".join(humanize(v) for v in colors)
            pieces.append(color_text)
        if patterns:
            pieces.append(humanize(patterns[0]))
        pieces.append(join_english(garments))
        sentence = " ".join(piece for piece in pieces if piece).strip()
        if materials:
            sentence += f" in {join_english(materials)}"
        sentence = sentence[0].upper() + sentence[1:] + "."
        return trim_text(sentence, limit=180)
    return first_sentence(description_raw, limit=180)


def build_remark_short(facets: Dict[str, List[str]], remark_raw: Optional[str], working_process: Optional[str]) -> Optional[str]:
    if remark_raw:
        return first_sentence(remark_raw, limit=180)
    style_tags = facets["style_tags"][:2]
    embellishments = facets["embellishment_tags"][:2]
    if style_tags or embellishments:
        parts = []
        if style_tags:
            parts.append(join_english(style_tags))
        if embellishments:
            parts.append(join_english(embellishments))
        sentence = " with ".join(parts)
        sentence = sentence[0].upper() + sentence[1:] + "."
        return trim_text(sentence, limit=180)
    return first_sentence(working_process, limit=180)


def derive_facets(record: Dict[str, Any], compiled_patterns: Dict[str, Dict[str, List[re.Pattern[str]]]], allowed_by_field: Dict[str, Set[str]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    norm = record.get("normalized", {})
    object_text = normalize_text(norm.get("object") or "")
    description_text = normalize_text(norm.get("description") or "")
    materials_text = normalize_text(norm.get("materials") or "")
    working_text = normalize_text(norm.get("working_process") or "")
    remark_text = normalize_text(norm.get("remark") or "")
    season_text = normalize_text(record.get("season") or "")

    text_by_field = {
        "garments": " ".join([object_text, description_text]),
        "colors": " ".join([description_text, materials_text]),
        "material_families": " ".join([materials_text, description_text]),
        "patterns": " ".join([description_text, materials_text, object_text]),
        "silhouette_tags": description_text,
        "length_tags": description_text,
        "neckline_tags": description_text,
        "sleeve_tags": description_text,
        "closure_tags": description_text,
        "embellishment_tags": " ".join([working_text, description_text, materials_text]),
        "style_tags": " ".join([remark_text, description_text, object_text, season_text]),
    }

    facets: Dict[str, List[str]] = {}
    derivation_trace: Dict[str, List[str]] = {}

    for field in FACET_FIELDS:
        extracted = extract_from_text(text_by_field[field], field, compiled_patterns, allowed_by_field[field])
        facets[field] = dedupe_keep_order(extracted)
        derivation_trace[field] = facets[field][:]

    # Archive-specific corrections.
    if "evening_dress" in facets["garments"] and "eveningwear" not in facets["style_tags"]:
        facets["style_tags"].append("eveningwear")
    if "dress" in facets["garments"] and "evening_dress" not in facets["garments"]:
        # Keep dress as-is; do not infer eveningwear.
        pass
    if "shirt" in facets["garments"] and "t_shirt" in facets["garments"]:
        shirt_evidence = re.search(r"(?<!t-)\bshirt\b|\bcamicia\b", " ".join([object_text, description_text]))
        if not shirt_evidence:
            facets["garments"] = [v for v in facets["garments"] if v != "shirt"]
    if not facets["garments"] and object_text in {"abito", "dress", "evening dress", "evening gown"}:
        # Safeguard for very short object-only records.
        facets["garments"] = ["evening_dress"] if "evening" in object_text or "sera" in object_text else ["dress"]
    if "redingote" in object_text and "dress" in facets["garments"] and "coat" not in facets["garments"]:
        facets["garments"].append("coat")
    if "hood" in description_text and "hood" not in facets["neckline_tags"]:
        facets["neckline_tags"].append("hood")
    if "cornely" in working_text and "cornely_embroidery" not in facets["embellishment_tags"]:
        facets["embellishment_tags"].append("cornely_embroidery")

    remove_redundant_tags(facets)
    valid, invalid = validate_facets(facets, allowed_by_field)

    description_raw = norm.get("description")
    materials_raw = norm.get("materials")
    remark_raw = norm.get("remark")

    compact = {
        "record_id": record["record_id"],
        "source_path": record["source_path"],
        "source_file": record["source_file"],
        "outfit_id": record.get("outfit_id"),
        "season": record.get("season"),
        "year": record.get("year"),
        "collection": record.get("collection"),
        "collection_line": record.get("collection_line"),
        "look": record.get("look"),
        "file": record.get("file"),
        "asset_type": record.get("asset_type"),
        "metadata_source": record.get("metadata_source"),
        "raw_completeness": record.get("raw_completeness"),
        **facets,
        "description_raw": description_raw,
        "materials_raw": materials_raw,
        "remark_raw": remark_raw,
        "description_short": build_description_short(facets, description_raw),
        "remark_short": build_remark_short(facets, remark_raw, norm.get("working_process")),
        "facet_confidence": compute_facet_confidence(record, facets),
        "facet_source": "derived_from_grounded_v1",
        "needs_image_fill": record.get("raw_completeness") != "full" or count_nonempty_groups(facets) <= 2,
        "validation_passed": valid,
        "validation_errors": invalid,
        "schema_version": SCHEMA_VERSION,
    }
    diagnostics = {
        "record_id": record["record_id"],
        "nonempty_facet_groups": count_nonempty_groups(facets),
        "derivation_trace": derivation_trace,
        "validation_errors": invalid,
    }
    return compact, diagnostics


def derive_archive(input_jsonl: Path = INPUT_JSONL, vocab_path: Path = VOCAB_PATH, output_jsonl: Path = OUTPUT_JSONL, output_report: Path = OUTPUT_REPORT) -> Tuple[Path, Path]:
    vocab = load_vocabulary(vocab_path)
    compiled_patterns = compile_patterns(vocab)
    allowed_by_field = {field: set(values) for field, values in vocab["facet_fields"].items()}

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    season_counts = Counter()
    asset_type_counts = Counter()
    needs_image_fill_counts = Counter()
    validation_counts = Counter()
    nonempty_field_counts = Counter()
    confidence_bands = Counter()
    facet_population = Counter()
    validation_error_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    with output_jsonl.open("w", encoding="utf-8") as f_out:
        for record in load_jsonl(input_jsonl):
            compact, diagnostics = derive_facets(record, compiled_patterns, allowed_by_field)
            f_out.write(json.dumps(compact, ensure_ascii=False) + "\n")

            total += 1
            season_counts[compact["season"]] += 1
            asset_type_counts[compact["asset_type"]] += 1
            needs_image_fill_counts[str(compact["needs_image_fill"])] += 1
            validation_counts[str(compact["validation_passed"])] += 1

            for field in FACET_FIELDS:
                if compact[field]:
                    nonempty_field_counts[field] += 1
                    facet_population[field] += len(compact[field])

            conf = compact["facet_confidence"]
            if conf >= 0.75:
                confidence_bands[">=0.75"] += 1
            elif conf >= 0.50:
                confidence_bands["0.50-0.74"] += 1
            else:
                confidence_bands["<0.50"] += 1

            if diagnostics["validation_errors"]:
                for field, bad_values in diagnostics["validation_errors"].items():
                    key = f"{field}: {', '.join(bad_values)}"
                    if len(validation_error_examples[key]) < 5:
                        validation_error_examples[key].append({
                            "record_id": compact["record_id"],
                            "season": compact["season"],
                            "values": bad_values,
                        })

    report = {
        "schema_version": SCHEMA_VERSION,
        "input_jsonl": str(input_jsonl),
        "vocabulary": str(vocab_path),
        "output_jsonl": str(output_jsonl),
        "total_records": total,
        "season_counts": dict(season_counts),
        "asset_type_counts": dict(asset_type_counts),
        "needs_image_fill_counts": dict(needs_image_fill_counts),
        "validation_counts": dict(validation_counts),
        "confidence_bands": dict(confidence_bands),
        "nonempty_field_counts": dict(nonempty_field_counts),
        "avg_values_per_nonempty_field": {
            field: round(facet_population[field] / nonempty_field_counts[field], 3)
            for field in FACET_FIELDS if nonempty_field_counts[field]
        },
        "validation_error_examples": dict(validation_error_examples),
    }
    output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_jsonl, output_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Derive Ferré compact retrieval facets from normalized grounded metadata.")
    parser.add_argument("--input-jsonl", type=Path, default=INPUT_JSONL)
    parser.add_argument("--vocab-path", type=Path, default=VOCAB_PATH)
    parser.add_argument("--output-jsonl", type=Path, default=OUTPUT_JSONL)
    parser.add_argument("--output-report", type=Path, default=OUTPUT_REPORT)
    args = parser.parse_args()

    out_jsonl, out_report = derive_archive(args.input_jsonl, args.vocab_path, args.output_jsonl, args.output_report)
    print(f"Wrote {out_jsonl}")
    print(f"Wrote {out_report}")

