# Phase 03 — grounded to compact facet derivation

## New script

- `metadata/derive_grounded_compact_facets.py`

## Input

- `metadata/normalized_grounded/grounded_raw.jsonl`
- `metadata/config/ferre_facet_vocabulary_v1.json`

## Output

- `metadata/derived_grounded_facets/grounded_compact.jsonl`
- `metadata/derived_grounded_facets/grounded_compact_report.json`

## What this phase does

This phase converts the normalized grounded archive into compact retrieval facets.
It is deterministic and does not call any LLM.

Derived fields:

- `garments`
- `colors`
- `material_families`
- `patterns`
- `silhouette_tags`
- `length_tags`
- `neckline_tags`
- `sleeve_tags`
- `closure_tags`
- `embellishment_tags`
- `style_tags`
- `description_short`
- `remark_short`
- `facet_confidence`
- `needs_image_fill`

It also keeps the free-text archive fields needed later for answer generation:

- `description_raw`
- `materials_raw`
- `remark_raw`

## Where to run it

From `src/vector-db`:

```bash
python metadata/derive_grounded_compact_facets.py
```

## What the next phase should read

Phase 04 should read:

- `metadata/derived_grounded_facets/grounded_compact.jsonl`

and use:

- `needs_image_fill`
- `raw_completeness`
- `facet_confidence`

to decide which grounded records need image-side completion.
