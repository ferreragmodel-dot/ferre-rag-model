# Phase 04 — fill partial grounded facets

## New script

- `metadata/fill_partial_grounded_facets.py`

## Inputs

Default inputs:

- `metadata/derived_grounded_facets/grounded_compact.jsonl`
- `metadata/normalized_grounded/grounded_raw.jsonl`
- `metadata/config/ferre_facet_vocabulary_v1.json`
- `input-datasets/ferre-designs/`

You can swap the phase-03 input to the LLM output later:

- `metadata/derived_grounded_facets/grounded_compact_llm.jsonl`

## Outputs

- `metadata/image_filled_grounded_facets/grounded_compact_filled.jsonl`
- `metadata/image_filled_grounded_facets/grounded_compact_filled_report.json`

## Default behavior

The script follows the phase-04 decision table:

1. `description` or `materials` exists -> keep text-derived facets only by default
2. `object` exists but `description/materials` are missing -> preserve confirmed text facets and fill missing facets from image
3. `object`, `description`, and `materials` are all missing -> generate the facet block from image

## Commands

Dry-run planning only:

```bash
cd ferre-rag-model/src/vector-db
python metadata/fill_partial_grounded_facets.py --dry-run
```

Real run with heuristic phase 03 input:

```bash
cd ferre-rag-model/src/vector-db
export GCP_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
export GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/llm-service-account.json"
python metadata/fill_partial_grounded_facets.py
```

Real run with the future LLM phase 03 file:

```bash
cd ferre-rag-model/src/vector-db
python metadata/fill_partial_grounded_facets.py \
  --input metadata/derived_grounded_facets/grounded_compact_llm.jsonl
```

Resume after interruption:

```bash
python metadata/fill_partial_grounded_facets.py --resume
```

Only process records already flagged for image fill:

```bash
python metadata/fill_partial_grounded_facets.py --only-needs-fill
```

Allow image fill even for low-confidence text-rich records:

```bash
python metadata/fill_partial_grounded_facets.py \
  --only-needs-fill \
  --fill-low-confidence-text
```
