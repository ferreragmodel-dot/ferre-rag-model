# Phase 01–02 install notes

Run these commands from `src/vector-db/`.

## New files
- `metadata/build_normalized_grounded_archive.py`
- `metadata/facet_vocabulary.py`

## Existing input folder
- `metadata/grounded_image_map/*.json`

## New output folders
- `metadata/normalized_grounded/`
- `metadata/config/`

## Commands
```bash
python metadata/build_normalized_grounded_archive.py
python metadata/facet_vocabulary.py
```

## Outputs
- `metadata/normalized_grounded/grounded_raw.jsonl`
- `metadata/normalized_grounded/grounded_raw_report.json`
- `metadata/config/ferre_facet_vocabulary_v1.json`

## Why these locations
These two phases are metadata-preprocessing steps, not embedding/chunk outputs. Keeping them under `metadata/` makes them easy to reuse later from image metadata generation, compact facet derivation, and Chroma loading.

## Next phase inputs
- normalized archive input: `metadata/normalized_grounded/grounded_raw.jsonl`
- vocabulary input: `metadata/config/ferre_facet_vocabulary_v1.json`
