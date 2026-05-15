# Image Metadata Pipeline

This module contains the cleaned final image metadata pipeline for Ferre archive retrieval. It combines PDF-grounded archive records, missing-PDF visual clustering, and a controlled fashion ontology to produce image-level metadata for ChromaDB retrieval and API/Postgres browsing.

## Folder Structure

- `fashion_ontology/`: controlled vocabulary and ontology used by the LLM metadata prompt.
- `grounded_clustering/`: prepares clusters for images with PDF/archive-grounded records.
- `missing_pdf_clustering/`: prepares clusters for images without usable PDF records using image embeddings.
- `final_metadata_generation/`: combines grounded and missing-PDF clusters, then generates final structured metadata.

## Final Inputs

- Grounded outfit records: `grounded_clustering/grounded_outfit/*.json`
- Cleaned grounded image map: `grounded_clustering/grounded_image_map/all_collections_merged_cleaned.json`
- Missing-PDF embedding input: `missing_pdf_clustering/embeddings_missing_from_grounded.json`
- Fashion ontology: `fashion_ontology/ferre_retrieval_ontology.json`

## Key Scripts

- `fashion_ontology/build_ferre_retrieval_ontology.py` builds the controlled retrieval ontology.
- `grounded_clustering/build_grounded_outfit_metadata.py` builds PDF-grounded outfit records.
- `grounded_clustering/build_grounded_outfit_dhash_clusters_registry.py` creates grounded outfit clusters.
- `grounded_clustering/clean_grounded_image_map.py` cleans and merges grounded image maps.
- `grounded_clustering/pdf_category.py` assigns PDF availability status.
- `missing_pdf_clustering/find_missing_grounded_embeddings.py` identifies embedded images not covered by grounded records.
- `missing_pdf_clustering/image_embedding_clustering_reclustered.py` clusters missing-PDF images by visual embedding.
- `final_metadata_generation/combine_clusters.py` combines grounded and missing-PDF clusters.
- `final_metadata_generation/metadata_generation.py` generates final image-level metadata.

## Final Outputs

- Combined cluster registry: `final_metadata_generation/all_outfit_clusters_registry_final.jsonl`
- Final image metadata: `final_metadata_generation/generated_image_metadata_final.jsonl`
- API seed copy: `src/api-service/api/seeds/generated_image_metadata_final.jsonl`

The API seed copy should always match the final image metadata file exactly.

## Downstream Use

`src/vector-db/cli.py::load_fashion_show_photos()` loads fashion-show image embeddings into the ChromaDB collection `images-fashion-show-photos` and enriches each record by matching embedding `path` to metadata `source_path`.

`src/api-service/api/seeds/seed.py` loads the API seed JSONL into Postgres as `FashionItem` rows. The frontend consumes this metadata indirectly through the `/archive/*` API routes.

## Rerunning

For normal use, rerunning is not required; use the checked-in final metadata artifact. To regenerate, run the stage scripts from `src/vector-db` so relative paths resolve consistently, rebuild the combined cluster registry, and then run `final_metadata_generation/metadata_generation.py`.

Older debug, batch, draft, and path-repair artifacts were removed from this clean submission repo. The full working history remains in the original development repo.
