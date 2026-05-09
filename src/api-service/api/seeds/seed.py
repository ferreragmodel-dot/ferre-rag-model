"""
Seed the database with fashion items from generated_image_metadata.jsonl.
Only records with asset_type == "Fashion show photos" are loaded.
Run from inside the container:
    python -m api.seeds.seed
"""
import json
from pathlib import Path

from sqlalchemy import func
from sqlmodel import Session, select

from api.db import create_db_and_tables, engine
from api.models.fashion_item import FashionItem

SEEDS_DIR = Path(__file__).parent


def seed():
    create_db_and_tables()

    valid_fields = set(FashionItem.model_fields.keys())

    with open(SEEDS_DIR / "generated_image_metadata_fixed_paths.jsonl", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    with Session(engine) as session:
        count = session.exec(select(func.count()).select_from(FashionItem)).one()
        if count > 0:
            print(f"[seed] {count} records already present, skipping.")
            return

        for raw in lines:
            record = json.loads(raw)

            if record.get("asset_type") != "Fashion show photos":
                continue

            item_data = {k: v for k, v in record.items() if k in valid_fields}
            session.add(FashionItem(**item_data))
            print(f"[insert] {item_data['source_path']}")

        session.commit()

    print("Done.")


if __name__ == "__main__":
    seed()
