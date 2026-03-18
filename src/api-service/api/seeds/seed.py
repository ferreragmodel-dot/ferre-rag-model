"""
Seed the database with fashion items from initial_database_loading.jsonl.
Run from inside the container:
    python -m api.seeds.seed
"""
import json
from pathlib import Path

from sqlmodel import Session, select

from api.db import create_db_and_tables, engine
from api.models.fashion_item import FashionItem

SEEDS_DIR = Path(__file__).parent


def seed():
    create_db_and_tables()

    valid_fields = set(FashionItem.model_fields.keys())

    with open(SEEDS_DIR / "initial_database_loading.jsonl", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    with Session(engine) as session:
        for raw in lines:
            item_data = {k: v for k, v in json.loads(raw).items() if k in valid_fields}

            existing = session.exec(
                select(FashionItem).where(FashionItem.source_path == item_data["source_path"])
            ).first()
            if existing:
                print(f"[skip] already exists: {item_data['source_path']}")
                continue

            item = FashionItem(**item_data)
            session.add(item)
            print(f"[insert] {item_data['source_path']}")

        session.commit()

    print("Done.")


if __name__ == "__main__":
    seed()
