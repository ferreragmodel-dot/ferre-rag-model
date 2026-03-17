"""
Seed the database with fashion items from fashion_items.json.
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

    with open(SEEDS_DIR / "fashion_items.json", encoding="utf-8") as f:
        items = json.load(f)

    with Session(engine) as session:
        for item_data in items:
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
