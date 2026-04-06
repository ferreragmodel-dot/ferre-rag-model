from typing import Optional, List
from sqlmodel import SQLModel, Field
from sqlalchemy import Column, JSON


class FashionItem(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    source_path: str
    collection_line: str
    season_path: Optional[str] = None
    year_path: Optional[str] = None
    asset_type: str
    season: Optional[str] = None
    label: Optional[str] = None
    acquisition: Optional[str] = None
    look: Optional[str] = None
    file: Optional[str] = None
    inventory: Optional[str] = None
    object: Optional[str] = None
    source: Optional[str] = None
    description: Optional[str] = None
    exhibitions: Optional[str] = None
    size: Optional[str] = None
    materials: Optional[str] = None
    present_location: Optional[str] = None
    remark: Optional[str] = None
    bibliography: Optional[str] = None
    designer: Optional[str] = None
    working_process: Optional[str] = None
    condition: Optional[str] = None
    collection: Optional[str] = None
    year: Optional[str] = None
    garments_tags: List[str] = Field(default=[], sa_column=Column(JSON))
    colors_tags: List[str] = Field(default=[], sa_column=Column(JSON))
    material_tags: List[str] = Field(default=[], sa_column=Column(JSON))
    patterns_tags: List[str] = Field(default=[], sa_column=Column(JSON))
    silhouette_tags: List[str] = Field(default=[], sa_column=Column(JSON))
    length_tags: List[str] = Field(default=[], sa_column=Column(JSON))
    neckline_tags: List[str] = Field(default=[], sa_column=Column(JSON))
    sleeve_tags: List[str] = Field(default=[], sa_column=Column(JSON))
    closure_tags: List[str] = Field(default=[], sa_column=Column(JSON))
    embellishment_tags: List[str] = Field(default=[], sa_column=Column(JSON))
    style_tags: List[str] = Field(default=[], sa_column=Column(JSON))
    llm_description: Optional[str] = None
    pdf_available: str
    cluster_id: str
    outfit_id: str