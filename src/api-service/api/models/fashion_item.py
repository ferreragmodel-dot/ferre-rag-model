from typing import Optional
from sqlmodel import SQLModel, Field


class FashionItem(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    outfit_id: Optional[str] = None
    label: Optional[str] = None
    acquisition: Optional[str] = None
    size: Optional[str] = None
    look: Optional[str] = None
    designer: Optional[str] = None
    bibliography: Optional[str] = None
    present_location: Optional[str] = None
    inventory: Optional[str] = None
    condition: Optional[str] = None
    working_process: Optional[str] = None
    exhibitions: Optional[str] = None
    asset_type: str
    source_path: str
    season: str
    year: str
    description: str
    remark: str
