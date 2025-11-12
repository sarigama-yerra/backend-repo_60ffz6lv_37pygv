"""
Database Schemas for Architectural AI Agent

Each Pydantic model here represents a MongoDB collection. The collection name
is the lowercase of the class name (e.g., Project -> "project").
"""
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

# =============================
# Core Collections
# =============================

class RequiredSpace(BaseModel):
    name: str
    min_area: float = Field(..., gt=0, description="Minimum area in m^2")

class SiteConstraints(BaseModel):
    width: float = Field(..., gt=0, description="Site width in meters")
    height: float = Field(..., gt=0, description="Site depth/height in meters")
    boundary_file_url: Optional[str] = None

class Project(BaseModel):
    title: str
    project_type: Literal[
        "Residential (Single-family)",
        "Residential (Multi-family)",
        "Commercial (Office)",
        "Commercial (Retail)",
        "Mixed-use",
        "Other"
    ] = "Residential (Single-family)"
    site: SiteConstraints
    required_spaces: List[RequiredSpace] = []
    adjacency_notes: Optional[str] = None
    orientation_notes: Optional[str] = None

    cultural_tuning: Literal[
        "General Vastu",
        "North Indian Vaastu",
        "South Indian Vaastu",
        "Islamic Beliefs",
        "Christian Beliefs",
        "None"
    ] = "General Vastu"

    municipal_code: Literal["BBMP", "BMC", "GDC/MCD", "National", "None"] = "National"

    status: Literal["created", "generated", "finalized"] = "created"

class RoomGeometry(BaseModel):
    name: str
    x: float
    y: float
    width: float
    height: float
    area: float

class Plan(BaseModel):
    plan_id: str
    site_width: float
    site_height: float
    rooms: List[RoomGeometry]
    score: float = 0.0
    notes: Optional[str] = None

class ComplianceItem(BaseModel):
    name: str
    passed: bool
    message: str
    category: Literal["regulatory", "cultural"]

class EstimateItem(BaseModel):
    item: str
    unit: str
    quantity: float
    unit_rate: Optional[float] = None
    cost: Optional[float] = None

class Estimate(BaseModel):
    plan_id: str
    bom: List[EstimateItem]
    total_cost_low: Optional[float] = None
    total_cost_high: Optional[float] = None

# Note: The database helper automatically timestamps documents when inserting.
