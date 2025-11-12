import os
import math
import random
from typing import List, Optional, Literal, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import create_document, get_documents, db

# --------------------------------------
# FastAPI setup
# --------------------------------------
app = FastAPI(title="Architectural AI Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------
# Schemas (align with backend/schemas.py models)
# --------------------------------------
class RequiredSpace(BaseModel):
    name: str
    min_area: float = Field(..., gt=0)

class SiteConstraints(BaseModel):
    width: float = Field(..., gt=0)
    height: float = Field(..., gt=0)
    boundary_file_url: Optional[str] = None

class Project(BaseModel):
    title: str
    project_type: Literal[
        "Residential (Single-family)",
        "Residential (Multi-family)",
        "Commercial (Office)",
        "Commercial (Retail)",
        "Mixed-use",
        "Other",
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
        "None",
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
    project_id: str
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


# --------------------------------------
# Utility functions
# --------------------------------------

def _pack_rooms_into_site(site_w: float, site_h: float, spaces: List[RequiredSpace]) -> List[RoomGeometry]:
    """Very simple rectangle packing by rows, using aspect ~1.2 for rooms.
    This is a deterministic layout generator for demo purposes.
    """
    rooms: List[RoomGeometry] = []
    padding = 0.5  # minimal spacing in meters between rooms
    cursor_x, cursor_y = padding, padding
    row_max_h = 0
    max_w = site_w - padding

    for sp in spaces:
        area = sp.min_area
        # pick width/height with aspect ratio range 1.0-1.6
        aspect = 1.2
        w = max(math.sqrt(area * aspect), 2.0)
        h = max(area / w, 2.0)

        # wrap to next row if exceeds site width
        if cursor_x + w + padding > max_w:
            cursor_x = padding
            cursor_y += row_max_h + padding
            row_max_h = 0

        # if exceeds site height, scale down uniformly to fit remaining space
        if cursor_y + h + padding > site_h:
            scale = max(0.6, (site_h - cursor_y - padding) / max(h, 0.001))
            w *= scale
            h *= scale

        rooms.append(RoomGeometry(name=sp.name, x=cursor_x, y=cursor_y, width=w, height=h, area=w * h))
        cursor_x += w + padding
        row_max_h = max(row_max_h, h)

    return rooms


def _score_plan(rooms: List[RoomGeometry], site_w: float, site_h: float) -> float:
    used_area = sum(r.area for r in rooms)
    site_area = site_w * site_h
    coverage = min(used_area / site_area, 1.0)
    compactness = 1.0 - (max(r.x + r.width for r in rooms) - min(r.x for r in rooms)) / site_w * 0.1
    return round(0.6 * coverage + 0.4 * compactness + random.uniform(-0.05, 0.05), 3)


def _regulatory_rules(code: str) -> Dict[str, float]:
    # very simplified demo rules
    if code == "BBMP":
        return {"setback": 1.5, "far": 1.75, "min_room": 7.5}
    if code == "BMC":
        return {"setback": 1.2, "far": 2.0, "min_room": 7.0}
    if code in ("GDC/MCD", "GDC", "MCD"):
        return {"setback": 1.0, "far": 1.8, "min_room": 7.0}
    # National default
    return {"setback": 1.2, "far": 1.8, "min_room": 7.0}


def _check_compliance(project: Project, plan: Plan) -> List[ComplianceItem]:
    items: List[ComplianceItem] = []

    # Regulatory checks
    rules = _regulatory_rules(project.municipal_code)
    site_w, site_h = plan.site_width, plan.site_height
    total_built = sum(r.area for r in plan.rooms)

    # Setbacks (assume rooms must be inside site minus setbacks)
    setback = rules["setback"]
    ok_setback = all(
        r.x >= setback and r.y >= setback and (r.x + r.width) <= (site_w - setback) and (r.y + r.height) <= (site_h - setback)
        for r in plan.rooms
    )
    items.append(ComplianceItem(
        name="Setbacks",
        passed=ok_setback,
        message=("Within required setbacks" if ok_setback else f"Increase setback to {setback}m or shrink rooms near edges"),
        category="regulatory",
    ))

    # FAR (total built-up area / site area)
    far_limit = rules["far"]
    far_value = total_built / (site_w * site_h)
    ok_far = far_value <= far_limit
    items.append(ComplianceItem(
        name="FAR",
        passed=ok_far,
        message=(f"FAR {far_value:.2f} within limit {far_limit}" if ok_far else f"Reduce built area to meet FAR {far_limit}"),
        category="regulatory",
    ))

    # Min room sizes
    min_room = rules["min_room"]
    small_rooms = [r.name for r in plan.rooms if r.area < min_room]
    ok_min = len(small_rooms) == 0
    items.append(ComplianceItem(
        name="Minimum Room Size",
        passed=ok_min,
        message=("All rooms meet minimum area" if ok_min else f"These rooms are small: {', '.join(small_rooms)} (min {min_room} m^2)"),
        category="regulatory",
    ))

    # Cultural checks (very simplified positional checks)
    def quadrant(rx: RoomGeometry) -> str:
        # Origin at top-left; assume North at top, East at right
        cx = rx.x + rx.width / 2
        cy = rx.y + rx.height / 2
        horiz = "East" if cx > site_w / 2 else "West"
        vert = "South" if cy > site_h / 2 else "North"
        return f"{vert}-{horiz}"

    room_by_name = {r.name.lower(): r for r in plan.rooms}

    if project.cultural_tuning in ("General Vastu", "North Indian Vaastu", "South Indian Vaastu"):
        # Kitchen in South-East
        if "kitchen" in room_by_name:
            q = quadrant(room_by_name["kitchen"])
            ok = q == "South-East"
            items.append(ComplianceItem(
                name="Vastu: Kitchen",
                passed=ok,
                message=("Kitchen in South-East (VASTU COMPLIANT)" if ok else f"Kitchen currently in {q}; prefer South-East"),
                category="cultural",
            ))
        # Master Bedroom in South-West (optional)
        if "master bedroom" in room_by_name:
            q = quadrant(room_by_name["master bedroom"])
            ok = q == "South-West"
            items.append(ComplianceItem(
                name="Vastu: Master Bedroom",
                passed=ok,
                message=("Master Bedroom in South-West" if ok else f"Master Bedroom in {q}; prefer South-West"),
                category="cultural",
            ))
        # Puja priority for South Indian Vaastu
        if project.cultural_tuning == "South Indian Vaastu" and "puja" in room_by_name:
            q = quadrant(room_by_name["puja"])
            ok = q in ("North-East", "East-North") if False else q == "North-East"  # normalize to NE
            items.append(ComplianceItem(
                name="Vastu: Puja Room",
                passed=ok,
                message=("Puja in North-East" if ok else f"Puja in {q}; prefer North-East"),
                category="cultural",
            ))

    if project.cultural_tuning == "Islamic Beliefs":
        # Living facing Qibla (approx West-SouthWest in India). We'll check living in West half.
        if "living" in room_by_name:
            q = quadrant(room_by_name["living"])
            ok = "West" in q
            items.append(ComplianceItem(
                name="Islamic: Living Orientation",
                passed=ok,
                message=("Living oriented West-half (Qibla consideration)" if ok else f"Living in {q}; consider West orientation"),
                category="cultural",
            ))

    if project.cultural_tuning == "Christian Beliefs":
        # Optional chapel/altar: prefer North-East
        if "chapel" in room_by_name or "altar" in room_by_name:
            r = room_by_name.get("chapel") or room_by_name.get("altar")
            q = quadrant(r)
            ok = q == "North-East"
            items.append(ComplianceItem(
                name="Christian: Altar/Chapel",
                passed=ok,
                message=("Altar/Chapel in North-East" if ok else f"Currently {q}; prefer North-East"),
                category="cultural",
            ))

    return items


def _estimate_materials(plan: Plan) -> Estimate:
    floor_area = sum(r.area for r in plan.rooms)
    perimeter = sum(2 * (r.width + r.height) for r in plan.rooms)
    wall_height = 3.0  # meters

    bom: List[EstimateItem] = []

    # Masonry blocks (assume 0.2m thick walls, 1 block = 0.2 x 0.2 x 0.4 ~ 0.016 m^3; use linear approx)
    wall_area = perimeter * wall_height
    blocks = wall_area / (0.2 * 0.4)  # approx blocks per m^2 using 0.2x0.4 face
    bom.append(EstimateItem(item="Concrete/Clay Blocks", unit="units", quantity=round(blocks, 0), unit_rate=35.0))

    # Flooring tiles
    bom.append(EstimateItem(item="Ceramic/Vitrified Tiles", unit="m^2", quantity=round(floor_area, 2), unit_rate=950.0))

    # Plaster
    bom.append(EstimateItem(item="Cement Plaster (both sides)", unit="m^2", quantity=round(wall_area * 2, 2), unit_rate=220.0))

    # Paint
    bom.append(EstimateItem(item="Interior Paint", unit="m^2", quantity=round(wall_area * 2 + floor_area, 2), unit_rate=120.0))

    # Doors (assume 1 per room)
    bom.append(EstimateItem(item="Flush Doors", unit="units", quantity=len(plan.rooms), unit_rate=4500.0))

    # Windows (assume 1 per room average)
    bom.append(EstimateItem(item="Aluminium Windows", unit="units", quantity=len(plan.rooms), unit_rate=3800.0))

    # Compute costs
    total_low, total_high = 0.0, 0.0
    for it in bom:
        if it.unit_rate is not None:
            cost = it.quantity * it.unit_rate
            it.cost = round(cost, 2)
            total_low += cost * 0.9
            total_high += cost * 1.1

    return Estimate(plan_id=plan.plan_id, bom=bom, total_cost_low=round(total_low, 2), total_cost_high=round(total_high, 2))


def _plan_to_svg(plan: Plan, show_dims: bool = True) -> str:
    w = plan.site_width
    h = plan.site_height
    scale = 40  # 1m = 40px
    svg_elems = []

    # Site boundary
    svg_elems.append(f'<rect x="0" y="0" width="{w*scale}" height="{h*scale}" fill="#f8fafc" stroke="#94a3b8" stroke-width="2" />')

    colors = ["#a7f3d0", "#fde68a", "#bfdbfe", "#fca5a5", "#c7d2fe", "#fdba74", "#86efac"]

    for idx, r in enumerate(plan.rooms):
        x, y, rw, rh = r.x*scale, r.y*scale, r.width*scale, r.height*scale
        fill = colors[idx % len(colors)]
        svg_elems.append(f'<rect x="{x}" y="{y}" width="{rw}" height="{rh}" fill="{fill}" stroke="#111827" stroke-width="1.5" />')
        svg_elems.append(f'<text x="{x+rw/2}" y="{y+rh/2}" text-anchor="middle" dominant-baseline="middle" font-size="12" fill="#111827">{r.name}\n{r.area:.1f} m²</text>')
        if show_dims:
            svg_elems.append(f'<text x="{x+4}" y="{y+12}" font-size="10" fill="#374151">{r.width:.1f} x {r.height:.1f}m</text>')

    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w*scale} {h*scale}" width="100%" height="100%">' + "".join(svg_elems) + "</svg>"
    return svg


# --------------------------------------
# API routes
# --------------------------------------
@app.get("/")
def read_root():
    return {"message": "Architectural AI Agent Backend"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": [],
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = db.name if hasattr(db, "name") else "Unknown"
            response["connection_status"] = "Connected"
            try:
                response["collections"] = db.list_collection_names()
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but error: {str(e)[:80]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response


# Create Project
@app.post("/api/projects")
def create_project(project: Project):
    pid = create_document("project", project)
    return {"project_id": pid}


# List Projects
@app.get("/api/projects")
def list_projects():
    return get_documents("project", {})


# Get single project
@app.get("/api/projects/{project_id}")
def get_project(project_id: str):
    from bson import ObjectId
    try:
        proj = db.project.find_one({"_id": ObjectId(project_id)})
        if not proj:
            raise HTTPException(status_code=404, detail="Project not found")
        proj["_id"] = str(proj["_id"]) 
        return proj
    except Exception:
        raise HTTPException(status_code=404, detail="Project not found")


class GenerateRequest(BaseModel):
    alternatives: int = 4


# Generate plans for a project
@app.post("/api/projects/{project_id}/generate")
def generate_plans(project_id: str, req: GenerateRequest):
    proj_data = get_project(project_id)
    project = Project(**{
        "title": proj_data.get("title"),
        "project_type": proj_data.get("project_type"),
        "site": proj_data.get("site"),
        "required_spaces": proj_data.get("required_spaces", []),
        "adjacency_notes": proj_data.get("adjacency_notes"),
        "orientation_notes": proj_data.get("orientation_notes"),
        "cultural_tuning": proj_data.get("cultural_tuning", "General Vastu"),
        "municipal_code": proj_data.get("municipal_code", "National"),
        "status": proj_data.get("status", "created"),
    })

    alternatives = max(1, min(5, req.alternatives))
    out: List[Plan] = []
    spaces = [RequiredSpace(**s) for s in project.required_spaces]

    for i in range(alternatives):
        # Shuffle spaces slightly per alternative
        random.shuffle(spaces)
        rooms = _pack_rooms_into_site(project.site.width, project.site.height, spaces)
        score = _score_plan(rooms, project.site.width, project.site.height)
        plan = Plan(
            plan_id=f"{project_id}-{i+1}",
            project_id=project_id,
            site_width=project.site.width,
            site_height=project.site.height,
            rooms=rooms,
            score=score,
            notes=f"Alt {i+1}",
        )
        create_document("plan", plan)
        out.append(plan)

    # update project status (insert new doc with same _id not provided -> skip; viewer is demo)
    return {"plans": [p.model_dump() for p in out]}


# Get plans for a project
@app.get("/api/projects/{project_id}/plans")
def get_plans(project_id: str):
    plans = get_documents("plan", {"project_id": project_id})
    for p in plans:
        p.pop("_id", None)
    return plans


# Compliance for a plan
@app.get("/api/projects/{project_id}/plans/{plan_id}/compliance")
def check_plan_compliance(project_id: str, plan_id: str):
    plans = get_documents("plan", {"project_id": project_id})
    plan_data = next((p for p in plans if p.get("plan_id") == plan_id), None)
    if not plan_data:
        raise HTTPException(status_code=404, detail="Plan not found")

    proj_data = get_project(project_id)
    project = Project(**proj_data)
    plan = Plan(**plan_data)

    items = _check_compliance(project, plan)
    return {"items": [i.model_dump() for i in items]}


# Estimate for a plan
@app.get("/api/projects/{project_id}/plans/{plan_id}/estimate")
def estimate_plan(project_id: str, plan_id: str):
    plans = get_documents("plan", {"project_id": project_id})
    plan_data = next((p for p in plans if p.get("plan_id") == plan_id), None)
    if not plan_data:
        raise HTTPException(status_code=404, detail="Plan not found")

    plan = Plan(**plan_data)
    est = _estimate_materials(plan)
    return est.model_dump()


# SVG export for a plan
@app.get("/api/projects/{project_id}/plans/{plan_id}/svg")
def export_plan_svg(project_id: str, plan_id: str):
    plans = get_documents("plan", {"project_id": project_id})
    plan_data = next((p for p in plans if p.get("plan_id") == plan_id), None)
    if not plan_data:
        raise HTTPException(status_code=404, detail="Plan not found")
    plan = Plan(**plan_data)
    svg = _plan_to_svg(plan)
    return {"svg": svg}


# CSV export for BOM
@app.get("/api/projects/{project_id}/plans/{plan_id}/bom.csv")
def export_bom_csv(project_id: str, plan_id: str):
    from fastapi.responses import PlainTextResponse

    plans = get_documents("plan", {"project_id": project_id})
    plan_data = next((p for p in plans if p.get("plan_id") == plan_id), None)
    if not plan_data:
        raise HTTPException(status_code=404, detail="Plan not found")

    plan = Plan(**plan_data)
    est = _estimate_materials(plan)

    lines = ["Item,Unit,Quantity,Unit Rate,Cost"]
    for it in est.bom:
        lines.append(f"{it.item},{it.unit},{it.quantity},{it.unit_rate or ''},{it.cost or ''}")
    content = "\n".join(lines)
    return PlainTextResponse(content, media_type="text/csv")


# Project report (JSON bundle)
@app.get("/api/projects/{project_id}/report")
def project_report(project_id: str):
    proj = get_project(project_id)
    plans = get_plans(project_id)
    if not plans:
        return {"project": proj, "plans": [], "message": "No plans generated yet"}

    first_plan_id = plans[0]["plan_id"]
    compliance = check_plan_compliance(project_id, first_plan_id)
    estimate = estimate_plan(project_id, first_plan_id)
    svg = export_plan_svg(project_id, first_plan_id)

    return {
        "project": proj,
        "selected_plan": first_plan_id,
        "compliance": compliance,
        "estimate": estimate,
        "svg": svg.get("svg"),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
