from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import jwt

logger = logging.getLogger("annpack.registry")

STORAGE_ROOT = Path(os.environ.get("REGISTRY_STORAGE", "registry_storage")).resolve()
DB_PATH = STORAGE_ROOT / "registry.db"
DEV_MODE = os.getenv("REGISTRY_DEV_MODE", "").lower() in {"1", "true", "yes"}
JWT_SECRET = os.getenv("REGISTRY_JWT_SECRET")
if not DEV_MODE and not JWT_SECRET:
    raise RuntimeError("REGISTRY_JWT_SECRET is required when REGISTRY_DEV_MODE is disabled")
if DEV_MODE and not JWT_SECRET:
    JWT_SECRET = "dev-secret"
    logger.warning("DEV_MODE enabled without REGISTRY_JWT_SECRET; using insecure dev secret.")
JWT_ALG = os.environ.get("REGISTRY_JWT_ALG", "HS256")
JWT_AUD = os.environ.get("REGISTRY_JWT_AUD", "annpack-registry")
JWT_ISS = os.environ.get("REGISTRY_JWT_ISS", "annpack-registry")
MAX_UPLOAD_MB = int(os.getenv("REGISTRY_MAX_UPLOAD_MB", "100") or "100")
MAX_UPLOAD_BYTES = max(1, MAX_UPLOAD_MB) * 1024 * 1024
MAX_JSON_BYTES = int(os.environ.get("REGISTRY_MAX_JSON_BYTES", str(1024 * 1024)))
RATE_LIMIT_RPS = float(os.getenv("REGISTRY_RATE_LIMIT_RPS", "5") or "5")
RATE_LIMIT_BURST = int(os.getenv("REGISTRY_RATE_LIMIT_BURST", "20") or "20")
_RATE_BUCKETS: Dict[str, Tuple[float, float]] = {}

app = FastAPI(title="ANNPack Registry", version="0.1")


def _rate_limit(request: Request) -> None:
    if RATE_LIMIT_RPS <= 0 or RATE_LIMIT_BURST <= 0:
        return
    key = request.client.host if request.client else "unknown"
    now = time.monotonic()
    tokens, last = _RATE_BUCKETS.get(key, (float(RATE_LIMIT_BURST), now))
    tokens = min(float(RATE_LIMIT_BURST), tokens + (now - last) * RATE_LIMIT_RPS)
    if tokens < 1.0:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    _RATE_BUCKETS[key] = (tokens - 1.0, now)


@app.middleware("http")
async def _rate_limit_middleware(request: Request, call_next):
    _rate_limit(request)
    return await call_next(request)


class _BodyLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length"})
            ctype = (request.headers.get("content-type") or "").lower()
            if "application/json" in ctype and size > MAX_JSON_BYTES:
                return JSONResponse(status_code=413, content={"detail": "JSON body too large"})
            if "multipart/form-data" in ctype and size > MAX_UPLOAD_BYTES:
                return JSONResponse(status_code=413, content={"detail": "Upload too large"})
        return await call_next(request)


app.add_middleware(_BodyLimitMiddleware)


def init_db() -> None:
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS orgs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            );
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                org_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                UNIQUE(org_id, name)
            );
            CREATE TABLE IF NOT EXISTS versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                version TEXT NOT NULL,
                path TEXT NOT NULL,
                created_at REAL NOT NULL,
                UNIQUE(project_id, version)
            );
            CREATE TABLE IF NOT EXISTS audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            """
        )


@app.on_event("startup")
def _startup() -> None:
    if not DEV_MODE:
        if not JWT_SECRET or len(JWT_SECRET) < 16:
            raise RuntimeError(
                "REGISTRY_JWT_SECRET is required and must be at least 16 characters "
                "when REGISTRY_DEV_MODE=0"
            )
        if not JWT_AUD or not JWT_ISS:
            raise RuntimeError("REGISTRY_JWT_AUD and REGISTRY_JWT_ISS must be set")
    init_db()


def log_audit(actor: str, action: str, details: dict) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO audit(actor, action, details, created_at) VALUES(?,?,?,?)",
            (actor, action, json.dumps(details), time.time()),
        )
        conn.commit()


def get_user(authorization: Optional[str] = Header(None)) -> dict:
    if DEV_MODE and not authorization:
        return {"sub": "dev", "role": "admin"}
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        options = {"require": ["sub"]}
        if DEV_MODE:
            data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG], options=options)
        else:
            data = jwt.decode(
                token,
                JWT_SECRET,
                algorithms=[JWT_ALG],
                audience=JWT_AUD,
                issuer=JWT_ISS,
                options=options,
            )
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
    return data


def require_role(user: dict, roles: set[str]) -> None:
    if user.get("role") not in roles:
        raise HTTPException(status_code=403, detail="Insufficient role")


@app.post("/auth/dev-token")
def dev_token(role: str = "admin"):
    if not DEV_MODE:
        raise HTTPException(status_code=403, detail="Dev tokens disabled")
    if not JWT_SECRET:
        raise HTTPException(status_code=500, detail="REGISTRY_JWT_SECRET is required")
    payload = {"sub": "dev", "role": role}
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)
    return {"token": token}


@app.post("/orgs")
def create_org(name: str, user: dict = Depends(get_user)):
    require_role(user, {"admin"})
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO orgs(name) VALUES(?)", (name,))
        conn.commit()
    log_audit(user["sub"], "create_org", {"name": name})
    return {"name": name}


@app.post("/orgs/{org}/projects")
def create_project(org: str, name: str, user: dict = Depends(get_user)):
    require_role(user, {"admin", "dev"})
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT id FROM orgs WHERE name=?", (org,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Org not found")
        org_id = row[0]
        conn.execute("INSERT INTO projects(org_id, name) VALUES(?,?)", (org_id, name))
        conn.commit()
    log_audit(user["sub"], "create_project", {"org": org, "name": name})
    return {"org": org, "name": name}


def _project_id(conn: sqlite3.Connection, org: str, project: str) -> int:
    cur = conn.execute(
        "SELECT projects.id FROM projects JOIN orgs ON projects.org_id=orgs.id WHERE orgs.name=? AND projects.name=?",
        (org, project),
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")
    return int(row[0])


def _safe_extract(zip_path: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            rel = Path(member.replace("\\", "/"))
            if rel.is_absolute() or ".." in rel.parts:
                raise HTTPException(status_code=400, detail="Unsafe zip path")
            target = out_dir / rel
            if not str(target.resolve()).startswith(str(out_dir.resolve())):
                raise HTTPException(status_code=400, detail="Unsafe zip path")
        zf.extractall(out_dir)


def _safe_path(base: Path, rel: str) -> Path:
    rel_path = Path(rel.replace("\\", "/"))
    if rel_path.is_absolute() or ".." in rel_path.parts:
        raise HTTPException(status_code=400, detail="Invalid path")
    target = (base / rel_path).resolve()
    if not str(target).startswith(str(base.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    return target


@app.post("/orgs/{org}/projects/{project}/packs")
def upload_pack(
    org: str,
    project: str,
    version: str,
    bundle: UploadFile = File(...),
    user: dict = Depends(get_user),
):
    require_role(user, {"admin", "dev"})
    tmp = STORAGE_ROOT / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    bundle_path = tmp / f"upload_{int(time.time() * 1000)}.zip"
    total = 0
    with bundle_path.open("wb") as handle:
        while True:
            chunk = bundle.file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Upload too large")
            handle.write(chunk)

    with sqlite3.connect(DB_PATH) as conn:
        project_id = _project_id(conn, org, project)
        cur = conn.execute(
            "SELECT 1 FROM versions WHERE project_id=? AND version=?", (project_id, version)
        )
        if cur.fetchone():
            raise HTTPException(status_code=409, detail="Version already exists")
        dest_dir = STORAGE_ROOT / org / project / version
        dest_dir.mkdir(parents=True, exist_ok=True)
        _safe_extract(bundle_path, dest_dir)
        conn.execute(
            "INSERT INTO versions(project_id, version, path, created_at) VALUES(?,?,?,?)",
            (project_id, version, str(dest_dir), time.time()),
        )
        conn.commit()
    log_audit(user["sub"], "upload_pack", {"org": org, "project": project, "version": version})
    return {"org": org, "project": project, "version": version}


@app.get("/orgs/{org}/projects/{project}/packs")
def list_versions(org: str, project: str, user: dict = Depends(get_user)):
    require_role(user, {"admin", "dev", "viewer"})
    with sqlite3.connect(DB_PATH) as conn:
        project_id = _project_id(conn, org, project)
        cur = conn.execute(
            "SELECT version FROM versions WHERE project_id=? ORDER BY created_at DESC",
            (project_id,),
        )
        versions = [row[0] for row in cur.fetchall()]
    return {"org": org, "project": project, "versions": versions}


def _version_path(org: str, project: str, version: str) -> Path:
    path = STORAGE_ROOT / org / project / version
    if not path.exists():
        raise HTTPException(status_code=404, detail="Version not found")
    return path


@app.get("/orgs/{org}/projects/{project}/packs/{version}/manifest")
def get_manifest(org: str, project: str, version: str, user: dict = Depends(get_user)):
    require_role(user, {"admin", "dev", "viewer"})
    path = _version_path(org, project, version)
    manifest = next(path.glob("*.manifest.json"), None)
    if not manifest:
        raise HTTPException(status_code=404, detail="Manifest not found")
    return FileResponse(manifest)


@app.get("/orgs/{org}/projects/{project}/packs/{version}/files/{file_path:path}")
def get_file(
    org: str,
    project: str,
    version: str,
    file_path: str,
    range: Optional[str] = Header(None),
    user: dict = Depends(get_user),
):
    require_role(user, {"admin", "dev", "viewer"})
    base = _version_path(org, project, version)
    target = _safe_path(base, file_path)
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")

    file_size = target.stat().st_size
    if range and range.startswith("bytes="):
        try:
            start_s, end_s = range.replace("bytes=", "").split("-")
            start = int(start_s)
            end = int(end_s) if end_s else file_size - 1
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Range header")
        if start >= file_size:
            return Response(status_code=416)
        end = min(end, file_size - 1)
        length = end - start + 1
        with target.open("rb") as f:
            f.seek(start)
            data = f.read(length)
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
        }
        return Response(
            content=data, status_code=206, headers=headers, media_type="application/octet-stream"
        )

    return FileResponse(target)
