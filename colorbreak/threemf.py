"""Reading and writing of PrusaSlicer 3.0 ``.3mf`` project files.

Only the parts needed by colorbreak are handled:

* ``3D/3dmodel.model`` - meshes (vertices / triangles), object/component hierarchy and
  transforms, parsed with regular expressions (fast enough for ~10^6 triangles).
* ``Metadata/Slic3r_facets_annotation.json`` - per-triangle painting codes.
* ``Metadata/PrusaSlicer3_project.json`` - project config incl. virtual extruders.

Everything else in the archive is copied through untouched on write.
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

MODEL_PATH = "3D/3dmodel.model"
FACETS_PATH = "Metadata/Slic3r_facets_annotation.json"
PROJECT_PATH = "Metadata/PrusaSlicer3_project.json"

_RE_OBJECT = re.compile(r"<object\b([^>]*)>(.*?)</object>", re.S)
_RE_ATTR = re.compile(r'(\w+)="([^"]*)"')
_RE_VERTEX = re.compile(r'<vertex\s+x="([^"]+)"\s+y="([^"]+)"\s+z="([^"]+)"')
_RE_TRIANGLE = re.compile(r'<triangle\s+v1="(\d+)"\s+v2="(\d+)"\s+v3="(\d+)"')
_RE_COMPONENT = re.compile(r"<component\b([^>]*)/>")
_RE_ITEM = re.compile(r"<item\b([^>]*)/>")


def _parse_transform(s: Optional[str]) -> np.ndarray:
    """3MF transform: 12 numbers, row-major 3x4 (3 rows of a rotation + translation row)."""
    m = np.eye(4)
    if not s:
        return m
    v = [float(x) for x in s.split()]
    if len(v) != 12:
        raise ValueError(f"bad transform {s!r}")
    a = np.array(v).reshape(4, 3)  # rows: m00 m01 m02 / m10.. / m20.. / tx ty tz
    m[:3, :3] = a[:3, :].T
    m[:3, 3] = a[3, :]
    return m


def _format_transform(m: np.ndarray) -> str:
    a = np.zeros((4, 3))
    a[:3, :] = m[:3, :3].T
    a[3, :] = m[:3, 3]
    return " ".join(repr(float(x)) for x in a.reshape(-1))


@dataclass
class Mesh:
    vertices: np.ndarray  # (V, 3) float64
    triangles: np.ndarray  # (T, 3) int64


@dataclass
class ModelObject:
    id: int
    name: str
    mesh: Optional[Mesh] = None
    components: List[Tuple[int, np.ndarray]] = field(default_factory=list)  # (objectid, transform)


@dataclass
class Painting:
    """Painting of one volume (3MF object id = volume object)."""

    volume_id: int
    codes: Dict[int, str]  # triangle index -> hex code
    version: int
    extra: Dict[str, object] = field(default_factory=dict)  # other facet sets (support, seam, ...)


class Project:
    """A loaded 3MF project."""

    def __init__(self, path: str):
        self.path = path
        with zipfile.ZipFile(path) as zf:
            self.names = zf.namelist()
            self.model_xml = zf.read(MODEL_PATH).decode("utf-8")
            self.facets_json = json.loads(zf.read(FACETS_PATH)) if FACETS_PATH in self.names else []
            self.project_json = json.loads(zf.read(PROJECT_PATH)) if PROJECT_PATH in self.names else None
        self.objects: Dict[int, ModelObject] = {}
        self.items: List[Tuple[int, np.ndarray]] = []
        self._parse_model()
        self.paintings: Dict[int, Painting] = {}
        self._parse_facets()

    # ------------------------------------------------------------------ model
    def _parse_model(self) -> None:
        res_start = self.model_xml.find("<resources>")
        res_end = self.model_xml.find("</resources>")
        resources = self.model_xml[res_start:res_end]
        for m in _RE_OBJECT.finditer(resources):
            attrs = dict(_RE_ATTR.findall(m.group(1)))
            body = m.group(2)
            obj = ModelObject(id=int(attrs["id"]), name=attrs.get("name", ""))
            if "<mesh>" in body:
                verts = np.array(_RE_VERTEX.findall(body), dtype=np.float64)
                tris = np.array(_RE_TRIANGLE.findall(body), dtype=np.int64)
                obj.mesh = Mesh(verts.reshape(-1, 3), tris.reshape(-1, 3))
            for c in _RE_COMPONENT.finditer(body):
                ca = dict(_RE_ATTR.findall(c.group(1)))
                obj.components.append((int(ca["objectid"]), _parse_transform(ca.get("transform"))))
            self.objects[obj.id] = obj
        build = self.model_xml[self.model_xml.find("<build>") :]
        for it in _RE_ITEM.finditer(build):
            ia = dict(_RE_ATTR.findall(it.group(1)))
            self.items.append((int(ia["objectid"]), _parse_transform(ia.get("transform"))))

    def _parse_facets(self) -> None:
        for entry in self.facets_json:
            vid = int(entry["id"])
            codes = {int(f["triangle"]): f["dividing"] for f in entry.get("mmSegmentationFacets", [])}
            version = int(entry.get("mmSegmentationFacetsVersion", 1))
            extra = {
                k: v for k, v in entry.items() if k not in ("id", "mmSegmentationFacets", "mmSegmentationFacetsVersion")
            }
            self.paintings[vid] = Painting(vid, codes, version, extra)

    # --------------------------------------------------------------- geometry
    def mesh_of(self, object_id: int) -> Tuple[Mesh, np.ndarray]:
        """Mesh of an object, following single-component chains; returns (mesh, local transform)."""
        obj = self.objects[object_id]
        trafo = np.eye(4)
        while obj.mesh is None:
            if len(obj.components) != 1:
                raise ValueError(f"object {obj.id} has {len(obj.components)} components; expected a single mesh chain")
            cid, ct = obj.components[0]
            trafo = trafo @ ct
            obj = self.objects[cid]
        return obj.mesh, trafo

    def world_transform(self, object_id: int) -> np.ndarray:
        """Compose build-item and component transforms down to ``object_id`` (first instance found)."""

        def search(oid: int, acc: np.ndarray) -> Optional[np.ndarray]:
            if oid == object_id:
                return acc
            for cid, ct in self.objects[oid].components:
                r = search(cid, acc @ ct)
                if r is not None:
                    return r
            return None

        for oid, it in self.items:
            r = search(oid, it)
            if r is not None:
                return r
        return np.eye(4)

    def world_mesh(self, volume_id: int) -> Mesh:
        """Mesh of a painted volume in world (printer bed) coordinates, in mm."""
        mesh, local = self.mesh_of(volume_id)
        world = self.world_transform(volume_id) @ local
        v = mesh.vertices @ world[:3, :3].T + world[:3, 3]
        return Mesh(v, mesh.triangles)

    # ------------------------------------------------------- virtual extruders
    @property
    def config(self) -> dict:
        return self.project_json["config_containers"][0]

    def virtual_extruders(self) -> List[dict]:
        if self.project_json is None:
            return []
        return self.config.setdefault("virtual_extruders", [])

    def physical_extruder_count(self) -> int:
        conf = self.config.get("configuration", {})
        fil = conf.get("filament_settings", {})
        col = fil.get("filament_colour")
        if isinstance(col, list):
            return len(col)
        pr = conf.get("printer_settings", {})
        nd = pr.get("nozzle_diameter")
        if isinstance(nd, list):
            return len(nd)
        return 1

    def extruder_colours(self) -> List[str]:
        conf = self.config.get("configuration", {})
        col = conf.get("printer_settings", {}).get("extruder_colour")
        fil = conf.get("filament_settings", {}).get("filament_colour", [])
        out = []
        for i in range(self.physical_extruder_count()):
            c = ""
            if isinstance(col, list) and i < len(col):
                c = col[i] or ""
            if not c and i < len(fil):
                c = fil[i] or ""
            out.append(c or "#808080")
        return out

    def layer_height(self) -> float:
        conf = self.config.get("configuration", {})
        return float(conf.get("print_settings", {}).get("layer_height", 0.2))

    # ------------------------------------------------------------------ write
    def write(self, out_path: str, project_modified: bool = False) -> None:
        """Write a copy of the archive with the modified painting (and project JSON if modified)."""
        facets = []
        for vid, p in self.paintings.items():
            entry: Dict[str, object] = {}
            if p.codes:
                entry["mmSegmentationFacets"] = [
                    {"triangle": int(t), "dividing": c} for t, c in sorted(p.codes.items())
                ]
                entry["mmSegmentationFacetsVersion"] = p.version
            entry.update(p.extra)
            if not entry:
                continue
            entry["id"] = vid
            facets.append(entry)
        facets_bytes = json.dumps(facets, separators=(", ", ": ")).encode("utf-8")
        project_bytes = None
        if project_modified and self.project_json is not None:
            project_bytes = json.dumps(self.project_json, indent=2).encode("utf-8")
        with zipfile.ZipFile(self.path) as zin, zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zout:
            names = list(zin.namelist())
            if FACETS_PATH not in names:
                names.append(FACETS_PATH)
            for name in names:
                if name == FACETS_PATH:
                    data = facets_bytes
                elif name == PROJECT_PATH and project_bytes is not None:
                    data = project_bytes
                else:
                    data = zin.read(name)
                zout.writestr(name, data)
