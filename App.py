#!/usr/bin/env python3
"""
Revisiegeotransportertool – FastAPI app met geotransporter pipeline
- Upload CSV → verwerk via geotransporter logica
- Export: GeoPackage (3 lagen: polygons, lines, points) en DXF
- Webpreview: polygons + polylines + points (in WGS84)
"""

from __future__ import annotations
import io, json, shutil, tempfile, zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from shapely.geometry import Point, LineString, Polygon, MultiPolygon, mapping, shape
from shapely.ops import transform as shp_transform, unary_union, polygonize
from shapely import wkt

try:
    from pyproj import Transformer
except Exception:
    Transformer = None

try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    import ezdxf
except Exception:
    ezdxf = None

app = FastAPI(title="Revisiegeotransportertool", version="4.1")
DEFAULT_CRS = "EPSG:28992"
WORKDIR = Path(tempfile.gettempdir()) / "revisiegeo"
WORKDIR.mkdir(exist_ok=True)
STATE_FILE = WORKDIR / "state.json"

# ======== Geotransporter Config ========
GEOM_WKT_CANDS = ["wkt", "geometry", "geom"]
X_CANDS = ["x", "east", "e", "easting"]
Y_CANDS = ["y", "north", "n", "northing"]
TYPE_CANDS = ["object_type", "type", "geometry", "geom_type", "feature_type", "shape_type"]
LAYERDESC_CANDS = ["layer_description", "layerdesc", "layer_id", "layer"]
SUBGROUP_CANDS = ["shape_id", "layer_name"]
POINT_ATTR_KEYS = ["layer_name", "layer_description", "layer_color", "attributes_name", "attributes_value"]

# ---------- JSON-safe helpers ----------
import math

def _to_builtin_scalar(v):
    if v is None or isinstance(v, (str, int, float, bool)):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if hasattr(v, "item"):
        try:
            return _to_builtin_scalar(v.item())
        except Exception:
            pass
    if hasattr(v, "isoformat"):
        try:
            return v.isoformat()
        except Exception:
            pass
    return str(v)

def _jsonify(obj):
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return _to_builtin_scalar(obj)

def save_state(d: dict) -> None:
    STATE_FILE.write_text(json.dumps(_jsonify(d), indent=2, ensure_ascii=False))

def load_state() -> dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}

# ======== Geotransporter Helpers ========

def sanitize_columns(df: pd.DataFrame):
    cleaned = []
    l2o = {}
    for c in df.columns:
        cc = c.replace("\ufeff", "").strip()
        cleaned.append(cc)
        l2o[cc.lower()] = cc
    df = df.copy()
    df.columns = cleaned
    return df, l2o

def col_lookup(l2o: Dict[str, str], *cands: List[str]) -> Optional[str]:
    for cand in cands:
        lc = cand.lower()
        if lc in l2o:
            return l2o[lc]
    return None

def normalize_float(v):
    if pd.isna(v):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    s = s.replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None

def is_point_row(row: pd.Series, type_col: Optional[str], wkt_col: Optional[str]) -> bool:
    if wkt_col:
        s = row.get(wkt_col)
        if isinstance(s, str) and s.strip():
            try:
                g = wkt.loads(s)
                if g.geom_type.lower() == "point":
                    return True
            except Exception:
                pass
    if type_col:
        t = str(row.get(type_col, "")).strip().lower()
        if t in {"point", "punt", "pt"}:
            return True
    return False

def should_close_for_export(row: pd.Series, type_col: Optional[str]) -> bool:
    if not type_col:
        return False
    t = str(row.get(type_col, "")).strip().lower()
    if t == "polygon":
        return True
    if t == "polyline":
        return False
    return False

def build_lines_for_export_and_polygonize(df: pd.DataFrame,
                                          xcol: Optional[str],
                                          ycol: Optional[str],
                                          wkt_col: Optional[str],
                                          type_col: Optional[str],
                                          layer_col: str
                                          ) -> Tuple[List[Dict], List[Dict], Dict[str, List[tuple]]]:
    export_lines: List[Dict] = []
    poly_input_lines: List[Dict] = []
    raw_coords_by_ld: Dict[str, List[tuple]] = {}

    def has_val(v):
        return pd.notna(v) and str(v).strip() != ""

    # WKT LINESTRING per rij
    if wkt_col:
        for _, r in df.iterrows():
            s = r.get(wkt_col)
            if not isinstance(s, str) or not s.strip():
                continue
            try:
                geom = wkt.loads(s)
            except Exception:
                continue
            if geom.geom_type.lower() != "linestring":
                continue

            coords = list(geom.coords)
            exp_geom = geom
            if len(coords) >= 2 and should_close_for_export(r, type_col):
                if coords[0] != coords[-1]:
                    coords_closed = coords + [coords[0]]
                    exp_geom = LineString(coords_closed)

            attrs = {k: r[k] for k in df.columns if k != wkt_col}
            attrs["length_m"] = round(exp_geom.length, 3)
            export_lines.append({"geom": exp_geom, "attrs": attrs})

            pcoords = list(geom.coords)
            if len(pcoords) >= 2 and pcoords[0] != pcoords[-1]:
                pcoords = pcoords + [pcoords[0]]
            poly_input_lines.append({"geom": LineString(pcoords), "attrs": attrs})

            ld = r.get(layer_col)
            if has_val(ld):
                raw_coords_by_ld.setdefault(str(ld), []).extend(list(geom.coords))

    # XY lijnen per subgroep
    x_ok, y_ok = xcol is not None, ycol is not None
    if x_ok and y_ok:
        non_point = df[~df.apply(lambda r: is_point_row(r, type_col, wkt_col), axis=1)].copy()
        if not non_point.empty:
            non_point[xcol] = non_point[xcol].apply(normalize_float)
            non_point[ycol] = non_point[ycol].apply(normalize_float)
            non_point = non_point.dropna(subset=[xcol, ycol])

            subcol = None
            for cand in SUBGROUP_CANDS:
                c = col_lookup({k.lower(): k for k in non_point.columns}, cand)
                if c:
                    subcol = c
                    break

            def line_from_df(dfpart: pd.DataFrame) -> Optional[Tuple[LineString, LineString, Dict]]:
                if len(dfpart) < 2:
                    return None
                pts = list(zip(dfpart[xcol].astype(float), dfpart[ycol].astype(float)))
                first_row = dfpart.iloc[0]

                exp_pts = list(pts)
                if should_close_for_export(first_row, type_col):
                    if exp_pts[0] != exp_pts[-1]:
                        exp_pts = exp_pts + [exp_pts[0]]
                exp_geom = LineString(exp_pts)

                poly_pts = list(pts)
                if poly_pts[0] != poly_pts[-1]:
                    poly_pts = poly_pts + [poly_pts[0]]
                poly_geom = LineString(poly_pts)

                attrs = {}
                for c in dfpart.columns:
                    if c in [xcol, ycol]:
                        continue
                    v = dfpart[c].dropna().astype(str)
                    v = v[v.str.strip().ne("").fillna(False)]
                    attrs[c] = v.iloc[0] if not v.empty else None
                attrs["length_m"] = round(exp_geom.length, 3)

                ld = first_row.get(layer_col)
                if has_val(ld):
                    raw_coords_by_ld.setdefault(str(ld), []).extend(pts)

                return exp_geom, poly_geom, attrs

            if subcol:
                for _, sg in non_point.groupby(subcol, sort=False):
                    res = line_from_df(sg)
                    if res:
                        exp_geom, poly_geom, attrs = res
                        export_lines.append({"geom": exp_geom, "attrs": dict(attrs)})
                        poly_input_lines.append({"geom": poly_geom, "attrs": dict(attrs)})
            else:
                res = line_from_df(non_point)
                if res:
                    exp_geom, poly_geom, attrs = res
                    export_lines.append({"geom": exp_geom, "attrs": dict(attrs)})
                    poly_input_lines.append({"geom": poly_geom, "attrs": dict(attrs)})

    return export_lines, poly_input_lines, raw_coords_by_ld

def clean_geom(geom):
    try:
        return geom.buffer(0)
    except Exception:
        return geom

def pick_best_point_attrs(points_df: pd.DataFrame,
                          layerdesc_col: str) -> Dict[str, Dict[str, Optional[str]]]:
    if points_df is None or points_df.empty:
        return {}
    for key in POINT_ATTR_KEYS:
        if key not in points_df.columns:
            points_df[key] = None

    points_df["_ld_str"] = points_df[layerdesc_col].astype(str)
    attr_cols = [c for c in POINT_ATTR_KEYS if c in points_df.columns]

    def score_row(r):
        return sum(1 for c in attr_cols if pd.notna(r.get(c)) and str(r.get(c)).strip() != "")

    best = {}
    for ld, grp in points_df.groupby("_ld_str"):
        grp = grp.copy()
        grp["_score"] = grp.apply(score_row, axis=1)
        grp = grp.sort_values(["_score"], kind="stable")
        r = grp.iloc[-1]
        best[ld] = {c: r.get(c) for c in attr_cols}
    return best

# ======== Geotransporter Pipeline ========

def process_geotransporter(df: pd.DataFrame) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Verwerk CSV via geotransporter logica.
    Retourneert: (gdf_polygons, gdf_lines, gdf_points)
    """
    if gpd is None:
        raise HTTPException(500, "geopandas niet geïnstalleerd")

    df, l2o = sanitize_columns(df)

    wkt_col = col_lookup(l2o, *GEOM_WKT_CANDS)
    xcol = col_lookup(l2o, *X_CANDS)
    ycol = col_lookup(l2o, *Y_CANDS)
    type_col = col_lookup(l2o, *TYPE_CANDS)
    layer_col = col_lookup(l2o, *LAYERDESC_CANDS)

    if layer_col is None:
        raise HTTPException(400, "Kolom 'layer_description' (of equivalent) niet gevonden")
    if (wkt_col is None) and (xcol is None or ycol is None):
        raise HTTPException(400, "Geen geometry-kolommen gevonden (WKT of X/Y)")

    # 1) Lijnen bouwen
    export_lines, poly_input_lines, raw_coords_by_ld = build_lines_for_export_and_polygonize(
        df, xcol, ycol, wkt_col, type_col, layer_col
    )

    # 1b) Groeps-ring per layer_description
    for ld_key, coords in raw_coords_by_ld.items():
        if len(coords) >= 3:
            ring_coords = list(coords)
            if ring_coords[0] != ring_coords[-1]:
                ring_coords = ring_coords + [ring_coords[0]]
            try:
                ring_line = LineString(ring_coords)
                poly_input_lines.append({"geom": ring_line, "attrs": {layer_col: ld_key}})
            except Exception:
                pass

    # 2) Punten splitsen
    def has_val(v):
        return pd.notna(v) and str(v).strip() != ""

    if type_col:
        points_mask = df[type_col].astype(str).str.lower().isin(["point", "punt", "pt"])
    else:
        points_mask = pd.Series([False]*len(df), index=df.index)

    wkt_point_mask = pd.Series([False]*len(df), index=df.index)
    if wkt_col:
        for idx, s in df[wkt_col].items():
            if isinstance(s, str) and s.strip():
                try:
                    g = wkt.loads(s)
                    if g.geom_type.lower() == "point":
                        wkt_point_mask.at[idx] = True
                except Exception:
                    pass

    is_point = points_mask | wkt_point_mask
    points_with_ld = df[is_point & df[layer_col].apply(has_val)].copy()
    points_without_ld = df[is_point & ~df[layer_col].apply(has_val)].copy()

    # 3) Attribuutlookup per layer_description
    best_point_attrs = pick_best_point_attrs(points_with_ld, layer_col)

    # 4) Polygoniseren per layer_description
    lines_by_ld: Dict[str, List[LineString]] = {}
    for ln in poly_input_lines:
        ld = None
        if isinstance(ln.get("attrs"), dict):
            ld = ln["attrs"].get(layer_col)
        if not has_val(ld):
            continue
        key = str(ld)
        lines_by_ld.setdefault(key, []).append(ln["geom"])

    polygon_features: List[Dict] = []
    for ld_key, geoms in lines_by_ld.items():
        if not geoms:
            continue
        try:
            merged = unary_union(geoms)
            polys = list(polygonize(merged))
        except Exception:
            polys = []

        if not polys:
            continue

        unioned = unary_union(polys)
        unioned = clean_geom(unioned)

        attrs = {k: None for k in POINT_ATTR_KEYS}
        if ld_key in best_point_attrs:
            attrs.update(best_point_attrs[ld_key])
        attrs["layer_description"] = ld_key
        polygon_features.append({"geom": unioned, "attrs": attrs})

    # 5) Cutout/hole fase
    AREA_EPS = 0.0001
    if len(polygon_features) > 1:
        polygon_features.sort(key=lambda f: f["geom"].area if f["geom"] else 0.0, reverse=True)
        geoms = [f["geom"] for f in polygon_features]

        processed: List[Dict] = []
        for i, bigf in enumerate(polygon_features):
            big = bigf["geom"]
            if big is None or big.is_empty:
                processed.append(bigf)
                continue

            cutters = []
            for j in range(i + 1, len(geoms)):
                small = geoms[j]
                if small is None or small.is_empty:
                    continue
                if small.area < big.area - AREA_EPS and (big.intersects(small) or big.contains(small)):
                    cutters.append(small)

            if cutters:
                try:
                    cut_union = unary_union(cutters)
                    new_geom = big.difference(cut_union)
                    new_geom = clean_geom(new_geom)
                    if isinstance(new_geom, (Polygon, MultiPolygon)) and not new_geom.is_empty:
                        bigf = {**bigf}
                        bigf["geom"] = new_geom
                except Exception:
                    pass

            processed.append(bigf)

        polygon_features = processed

    # 6) GeoDataFrames maken
    def to_gdf(features: List[Dict], crs: str):
        rows = []
        for f in features:
            r = dict(f.get("attrs", {}))
            r["geometry"] = f["geom"]
            rows.append(r)
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)

    gdf_poly = to_gdf(polygon_features, DEFAULT_CRS) if polygon_features else gpd.GeoDataFrame(geometry=[], crs=DEFAULT_CRS)
    gdf_lines = to_gdf(export_lines, DEFAULT_CRS) if export_lines else gpd.GeoDataFrame(geometry=[], crs=DEFAULT_CRS)

    # Area/length
    if not gdf_poly.empty:
        gdf_poly["area_m2"] = gdf_poly.geometry.area.round(3)
    if not gdf_lines.empty:
        if "length_m" not in gdf_lines.columns:
            gdf_lines["length_m"] = gdf_lines.geometry.length.round(3)

    # 7) Losse punten (zonder layer_description)
    point_features = []
    geom_cols = {c for c in [wkt_col, xcol, ycol] if c}
    if not points_without_ld.empty:
        if wkt_col:
            for _, r in points_without_ld.iterrows():
                s = r.get(wkt_col)
                if isinstance(s, str) and s.strip():
                    try:
                        g = wkt.loads(s)
                        if g.geom_type.lower() == "point":
                            attrs = {k: r[k] for k in points_without_ld.columns if k not in geom_cols}
                            point_features.append({"geom": g, "attrs": attrs})
                    except Exception:
                        pass
        if xcol and ycol:
            pts = points_without_ld.copy()
            pts[xcol] = pts[xcol].apply(normalize_float)
            pts[ycol] = pts[ycol].apply(normalize_float)
            pts = pts.dropna(subset=[xcol, ycol])
            for _, r in pts.iterrows():
                g = Point(float(r[xcol]), float(r[ycol]))
                attrs = {k: r[k] for k in points_without_ld.columns if k not in geom_cols}
                point_features.append({"geom": g, "attrs": attrs})

    gdf_points = to_gdf(point_features, DEFAULT_CRS) if point_features else gpd.GeoDataFrame(geometry=[], crs=DEFAULT_CRS)

    return gdf_poly, gdf_lines, gdf_points

# ---------------- HTML UI ----------------
INDEX_HTML = """<!doctype html><html lang='nl'><head><meta charset='utf-8'>
<title>Revisiegeotransportertool</title>
<link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'/>
<script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>
<style>
body{font-family:system-ui,Segoe UI,Arial;margin:0}
.page{margin:1rem;padding:1rem}
.map{height:70vh;border:2px solid #0e8;border-radius:12px;margin-top:1rem}
.btn{padding:0.6rem 1.2rem;margin:0.3rem;border:1px solid #0aa;border-radius:8px;background:#eef;cursor:pointer}
.btn:disabled{opacity:0.5}
.legend{margin-top:.5rem;font-size:.9rem;color:#234}
.badge{display:inline-block;padding:.1rem .4rem;border-radius:.5rem;border:1px solid #89a;background:#f5faff;margin-left:.3rem}
.manual{margin-top:1rem;font-size:.9rem;line-height:1.4;color:#123;background:#f7fafc;border-radius:10px;padding:0.8rem;border:1px solid #c5d6e5}
.manual h3{margin-top:0;font-size:1rem}
.manual b{color:#102a43}
</style></head><body>
<div class='page'>
<h2>Revisiegeotransportertool</h2>
<input id='file' type='file' accept='.csv' style='display:none'>
<button class='btn' onclick="document.getElementById('file').click()">Upload CSV</button>
<button id='btn-gpkg' class='btn' disabled>Export GeoPackage</button>
<button id='btn-dxf' class='btn' disabled>Export Lijntekening-DXF</button>
<button id='btn-bgt' class='btn' disabled>Export BGT-DXF</button>
<button id='btn-manual' class='btn'>Handleiding opleveringen</button>
<div id='status'>Geen data geladen</div>
<div class='legend'>
  <span class='badge'>Polygonen: blauw</span>
  <span class='badge'>Lijnen: paars</span>
  <span class='badge'>Punten: rood</span>
</div>

<div id='manual' class='manual' style='display:none'>
  <h3>Handleiding opleverformaten</h3>
  <p><b>-> NLCS-lijntekening</b><br>
  • Gebruik <i>Export Lijntekening-DXF</i> als basislijntekening.<br>
  • Explodeer waar nodig in AutoCAD/InfraCAD en zet alle lijnen/blocks om naar NLCS-lagen.<br>
  • Werk legenda en layout uit op het standaard Kepa-kader, controleer maatvoering en plot.</p>

  <p><b>-> OTL-database</b><br>
  • Exporteer eerst de GIS-vlakken via <i>Export GeoPackage</i> (GPKG).<br>
  • Open <code>&lt;Projectnaam&gt;_GEODATA_d5.x.gpkg</code> in QGIS en controleer vlakken/topologie.<br>
  • Zet vlakken over naar de juiste OTL-lagen en vul de vereiste attributen per laag in.<br>
  • Bundel de lagen tot: <code>&lt;Projectnaam&gt;_&lt;OTL-view&gt;_d5.x.zip</code> en genereer<br>
    eventueel <code>&lt;Projectnaam&gt;_STUFGEO_d5.x.gfs</code> voor oplevering.</p>

  <p><b>-> Hoeveelheden uit GIS</b><br>
  • Gebruik de GIS-vlakken (GPKG) om oppervlaktes en lengtes per materiaal te bepalen.<br>
  • Exporteer de attributentabel naar Excel en maak hieruit de hoeveelhedenstaat.</p>

  <p><b>-> BGT-verwerking</b><br>
  • Maak via <i>Export BGT-DXF</i> een BGT-geschikte DXF-export uit de API.<br>
  • Open deze DXF in GISKIT en koppel de juiste kenpunten aan de centroïden.<br>
  • Pas de vlakken in op de actuele BGT-kaart en corrigeer overlap/open grenzen.<br>
  • Lever STUFGEO aan de gemeente volgens de BGT-richtlijnen.</p>
</div>

<div id='map' class='map'></div></div>
<script>
let map=L.map('map').setView([52.1,5.3],8);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'© OSM'}).addTo(map);

let layerPolys=L.geoJSON(null,{style:f=>({color:'#0066ff',weight:2,opacity:.9,fillOpacity:.1})}).addTo(map);
let layerLines=L.geoJSON(null,{style:f=>({color:'#7a00c2',weight:2,opacity:.9})}).addTo(map);
let layerPts=L.geoJSON(null,{pointToLayer:(f,ll)=>L.circleMarker(ll,{radius:4,color:'#d00'})}).addTo(map);

async function refresh(){
  let r=await fetch('/api/preview'); if(!r.ok){return}
  let g=await r.json();
  layerPolys.clearLayers(); layerLines.clearLayers(); layerPts.clearLayers();
  if(g.polygons) layerPolys.addData(g.polygons);
  if(g.polylines) layerLines.addData(g.polylines);
  if(g.points) layerPts.addData(g.points);
  let fg=L.featureGroup([layerPolys,layerLines,layerPts]);
  if(fg.getLayers().length){ map.fitBounds(fg.getBounds().pad(0.2)); }
  document.getElementById('btn-gpkg').disabled=false;
  document.getElementById('btn-dxf').disabled=false;
  document.getElementById('btn-bgt').disabled=false;
  document.getElementById('status').innerText='Data geladen ('+(g.stats||'')+')';
}

file.onchange=async e=>{
  let f=e.target.files[0]; if(!f) return;
  let fd=new FormData(); fd.append('file',f);
  let r=await fetch('/api/upload',{method:'POST',body:fd});
  if(r.ok) refresh(); else alert(await r.text());
};

document.getElementById('btn-gpkg').onclick=()=>window.location='/api/export/gpkg';
document.getElementById('btn-dxf').onclick=()=>window.location='/api/export/dxf';
document.getElementById('btn-bgt').onclick=()=>window.location='/api/export/bgt';
document.getElementById('btn-manual').onclick=()=>{
  const el=document.getElementById('manual');
  el.style.display = (el.style.display==='none' || !el.style.display) ? 'block' : 'none';
};
</script></body></html>"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return INDEX_HTML

# ======== CSV lezen ========

def read_csv_any(c: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(c), dtype=str)
        if df.shape[1] == 1:
            raise ValueError
        return df
    except Exception:
        return pd.read_csv(io.BytesIO(c), sep=";", dtype=str)

# ======== API ========

@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Upload een CSV")
    c = await file.read()
    try:
        df = read_csv_any(c)
    except Exception as e:
        raise HTTPException(400, f"CSV kon niet worden gelezen: {e}")

    try:
        gdf_poly, gdf_lines, gdf_points = process_geotransporter(df)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Fout bij verwerken: {e}")

    # Opslaan in state
    state = {
        "crs": DEFAULT_CRS,
        "polygons": [mapping(g) for g in gdf_poly.geometry] if not gdf_poly.empty else [],
        "polylines": [mapping(g) for g in gdf_lines.geometry] if not gdf_lines.empty else [],
        "points": [mapping(g) for g in gdf_points.geometry] if not gdf_points.empty else [],
        "poly_props": gdf_poly.drop(columns=["geometry"]).to_dict("records") if not gdf_poly.empty else [],
        "line_props": gdf_lines.drop(columns=["geometry"]).to_dict("records") if not gdf_lines.empty else [],
        "point_props": gdf_points.drop(columns=["geometry"]).to_dict("records") if not gdf_points.empty else [],
    }

    # ---- Extra: ook punten mét layer_description bewaren (alleen voor DXF/BGT) ----
    try:
        df_up, l2o_up = sanitize_columns(df)
        wkt_col_up = col_lookup(l2o_up, *GEOM_WKT_CANDS)
        xcol_up = col_lookup(l2o_up, *X_CANDS)
        ycol_up = col_lookup(l2o_up, *Y_CANDS)
        type_col_up = col_lookup(l2o_up, *TYPE_CANDS)
        layer_col_up = col_lookup(l2o_up, *LAYERDESC_CANDS)

        def _has_val(v): return pd.notna(v) and str(v).strip() != ""

        if type_col_up:
            mask_type = df_up[type_col_up].astype(str).str.lower().isin(["point", "punt", "pt"])
        else:
            mask_type = pd.Series([False]*len(df_up), index=df_up.index)

        mask_wkt_point = pd.Series([False]*len(df_up), index=df_up.index)
        if wkt_col_up:
            for idx, sraw in df_up[wkt_col_up].items():
                if isinstance(sraw, str) and sraw.strip():
                    try:
                        if wkt.loads(sraw).geom_type.lower() == "point":
                            mask_wkt_point.at[idx] = True
                    except Exception:
                        pass

        is_point_up = mask_type | mask_wkt_point
        pts_with_ld = df_up[is_point_up & df_up[layer_col_up].apply(_has_val)].copy() if layer_col_up else pd.DataFrame()

        point_ld_features = []
        geom_cols_up = {c for c in [wkt_col_up, xcol_up, ycol_up] if c}

        if not pts_with_ld.empty:
            if wkt_col_up:
                for _, r in pts_with_ld.iterrows():
                    sraw = r.get(wkt_col_up)
                    if isinstance(sraw, str) and sraw.strip():
                        try:
                            g = wkt.loads(sraw)
                            if g.geom_type.lower() == "point":
                                attrs = {k: r[k] for k in pts_with_ld.columns if k not in geom_cols_up}
                                point_ld_features.append({"geom": g, "attrs": attrs})
                        except Exception:
                            pass
            if xcol_up and ycol_up:
                pts_xy = pts_with_ld.copy()
                pts_xy[xcol_up] = pts_xy[xcol_up].apply(normalize_float)
                pts_xy[ycol_up] = pts_xy[ycol_up].apply(normalize_float)
                pts_xy = pts_xy.dropna(subset=[xcol_up, ycol_up])
                for _, r in pts_xy.iterrows():
                    g = Point(float(r[xcol_up]), float(r[ycol_up]))
                    attrs = {k: r[k] for k in pts_with_ld.columns if k not in geom_cols_up}
                    point_ld_features.append({"geom": g, "attrs": attrs})

        state["points_ld"] = [mapping(f["geom"]) for f in point_ld_features]
        state["point_ld_props"] = [{k: _to_builtin_scalar(v) for k, v in f["attrs"].items()} for f in point_ld_features]
    except Exception:
        state["points_ld"] = []
        state["point_ld_props"] = []

    save_state(state)
    return {
        "polygons": len(gdf_poly),
        "polylines": len(gdf_lines),
        "points": len(gdf_points),
        "crs": DEFAULT_CRS,
    }

# RD → WGS84 (voor preview)
def _ensure_preview_transformer():
    if Transformer is None:
        raise HTTPException(500, "pyproj niet geïnstalleerd (nodig voor preview)")
    return Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)

def _geomdict_rd_to_wgs84(geom_dict: dict, transformer: "Transformer") -> dict:
    g = shape(geom_dict)
    g4326 = shp_transform(transformer.transform, g)
    return mapping(g4326)

@app.get("/api/preview")
async def preview():
    s = load_state()
    if not s:
        raise HTTPException(404, "Geen data")

    tr = _ensure_preview_transformer()
    polys_geom = [_geomdict_rd_to_wgs84(g, tr) for g in s.get("polygons", [])]
    lines_geom = [_geomdict_rd_to_wgs84(g, tr) for g in s.get("polylines", [])]
    points_geom = [_geomdict_rd_to_wgs84(g, tr) for g in s.get("points", [])]

    polys = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": g, "properties": (s["poly_props"][i] if i < len(s.get("poly_props", [])) else {})}
            for i, g in enumerate(polys_geom)
        ],
    }
    lines = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": g, "properties": (s["line_props"][i] if i < len(s.get("line_props", [])) else {})}
            for i, g in enumerate(lines_geom)
        ],
    }
    points = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": g, "properties": (s["point_props"][i] if i < len(s.get("point_props", [])) else {})}
            for i, g in enumerate(points_geom)
        ],
    }
    stats = f"{len(polys['features'])} polys · {len(lines['features'])} lines · {len(points['features'])} points"
    return JSONResponse({"crs": "EPSG:4326", "polygons": polys, "polylines": lines, "points": points, "stats": stats})

@app.get("/api/export/gpkg")
async def export_gpkg():
    if gpd is None:
        raise HTTPException(500, "Geopandas niet geïnstalleerd")
    s = load_state()
    if not s:
        raise HTTPException(404, "Geen data om te exporteren")
    crs = s.get("crs", DEFAULT_CRS)

    # GeoDataFrames reconstrueren
    gdf_poly = gpd.GeoDataFrame(
        s.get("poly_props", []),
        geometry=[shape(g) for g in s.get("polygons", [])],
        crs=crs
    ) if s.get("polygons") else gpd.GeoDataFrame(geometry=[], crs=crs)

    gdf_lines = gpd.GeoDataFrame(
        s.get("line_props", []),
        geometry=[shape(g) for g in s.get("polylines", [])],
        crs=crs
    ) if s.get("polylines") else gpd.GeoDataFrame(geometry=[], crs=crs)

    gdf_points = gpd.GeoDataFrame(
        s.get("point_props", []),
        geometry=[shape(g) for g in s.get("points", [])],
        crs=crs
    ) if s.get("points") else gpd.GeoDataFrame(geometry=[], crs=crs)

    # GeoPackage schrijven
    gpkg_path = WORKDIR / "export.gpkg"
    if gpkg_path.exists():
        gpkg_path.unlink()

    if not gdf_poly.empty:
        gdf_poly.to_file(gpkg_path, layer="polygons", driver="GPKG")
    if not gdf_lines.empty:
        gdf_lines.to_file(gpkg_path, layer="lines", driver="GPKG", mode="a")
    if not gdf_points.empty:
        gdf_points.to_file(gpkg_path, layer="points", driver="GPKG", mode="a")

    return FileResponse(gpkg_path, media_type="application/geopackage+sqlite3", filename="export.gpkg")

# ================== DXF EXPORT (alleen CSV-lijnen, POINTs; geen laag 0) ==================
@app.get("/api/export/dxf")
async def export_dxf():
    if ezdxf is None:
        raise HTTPException(500, "ezdxf niet geïnstalleerd")
    s = load_state()
    if not s:
        raise HTTPException(404, "Geen data om te exporteren")

    from shapely.geometry import LineString, Point
    from shapely.geometry import shape as shp_shape

    # ---------- helpers ----------
    def rgb_to_int(rgb):
        r, g, b = rgb
        return (int(r) << 16) + (int(g) << 8) + int(b)

    def parse_color(v):
        if v is None:
            return None, None
        sv = str(v).strip()
        try:
            aci = int(sv)
            if 0 <= aci <= 255:
                return aci, None
        except Exception:
            pass
        if sv.startswith("#") and len(sv) == 7:
            try:
                r = int(sv[1:3], 16); g = int(sv[3:5], 16); b = int(sv[5:7], 16)
                return None, (r, g, b)
            except Exception:
                return None, None
        if "," in sv:
            try:
                parts = [int(p) for p in sv.split(",")]
                if len(parts) == 3 and all(0 <= p <= 255 for p in parts):
                    return None, (parts[0], parts[1], parts[2])
            except Exception:
                return None, None
        return None, None

    def is_blocked_layer(name: Optional[str]) -> bool:
        if name is None:
            return True
        n = str(name).strip()
        return (n == "") or (n == "0")

    def ensure_layer(doc, name, color_hint):
        if is_blocked_layer(name):
            return None
        if name in doc.layers:
            return name
        aci, _ = parse_color(color_hint)
        if aci is not None:
            doc.layers.add(name, dxfattribs={"color": aci})
        else:
            doc.layers.add(name)
        return name

    def ent_attrs(layer_name, color_value):
        d = {}
        if not is_blocked_layer(layer_name):
            d["layer"] = layer_name
        aci, rgb = parse_color(color_value)
        if aci is not None:
            d["color"] = aci
        if rgb is not None:
            d["true_color"] = rgb_to_int(rgb)
        return d

    def as_xy(seq):
        return [(float(x), float(y)) for (x, y) in seq]

    def build_label_text(props, extra=None):
        ln = props.get("layer_name")
        ld = props.get("layer_description")
        an = props.get("attributes_name")
        av = props.get("attributes_value")
        lines = []
        if ln: lines.append(f"layer: {ln}")
        if ld: lines.append(f"desc: {ld}")
        if an or av: lines.append(f"{an or 'attr'}: {av if av is not None else ''}")
        if extra and (extra.get("length_m") is not None):
            lines.append(f"len: {extra['length_m']} m")
        return "\\P".join(lines) if lines else "obj"

    def add_label(msp, layer, x, y, text, height=0.25):
        mtx = msp.add_mtext(text, dxfattribs={"layer": layer})
        mtx.set_location((float(x), float(y)))
        mtx.dxf.char_height = float(height)

    def line_key(coords, tol=4):
        pts = [(round(float(x), tol), round(float(y), tol)) for x, y in coords]
        if not pts:
            return None
        a = tuple(pts); b = tuple(reversed(pts))
        return a if a <= b else b

    # ---------- DXF document ----------
    doc = ezdxf.new("R2010")
    doc.units = ezdxf.units.M
    # zichtbare POINT-stijl
    doc.header["$PDMODE"] = 34
    doc.header["$PDSIZE"] = 0.25
    msp = doc.modelspace()

    LABEL_LAYER = "LABELS"
    if LABEL_LAYER not in doc.layers:
        doc.layers.add(LABEL_LAYER, dxfattribs={"color": 7})

    seen_lines = set()

    # (1) Alleen CSV-lijnen (geen polygon-randen)
    for geom, props in zip(s.get("polylines", []), s.get("line_props", [])):
        g = shp_shape(geom)
        if not isinstance(g, LineString):
            continue
        lname = props.get("layer_name")
        if is_blocked_layer(lname):
            continue
        lcolor = props.get("layer_color")
        ensure_layer(doc, lname, lcolor)

        k = line_key(g.coords)
        if k in seen_lines:
            continue
        seen_lines.add(k)

        msp.add_lwpolyline(as_xy(g.coords), close=False, dxfattribs=ent_attrs(lname, lcolor))
        mid = g.interpolate(0.5, normalized=True)
        add_label(msp, LABEL_LAYER, mid.x, mid.y, build_label_text(props, {"length_m": props.get("length_m")}))

    # (2) Punten zonder layer_description → DXF POINT
    for geom, props in zip(s.get("points", []), s.get("point_props", [])):
        g = shp_shape(geom)
        if not isinstance(g, Point):
            continue
        lname = props.get("layer_name")
        if is_blocked_layer(lname):
            continue
        lcolor = props.get("layer_color")
        ensure_layer(doc, lname, lcolor)
        msp.add_point((float(g.x), float(g.y)), dxfattribs=ent_attrs(lname, lcolor))
        add_label(msp, LABEL_LAYER, g.x + 0.25, g.y + 0.25, build_label_text(props))

    # (3) Punten mét layer_description → DXF POINT
    for geom, props in zip(s.get("points_ld", []) or [], s.get("point_ld_props", []) or []):
        g = shp_shape(geom)
        if not isinstance(g, Point):
            continue
        lname = props.get("layer_name")
        if is_blocked_layer(lname):
            continue
        lcolor = props.get("layer_color")
        ensure_layer(doc, lname, lcolor)
        msp.add_point((float(g.x), float(g.y)), dxfattribs=ent_attrs(lname, lcolor))
        add_label(msp, LABEL_LAYER, g.x + 0.25, g.y + 0.25, build_label_text(props))

    dxf_path = WORKDIR / "export.dxf"
    doc.saveas(dxf_path)
    return FileResponse(dxf_path, media_type="application/dxf", filename="export.dxf")

# ================== BGT DXF EXPORT (samengevoegde randen + LD-points) ==================
from shapely.ops import linemerge

from shapely.ops import linemerge

from shapely.ops import linemerge

@app.get("/api/export/bgt")
async def export_bgt():
    """
    BGT-export:
    - Alleen polygon-randen van polygonen met NIET-lege 'layer_description'
    - Randlijnen samengevoegd (linemerge) en gededupliceerd
    - Punten mét layer_description als DXF POINT met eigen laag/kleur + label (zoals DXF-export)
    - Geen laag '0'
    """
    if ezdxf is None:
        raise HTTPException(500, "ezdxf niet geïnstalleerd")
    s = load_state()
    if not s:
        raise HTTPException(404, "Geen data om te exporteren")

    from shapely.geometry import Polygon, MultiPolygon, LineString, Point
    from shapely.geometry import shape as shp_shape

    # ---------- helpers (lokaal) ----------
    def _has_val(v) -> bool:
        return v is not None and str(v).strip() != ""

    def is_blocked_layer(name: Optional[str]) -> bool:
        if name is None:
            return True
        n = str(name).strip()
        return (n == "") or (n == "0")

    def as_xy(seq):
        return [(float(x), float(y)) for (x, y) in seq]

    def rgb_to_int(rgb):
        r, g, b = rgb
        return (int(r) << 16) + (int(g) << 8) + int(b)

    def parse_color(v):
        if v is None:
            return None, None
        sv = str(v).strip()
        try:
            aci = int(sv)
            if 0 <= aci <= 255:
                return aci, None
        except Exception:
            pass
        if sv.startswith("#") and len(sv) == 7:
            try:
                r = int(sv[1:3], 16); g = int(sv[3:5], 16); b = int(sv[5:7], 16)
                return None, (r, g, b)
            except Exception:
                return None, None
        if "," in sv:
            try:
                parts = [int(p) for p in sv.split(",")]
                if len(parts) == 3 and all(0 <= p <= 255 for p in parts):
                    return None, (parts[0], parts[1], parts[2])
            except Exception:
                return None, None
        return None, None

    def ensure_layer(doc, name, color_hint=None):
        if is_blocked_layer(name):
            return None
        if name in doc.layers:
            return name
        aci, _ = parse_color(color_hint)
        if aci is not None:
            doc.layers.add(name, dxfattribs={"color": aci})
        else:
            doc.layers.add(name)
        return name

    def ent_attrs(layer_name, color_value):
        d = {}
        if not is_blocked_layer(layer_name):
            d["layer"] = layer_name
        aci, rgb = parse_color(color_value)
        if aci is not None:
            d["color"] = aci
        if rgb is not None:
            d["true_color"] = rgb_to_int(rgb)
        return d

    def build_label_text(props, extra=None):
        ln = props.get("layer_name")
        ld = props.get("layer_description")
        an = props.get("attributes_name")
        av = props.get("attributes_value")
        lines = []
        if ln: lines.append(f"layer: {ln}")
        if ld: lines.append(f"desc: {ld}")
        if an or av: lines.append(f"{an or 'attr'}: {av if av is not None else ''}")
        if extra and (extra.get("length_m") is not None):
            lines.append(f"len: {extra['length_m']} m")
        return "\\P".join(lines) if lines else "obj"

    def add_label(msp, layer, x, y, text, height=0.25):
        mtx = msp.add_mtext(text, dxfattribs={"layer": layer})
        mtx.set_location((float(x), float(y)))
        mtx.dxf.char_height = float(height)

    def line_key(coords, tol=4):
        """Richtings-onafhankelijke key met afronding (default 0.0001 m)."""
        pts = [(round(float(x), tol), round(float(y), tol)) for x, y in coords]
        if not pts:
            return None
        a = tuple(pts); b = tuple(reversed(pts))
        return a if a <= b else b

    # ---------- verzamel & merge polygon-randen MET layer_description ----------
    edge_lines: List[LineString] = []
    polys = s.get("polygons", []) or []
    poly_props = s.get("poly_props", []) or []
    for geom, props in zip(polys, poly_props):
        # alleen polygonen waarvan 'layer_description' bestaat en niet-blank is
        ld = props.get("layer_description")
        if not _has_val(ld):
            continue
        g = shp_shape(geom)
        if isinstance(g, Polygon):
            edge_lines.append(LineString(list(g.exterior.coords)))
            for hole in g.interiors:
                edge_lines.append(LineString(list(hole.coords)))
        elif isinstance(g, MultiPolygon):
            for pg in g.geoms:
                edge_lines.append(LineString(list(pg.exterior.coords)))
                for hole in pg.interiors:
                    edge_lines.append(LineString(list(hole.coords)))

    merged_lines: List[LineString] = []
    if edge_lines:
        try:
            u = unary_union(edge_lines)
            lm = linemerge(u)
            if isinstance(lm, LineString):
                merged_lines = [lm]
            else:
                merged_lines = list(lm.geoms)
        except Exception:
            merged_lines = edge_lines  # fallback

    # Dedup
    seen = set()
    unique_lines = []
    for ls in merged_lines:
        if not isinstance(ls, LineString):
            continue
        k = line_key(ls.coords)
        if k and k not in seen:
            seen.add(k)
            unique_lines.append(ls)

    # ---------- DXF document ----------
    doc = ezdxf.new("R2010")
    doc.units = ezdxf.units.M
    # zichtbare POINT-stijl
    doc.header["$PDMODE"] = 34
    doc.header["$PDSIZE"] = 0.25
    msp = doc.modelspace()

    # Lijnen op één BGT-laag
    BGT_LINE_LAYER = "BGT_BOUNDARIES"
    if BGT_LINE_LAYER not in doc.layers:
        doc.layers.add(BGT_LINE_LAYER, dxfattribs={"color": 7})

    for ls in unique_lines:
        msp.add_lwpolyline(as_xy(ls.coords), close=False, dxfattribs={"layer": BGT_LINE_LAYER})

    # Labels-laag (voor puntlabels)
    LABEL_LAYER = "LABELS"
    if LABEL_LAYER not in doc.layers:
        doc.layers.add(LABEL_LAYER, dxfattribs={"color": 7})

    # Punten mét layer_description: zoals DXF-export (eigen laag/kleur + label)
    for geom, props in zip(s.get("points_ld", []) or [], s.get("point_ld_props", []) or []):
        g = shp_shape(geom)
        if not isinstance(g, Point):
            continue
        lname = props.get("layer_name")
        if is_blocked_layer(lname):
            continue
        lcolor = props.get("layer_color")
        ensure_layer(doc, lname, lcolor)

        msp.add_point((float(g.x), float(g.y)), dxfattribs=ent_attrs(lname, lcolor))
        add_label(msp, LABEL_LAYER, g.x + 0.25, g.y + 0.25, build_label_text(props))

    dxf_path = WORKDIR / "export_bgt.dxf"
    doc.saveas(dxf_path)
    return FileResponse(dxf_path, media_type="application/dxf", filename="export_bgt.dxf")

if __name__ == "__main__":
    import uvicorn
    # host="0.0.0.0" maakt de app toegankelijk op het lokale netwerk
    uvicorn.run(app, host="0.0.0.0", port=8000)
