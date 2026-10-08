"""
Fetch waterways and major roads for the AOI from OpenStreetMap via the
public Overpass API (https://overpass-api.de).

NOTE (2026-10-07): overpass-api.de returned HTTP 403 Forbidden on every
attempt from the build machine, so this script could not produce data here.
The project instead used src/extract_osm.py against the Geofabrik Ghana
extract (same OSM data, same tag selection). Kept for reproducibility on
networks where Overpass is reachable.

Saves:
  data/waterways.geojson  (waterway=river|stream|canal|ditch)
  data/roads.geojson      (highway=motorway|trunk|primary|secondary|tertiary)

Rate-limit friendly: single query, modest bbox, retries with backoff.
"""
import json
import os
import time

import requests

BBOX = (-0.55, 5.45, 0.15, 5.85)  # (lon_min, lat_min, lon_max, lat_max)
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

QUERY = """
[out:json][timeout:120];
(
  way["waterway"~"^(river|stream|canal|ditch)$"]({s},{w},{n},{e});
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"]({s},{w},{n},{e});
);
out geom;
""".format(s=BBOX[1], w=BBOX[0], n=BBOX[3], e=BBOX[2])


def overpass_query(query, retries=5):
    # Overpass usage policy requires a descriptive User-Agent; the default
    # python-requests UA is blocked (HTTP 403).
    headers = {"User-Agent":
               "accra-flood-susceptibility/1.0 (open research project; "
               "contact samhired57@gmail.com)"}
    for attempt in range(retries):
        try:
            r = requests.post("https://overpass-api.de/api/interpreter",
                              data={"data": query}, timeout=180,
                              headers=headers)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 504):
                wait = 30 * (attempt + 1)
                print(f"  overpass {r.status_code}, waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  attempt {attempt+1} failed: {e}")
            time.sleep(20 * (attempt + 1))
    raise RuntimeError("Overpass query failed after retries")


def ways_to_geojson(elements, tag_key):
    feats = []
    for el in elements:
        if el.get("type") != "way" or "geometry" not in el:
            continue
        tags = el.get("tags", {})
        if tag_key not in tags:
            continue
        coords = [[p["lon"], p["lat"]] for p in el["geometry"]]
        if len(coords) < 2:
            continue
        feats.append({
            "type": "Feature",
            "properties": {tag_key: tags[tag_key],
                           "name": tags.get("name", "")},
            "geometry": {"type": "LineString", "coordinates": coords},
        })
    return {"type": "FeatureCollection", "features": feats}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Querying Overpass API...")
    data = overpass_query(QUERY)
    elements = data.get("elements", [])
    print(f"  got {len(elements)} elements")

    water = ways_to_geojson(elements, "waterway")
    roads = ways_to_geojson(elements, "highway")

    with open(os.path.join(OUT_DIR, "waterways.geojson"), "w") as f:
        json.dump(water, f)
    with open(os.path.join(OUT_DIR, "roads.geojson"), "w") as f:
        json.dump(roads, f)
    print(f"  waterways: {len(water['features'])} ways")
    print(f"  roads:     {len(roads['features'])} ways")


if __name__ == "__main__":
    main()
