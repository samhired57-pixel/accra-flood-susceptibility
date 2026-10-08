"""
Extract waterways and major roads for the AOI from a local OSM PBF extract.

FALLBACK DOCUMENTATION: the public Overpass API (overpass-api.de) returned
HTTP 403 Forbidden on all attempts from this machine (2026-10-07), even with
a policy-compliant User-Agent, so the query in src/fetch_osm.py could not
run. Instead, the same OpenStreetMap data is read from the Geofabrik Ghana
extract (https://download.geofabrik.de/africa/ghana-latest.osm.pbf,
downloaded 2026-10-07) and filtered locally with pyosmium. Identical tag
selection as the Overpass query.

Reads:  data/ghana-latest.osm.pbf   (not committed; re-download per README)
Writes: data/waterways.geojson, data/roads.geojson
"""
import json
import os

import osmium

BBOX = (-0.55, 5.45, 0.15, 5.85)  # (lon_min, lat_min, lon_max, lat_max)
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
PBF = os.path.join(DATA, "ghana-latest.osm.pbf")

WATERWAY_VALS = {"river", "stream", "canal", "ditch"}
HIGHWAY_VALS = {"motorway", "trunk", "primary", "secondary", "tertiary"}


class WayHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.water = []
        self.roads = []

    def way(self, w):
        if not w.tags:
            return
        try:
            coords = [(n.lon, n.lat) for n in w.nodes]
        except osmium.InvalidLocationError:
            return
        if len(coords) < 2:
            return
        # keep ways with at least one node inside the bbox
        if not any(BBOX[0] <= lon <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]
                   for lon, lat in coords):
            return
        ww = w.tags.get("waterway")
        hw = w.tags.get("highway")
        feat = {"type": "Feature",
                "properties": {"name": w.tags.get("name", "")},
                "geometry": {"type": "LineString", "coordinates": coords}}
        if ww in WATERWAY_VALS:
            feat["properties"]["waterway"] = ww
            self.water.append(feat)
        if hw in HIGHWAY_VALS:
            feat["properties"]["highway"] = hw
            self.roads.append(feat)


def main():
    if not os.path.exists(PBF):
        raise SystemExit(
            f"Missing {PBF}. Download it first:\n"
            "  curl -L -o data/ghana-latest.osm.pbf "
            "https://download.geofabrik.de/africa/ghana-latest.osm.pbf")
    h = WayHandler()
    print("Scanning Ghana OSM extract (this takes a few minutes)...")
    h.apply_file(PBF, locations=True)
    for name, feats in [("waterways", h.water), ("roads", h.roads)]:
        path = os.path.join(DATA, f"{name}.geojson")
        with open(path, "w") as f:
            json.dump({"type": "FeatureCollection", "features": feats}, f)
        print(f"  {name}: {len(feats)} ways -> {path}")


if __name__ == "__main__":
    main()
