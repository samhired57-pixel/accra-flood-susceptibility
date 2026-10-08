"""
Fetch SRTM elevation for the Greater Accra AOI from Mapzen Terrarium tiles
(https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png).

Terrarium encoding: elevation_m = R*256 + G + B/256 - 32768.

Tiles are mosaicked into data/dem.tif (EPSG:4326, float32).
"""
import io
import math
import os
import time

import numpy as np
import requests
from PIL import Image

BASE = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "dem.tif")

# Area of interest: Greater Accra (lon_min, lat_min, lon_max, lat_max)
BBOX = (-0.55, 5.45, 0.15, 5.85)
ZOOM = 11  # ~75 m/px at this latitude


def deg2num(lon_deg, lat_deg, zoom):
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat_deg)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    return lon_deg, math.degrees(lat_rad)


def fetch_tile(z, x, y, retries=4):
    url = f"{BASE}/{z}/{x}/{y}.png"
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                return np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))
            time.sleep(2 * (attempt + 1))
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch tile {z}/{x}/{y} after {retries} attempts")


def terrarium_to_elevation(rgb):
    r = rgb[:, :, 0].astype(np.float64)
    g = rgb[:, :, 1].astype(np.float64)
    b = rgb[:, :, 2].astype(np.float64)
    return (r * 256.0 + g + b / 256.0 - 32768.0).astype(np.float32)


def main():
    lon_min, lat_min, lon_max, lat_max = BBOX
    x0, y0 = deg2num(lon_min, lat_max, ZOOM)  # NW corner
    x1, y1 = deg2num(lon_max, lat_min, ZOOM)  # SE corner
    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    print(f"Fetching {nx} x {ny} = {nx*ny} tiles at z{ ZOOM }...")

    mosaic = np.zeros((ny * 256, nx * 256), dtype=np.float32)
    for iy in range(ny):
        for ix in range(nx):
            rgb = fetch_tile(ZOOM, x0 + ix, y0 + iy)
            mosaic[iy * 256:(iy + 1) * 256, ix * 256:(ix + 1) * 256] = terrarium_to_elevation(rgb)
            print(f"  tile {ix+1}/{nx},{iy+1}/{ny} done", end="\r")

    # Georeferencing: top-left corner of the mosaic
    lon_tl, lat_tl = num2deg(x0, y0, ZOOM)
    lon_br, _ = num2deg(x1 + 1, y0, ZOOM)
    _, lat_br = num2deg(x0, y1 + 1, ZOOM)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    import rasterio
    from rasterio.transform import from_bounds

    transform = from_bounds(lon_tl, lat_br, lon_br, lat_tl,
                            mosaic.shape[1], mosaic.shape[0])
    with rasterio.open(OUT, "w", driver="GTiff", height=mosaic.shape[0],
                       width=mosaic.shape[1], count=1, dtype="float32",
                       crs="EPSG:4326", transform=transform,
                       compress="deflate") as dst:
        dst.write(mosaic, 1)
    print(f"\nWrote {OUT}  shape={mosaic.shape}  "
          f"elev range=[{np.nanmin(mosaic):.1f}, {np.nanmax(mosaic):.1f}] m")


if __name__ == "__main__":
    main()
