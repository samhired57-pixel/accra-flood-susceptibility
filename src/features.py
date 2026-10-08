"""
Build the modeling dataset for Accra flood susceptibility.

Steps:
 1. Load data/dem.tif, reproject to EPSG:32630 (UTM 30N, metres).
 2. Mask ocean (elevation < 1 m). Snap any hotspot point that falls on water
    to the nearest land cell (logged).
 3. Depression-fill (Planchon & Darboux, vectorized), D8 flow accumulation,
    slope, and Topographic Wetness Index (TWI).
 4. Rasterize OSM waterways/roads -> distance-to-feature rasters (metres).
 5. Positives: each documented hotspot -> 5 points (centroid + 4 jittered
    within 500 m, seed 42). Negatives: random land cells >= 2 km from any
    hotspot and >= 300 m from a waterway.
 6. Save data/samples.csv and derived rasters under data/derived/.

All randomness uses fixed seeds; the run is fully reproducible.
"""
import csv
import os

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

SRC_DIR = os.path.dirname(__file__)
DATA = os.path.join(SRC_DIR, "..", "data")
DERIVED = os.path.join(DATA, "derived")
DST_CRS = "EPSG:32630"
WATER_ELEV_MASK = 1.0   # cells below this (m) treated as ocean/lagoon
SEED = 42


def reproject_dem():
    """Reproject DEM to metric CRS at ~75 m resolution."""
    src_path = os.path.join(DATA, "dem.tif")
    with rasterio.open(src_path) as src:
        transform, w, h = calculate_default_transform(
            src.crs, DST_CRS, src.width, src.height, *src.bounds,
            resolution=75)
        dem = np.empty((h, w), dtype=np.float32)
        reproject(rasterio.band(src, 1), dem,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs=DST_CRS,
                  resampling=Resampling.bilinear)
    print(f"DEM reprojected: {dem.shape}, res=75m")
    return dem, transform


def fill_depressions(dem):
    """Planchon & Darboux (2001) depression filling, vectorized."""
    w = dem.copy()
    eps = 1e-3
    filled_prev = np.inf
    for it in range(2000):
        # min over 8 neighbours of (w + eps)
        padded = np.pad(w, 1, mode="edge")
        neigh_min = np.minimum.reduce([
            padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:],
            padded[1:-1, :-2],                 padded[1:-1, 2:],
            padded[2:, :-2],  padded[2:, 1:-1], padded[2:, 2:],
        ]) + eps
        w_new = np.maximum(dem, neigh_min)
        # borders drain freely
        w_new[0, :] = dem[0, :]; w_new[-1, :] = dem[-1, :]
        w_new[:, 0] = dem[:, 0]; w_new[:, -1] = dem[:, -1]
        change = np.abs(w_new - w).max()
        w = w_new
        if change < 1e-6:
            print(f"  depression fill converged in {it+1} iterations")
            break
    return w


def d8_flow_accumulation(filled):
    """D8 steepest-descent flow accumulation. Returns (acc_cells, slope_rad)."""
    rows, cols = filled.shape
    dy = np.zeros_like(filled)
    # gradient via central differences (cell size 75 m)
    gy, gx = np.gradient(filled, 75.0, 75.0)
    slope = np.arctan(np.hypot(gx, gy))

    # steepest descent neighbour offsets (d8), distances in metres
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    dists = np.array([106.07, 75, 106.07, 75, 75, 106.07, 75, 106.07])
    padded = np.pad(filled, 1, mode="edge")
    best_drop = np.zeros_like(filled)
    best_k = np.zeros(filled.shape, dtype=np.int8)
    for k, (dr, dc) in enumerate(offs):
        drop = (filled - padded[1 + dr:1 + dr + rows, 1 + dc:1 + dc + cols]) / dists[k]
        better = drop > best_drop
        best_drop[better] = drop[better]
        best_k[better] = k + 1  # 0 = no lower neighbour (pit/edge)
    # downstream flat index; -1 for pits/edge cells
    flat = np.arange(rows * cols).reshape(rows, cols)
    down = np.full(rows * cols, -1, dtype=np.int64)
    rr, cc = np.mgrid[0:rows, 0:cols]
    has = best_k > 0
    dr = np.array([o[0] for o in offs])[best_k[has] - 1]
    dc = np.array([o[1] for o in offs])[best_k[has] - 1]
    down[flat[has]] = flat[rr[has] + dr, cc[has] + dc]

    # accumulate in descending elevation order
    order = np.argsort(filled.ravel(), kind="mergesort")[::-1]
    acc = np.ones(rows * cols, dtype=np.float32)
    for i in order:
        d = down[i]
        if d >= 0:
            acc[d] += acc[i]
    return acc.reshape(rows, cols), slope


def rasterize_lines(geojson_path, transform, shape):
    import geopandas as gpd
    gdf = gpd.read_file(geojson_path)
    if len(gdf) == 0:
        return None
    gdf = gdf.to_crs(DST_CRS)
    geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
    if not geoms:
        return None
    mask = rasterize([(g, 1) for g in geoms], out_shape=shape,
                     transform=transform, fill=0, dtype="uint8")
    return mask


def main():
    os.makedirs(DERIVED, exist_ok=True)
    rng = np.random.default_rng(SEED)

    dem, transform = reproject_dem()
    land = dem > WATER_ELEV_MASK
    print(f"land fraction: {land.mean():.3f}")

    filled = fill_depressions(np.where(land, dem, -100.0))  # ocean set low so it drains
    filled = np.where(land, filled, dem)  # keep ocean as-is (excluded later)
    acc, slope = d8_flow_accumulation(np.where(land, filled, -9999.0))
    acc = np.where(land, acc, 0)
    cell_area = 75.0 * 75.0
    with np.errstate(divide="ignore", invalid="ignore"):
        twi = np.log((acc * cell_area + 1.0) / (np.tan(slope) + 1e-3))
    twi = np.where(land, twi, np.nan)
    slope_deg = np.degrees(slope)

    # distance rasters
    print("Rasterizing OSM features...")
    wmask = rasterize_lines(os.path.join(DATA, "waterways.geojson"), transform, dem.shape)
    rmask = rasterize_lines(os.path.join(DATA, "roads.geojson"), transform, dem.shape)
    dist_water = distance_transform_edt(wmask == 0) * 75.0 if wmask is not None else np.full(dem.shape, np.nan)
    dist_road = distance_transform_edt(rmask == 0) * 75.0 if rmask is not None else np.full(dem.shape, np.nan)
    print(f"  waterway cells: {(wmask==1).sum() if wmask is not None else 0}, "
          f"road cells: {(rmask==1).sum() if rmask is not None else 0}")

    # save derived rasters
    meta = dict(driver="GTiff", height=dem.shape[0], width=dem.shape[1], count=1,
                dtype="float32", crs=DST_CRS, transform=transform, compress="deflate")
    for name, arr in [("elevation", dem), ("slope_deg", slope_deg.astype(np.float32)),
                      ("log_flowacc", np.log1p(acc).astype(np.float32)),
                      ("twi", twi.astype(np.float32)),
                      ("dist_waterway_m", dist_water.astype(np.float32)),
                      ("dist_road_m", dist_road.astype(np.float32))]:
        with rasterio.open(os.path.join(DERIVED, f"{name}.tif"), "w", **meta) as dst:
            dst.write(np.nan_to_num(arr, nan=-9999).astype(np.float32), 1)
    print("derived rasters saved")

    # ---- sampling ----
    import geopandas as gpd
    hotspots = []
    with open(os.path.join(DATA, "label_provenance.csv")) as f:
        for row in csv.DictReader(f):
            hotspots.append((row["name"], float(row["lon"]), float(row["lat"])))

    # project hotspot lon/lat to grid
    from rasterio.warp import transform as warp_transform
    xs = [h[1] for h in hotspots]; ys = [h[2] for h in hotspots]
    px, py = warp_transform("EPSG:4326", DST_CRS, xs, ys)

    def rc(x, y):
        c = int((x - transform.c) / transform.a)
        r = int((y - transform.f) / transform.e)
        return r, c

    # snap water points to nearest land
    dist_land = distance_transform_edt(~land)
    rows_, cols_ = np.mgrid[0:dem.shape[0], 0:dem.shape[1]]
    samples = []  # (lon, lat, label, source)
    snapped = []
    for (name, lon, lat), x, y in zip(hotspots, px, py):
        r, c = rc(x, y)
        if not (0 <= r < dem.shape[0] and 0 <= c < dem.shape[1]) or not land[r, c]:
            # nearest land cell within 5 km
            rr = np.clip(r, 0, dem.shape[0]-1); cc = np.clip(c, 0, dem.shape[1]-1)
            d = dist_land
            # search window
            wr = 70  # ~5.25 km
            r0, r1 = max(0, rr-wr), min(dem.shape[0], rr+wr+1)
            c0, c1 = max(0, cc-wr), min(dem.shape[1], cc+wr+1)
            sub = d[r0:r1, c0:c1].copy()
            sub[~land[r0:r1, c0:c1]] = np.inf
            if np.isfinite(sub).any():
                ir, ic = np.unravel_index(np.argmin(sub), sub.shape)
                nr, nc = r0+ir, c0+ic
                shift_m = sub[ir, ic] * 75.0
                snapped.append((name, shift_m))
                r, c = nr, nc
        # centroid + 4 jittered within 500 m
        for j in range(5):
            jr = r + int(rng.integers(-7, 8)); jc = c + int(rng.integers(-7, 8))
            jr = np.clip(jr, 0, dem.shape[0]-1); jc = np.clip(jc, 0, dem.shape[1]-1)
            if not land[jr, jc]:
                jr, jc = r, c
            gx = transform.c + (jc + 0.5) * transform.a
            gy = transform.f + (jr + 0.5) * transform.e
            glon, glat = warp_transform(DST_CRS, "EPSG:4326", [gx], [gy])
            samples.append((glon[0], glat[0], 1, name))
    if snapped:
        print("  snapped to nearest land:", snapped)

    # negatives: land, >=2km from any hotspot, >=300m from waterway
    hs_px = np.array(px); hs_py = np.array(py)
    n_neg_target = 400
    neg = []
    tries = 0
    while len(neg) < n_neg_target and tries < 60000:
        tries += 1
        r = rng.integers(0, dem.shape[0]); c = rng.integers(0, dem.shape[1])
        if not land[r, c] or dem[r, c] < 2.0:
            continue
        if wmask is not None and dist_water[r, c] < 300:
            continue
        x = transform.c + (c + 0.5) * transform.a
        y = transform.f + (r + 0.5) * transform.e
        if np.min(np.hypot(hs_px - x, hs_py - y)) < 2000:
            continue
        glon, glat = warp_transform(DST_CRS, "EPSG:4326", [x], [y])
        neg.append((glon[0], glat[0], 0, "background sample"))
    print(f"  negatives sampled: {len(neg)} (tries={tries})")

    # extract features
    stack = {
        "elevation": dem,
        "slope_deg": slope_deg.astype(np.float32),
        "log_flowacc": np.log1p(acc).astype(np.float32),
        "twi": twi.astype(np.float32),
        "dist_waterway_m": dist_water.astype(np.float32),
        "dist_road_m": dist_road.astype(np.float32),
    }
    with open(os.path.join(DATA, "samples.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lon", "lat", "label", "source"] + list(stack.keys()))
        for lon, lat, label, source in samples + neg:
            x, y = warp_transform("EPSG:4326", DST_CRS, [lon], [lat])
            r, c = rc(x[0], y[0])
            r = np.clip(r, 0, dem.shape[0]-1); c = np.clip(c, 0, dem.shape[1]-1)
            w.writerow([f"{lon:.6f}", f"{lat:.6f}", label, source] +
                       [f"{stack[k][r, c]:.4f}" for k in stack])
    print(f"wrote data/samples.csv: {len(samples)} positives, {len(neg)} negatives")


if __name__ == "__main__":
    main()
