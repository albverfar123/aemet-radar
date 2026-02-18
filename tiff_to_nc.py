import numpy as np
import rasterio
import xarray as xr
import ast
from scipy.spatial import cKDTree
from rasterio.transform import rowcol, xy
from pathlib import Path

DATA_DIR = Path("data")

# -----------------------
# Zona d'interès
# -----------------------
lon_min, lon_max = 0.0, 3.5
lat_min, lat_max = 40.0, 43.5

def is_yellow(rgb_pixel):
    r, g, b = rgb_pixel
    return (r >= 200) & (g >= 180) & (b <= 50)

tiffs = sorted(DATA_DIR.glob("GLD_RNN6H_*.tif"))

for tiff_file in tiffs:

    nc_file = tiff_file.with_suffix(".nc")
    if nc_file.exists():
        continue

    print("Processing", tiff_file.name)

    with rasterio.open(tiff_file) as src:
        R = src.read(1)
        G = src.read(2)
        B = src.read(3)
        rgb = np.stack([R, G, B], axis=-1)
        transform = src.transform

        # -----------------------------------
        # Retall zona d'interès
        # -----------------------------------
        row_min, col_min = rowcol(transform, lon_min, lat_max)
        row_max, col_max = rowcol(transform, lon_max, lat_min)

        r0, r1 = sorted([row_min, row_max])
        c0, c1 = sorted([col_min, col_max])

        rgb_subset = rgb[r0:r1+1, c0:c1+1, :]
        h_sub, w_sub = rgb_subset.shape[:2]

        # -----------------------------------
        # Escala
        # -----------------------------------
        escala_dict = ast.literal_eval(src.tags()["ESCALA"])
        lista_rgba = escala_dict["Lista RGBA"]

        paleta = []
        val_mm = []

        for item in lista_rgba:
            rgba = tuple(int(x) for x in item["RGBA"][:3])
            paleta.append(rgba)

            v = item["Valores"]
            if v[1] == "":
                mm_val = float(v[0])
            else:
                mm_val = (float(v[0]) + float(v[1])) / 2

            val_mm.append(mm_val)

        paleta = np.array(paleta)
        val_mm = np.array(val_mm)

        # -----------------------------------
        # Classificació
        # -----------------------------------
        tree = cKDTree(paleta)
        rgb_flat = rgb_subset.reshape(-1, 3)
        _, idx = tree.query(rgb_flat)

        data_mm_flat = val_mm[idx]

        yellow_mask = np.array([is_yellow(p) for p in rgb_flat])
        data_mm_flat[yellow_mask] = np.nan

        data_mm = data_mm_flat.reshape(h_sub, w_sub).astype(np.float32)

        # -----------------------------------
        # Coordenades
        # -----------------------------------
        lats = np.array([
            xy(transform, r, c0, offset="center")[1]
            for r in range(r0, r1+1)
        ])

        lons = np.array([
            xy(transform, r0, c, offset="center")[0]
            for c in range(c0, c1+1)
        ])

    ds = xr.Dataset(
        {"precipitation_mm": (("lat", "lon"), data_mm)},
        coords={"lat": lats, "lon": lons},
    )

    # -----------------------------------
    # Compressió NetCDF
    # -----------------------------------
    ds.to_netcdf(nc_file)

print("Done.")

