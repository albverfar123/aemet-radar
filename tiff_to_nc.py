import numpy as np
import rasterio
import xarray as xr
import ast
from scipy.spatial import cKDTree
from rasterio.transform import rowcol, xy
from pathlib import Path
import os
import matplotlib.pyplot as plt

DATA_DIR = Path("data")
PNG_DIR = DATA_DIR / "png"
PNG_DIR.mkdir(exist_ok=True)

# -----------------------
# Zona d'interès
# -----------------------
lon_min, lon_max = 0.0, 3.5
lat_min, lat_max = 40.0, 43.1

# escala visual PNG
vmin = 0
vmax = 40

def is_yellow(rgb_pixel):
    r, g, b = rgb_pixel
    return (r >= 200) & (g >= 180) & (b <= 50)

tiffs = sorted(DATA_DIR.glob("GLD_RNN6H_*.tif"))

for tiff_file in tiffs:

    nc_file = tiff_file.with_suffix(".nc")
    png_file = PNG_DIR / (tiff_file.stem + ".png")

    # si ja existeix tot → saltar
    if nc_file.exists() and png_file.exists():
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

    # =============================
    # NETCDF
    # =============================
    ds = xr.Dataset(
        {"precipitation_mm": (("lat", "lon"), data_mm)},
        coords={"lat": lats, "lon": lons},
    )

    ds.to_netcdf(nc_file)
    print("  💾 NetCDF guardat")

    # =============================
    # PNG TRANSPARENT (NOU)
    # =============================
    try:
        plot_data = data_mm.copy()
        plot_data[plot_data <= 0] = np.nan

        plt.figure(figsize=(6, 5))

        plt.imshow(
            plot_data,
            origin="upper",
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
        )

        plt.axis("off")

        plt.savefig(
            png_file,
            dpi=150,
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
        )

        plt.close()

        print("  🖼️ PNG creat")

    except Exception as e:
        print(f"  ⚠️ Error creant PNG: {e}")

    # =============================
    # ESBORRAR TIFF
    # =============================
    try:
        os.remove(tiff_file)
        print("  🗑️ TIFF esborrat")
    except Exception as e:
        print(f"  ⚠️ No s'ha pogut esborrar el TIFF: {e}")

print("Done.")

