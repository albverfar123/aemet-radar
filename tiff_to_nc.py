import numpy as np
import rasterio
import xarray as xr
import ast
from scipy.spatial import cKDTree
from rasterio.transform import rowcol, xy
from pathlib import Path
import os

# 🔥 IMPORTANT per GitHub Actions (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path("data")
PNG_DIR = Path("png")
PNG_DIR.mkdir(exist_ok=True)

# -----------------------
# Zona d'interès
# -----------------------
lon_min, lon_max = 0.0, 3.5
lat_min, lat_max = 40.0, 43.1

def is_yellow(rgb_pixel):
    r, g, b = rgb_pixel
    return (r >= 200) & (g >= 180) & (b <= 50)

# -----------------------
# Buscar TIFFs
# -----------------------
tiffs = list(DATA_DIR.glob("GLD_RNN6H_*.tif"))
print(f"🔎 TIFF trobats: {len(tiffs)}")

for tiff_file in tiffs:

    print("----")
    print("📂 Processing", tiff_file.name)

    nc_file = tiff_file.with_suffix(".nc")
    png_file = PNG_DIR / (tiff_file.stem + ".png")

    print("   NC existeix:", nc_file.exists())
    print("   PNG existeix:", png_file.exists())

    # si tot ja existeix → saltar
    if nc_file.exists() and png_file.exists():
        print("   ⏭️ Saltant (ja processat)")
        continue

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

    # =========================================================
    # 💾 NETCDF
    # =========================================================
    if not nc_file.exists():
        print("   💾 Guardant NetCDF...")
        ds = xr.Dataset(
            {"precipitation_mm": (("lat", "lon"), data_mm)},
            coords={"lat": lats, "lon": lons},
        )
        ds.to_netcdf(nc_file)

    # =========================================================
    # 🖼️ PNG TRANSPARENT
    # =========================================================
    if not png_file.exists():
        print("   🎨 Generant PNG...")

        data_plot = data_mm.copy()
        data_plot[data_plot <= 0] = np.nan

        plt.figure(figsize=(6, 6))
        cmap = plt.cm.Blues.copy()
        cmap.set_bad(alpha=0)

        im = plt.imshow(
            data_plot,
            origin="upper",
            cmap=cmap,
            vmin=0,
            vmax=50  # ⚠️ pots ajustar
        )

        plt.colorbar(im, label="Precipitació (mm)")
        plt.axis("off")
        plt.tight_layout()

        plt.savefig(png_file, dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close()

        print(f"   ✅ PNG guardat: {png_file}")

    # =========================================================
    # 🗑️ BORRAR TIFF
    # =========================================================
    try:
        os.remove(tiff_file)
        print("   🗑️ TIFF esborrat")
    except Exception as e:
        print(f"   ⚠️ No s'ha pogut esborrar el TIFF: {e}")

print("🏁 Procés completat")



