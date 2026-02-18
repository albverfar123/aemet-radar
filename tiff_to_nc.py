import numpy as np
import rasterio
import xarray as xr
import ast
from scipy.spatial import cKDTree
from pathlib import Path

DATA_DIR = Path("data")

# zona
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

        tree = cKDTree(paleta)

        rgb_flat = rgb.reshape(-1, 3)
        _, idx = tree.query(rgb_flat)
        data_mm_flat = val_mm[idx]

        yellow_mask = np.array([is_yellow(p) for p in rgb_flat])
        data_mm_flat[yellow_mask] = np.nan

        data_mm = data_mm_flat.reshape(R.shape)

        lon = np.array([
            rasterio.transform.xy(transform, 0, c, offset="center")[0]
            for c in range(R.shape[1])
        ])

        lat = np.array([
            rasterio.transform.xy(transform, r, 0, offset="center")[1]
            for r in range(R.shape[0])
        ])

    ds = xr.Dataset(
        {"precipitation_mm": (("lat", "lon"), data_mm)},
        coords={"lat": lat, "lon": lon},
    )

    ds.to_netcdf(nc_file)

print("Done.")