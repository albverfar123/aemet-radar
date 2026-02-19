from pathlib import Path
import xarray as xr
import pandas as pd

DATA_DIR = Path("data")

files = sorted(DATA_DIR.glob("GLD_RNN6H_*.nc"))

if not files:
    raise SystemExit("No files")

groups = {}

for f in files:
    date_str = f.name.split("_")[2]
    day = pd.to_datetime(date_str).date()
    groups.setdefault(day, []).append(f)

for day, flist in groups.items():
    if len(flist) < 4:
        continue

    out_file = DATA_DIR / f"GLD_RNN24H_{day.strftime('%Y%m%d')}.nc"
    txt_file = DATA_DIR / f"GLD_RNN24H_{day.strftime('%Y%m%d')}.txt"

    if out_file.exists():
        continue

    # -----------------------
    # Obrim datasets
    # -----------------------
    sorted_files = sorted(flist)
    ds_list = [xr.open_dataset(f) for f in sorted_files]
    ds_all = xr.concat(ds_list, dim="time")

    valid_count = ds_all["precipitation_mm"].notnull().sum(dim="time")
    daily_sum = ds_all["precipitation_mm"].sum(dim="time", skipna=True)
    daily_sum = daily_sum.where(valid_count > 0)

    out = xr.Dataset({"precipitation_mm_24h": daily_sum})
    out.to_netcdf(out_file)

    # -----------------------
    # ✏️ Guardar TXT traçabilitat
    # -----------------------
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(f"Daily accumulation for {day}\n")
        f.write("Input files used:\n")
        for nc in sorted_files:
            f.write(nc.name + "\n")

    print("Saved", out_file.name)
    print("Saved", txt_file.name)
