from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

DATA_DIR = Path("data")
PNG_DIR = Path("png")
PNG_DIR.mkdir(exist_ok=True)

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
    png_file = PNG_DIR / f"GLD_RNN24H_{day.strftime('%Y%m%d')}.png"

    if out_file.exists():
        continue

    print("----")
    print(f"📊 Calculant acumulat {day}")

    # -----------------------
    # Obrim datasets
    # -----------------------
    sorted_files = sorted(flist)

    ds_list = []
    for f in sorted_files:
        ds_list.append(xr.open_dataset(f))

    ds_all = xr.concat(ds_list, dim="time")

    valid_count = ds_all["precipitation_mm"].notnull().sum(dim="time")
    daily_sum = ds_all["precipitation_mm"].sum(dim="time", skipna=True)
    daily_sum = daily_sum.where(valid_count > 0)

    out = xr.Dataset({"precipitation_mm_24h": daily_sum})
    out.to_netcdf(out_file)

    # =========================
    # 🖼️ GENERAR PNG
    # =========================
    print("   🎨 Generant PNG...")

    data = daily_sum.values.astype(float)
    data[data <= 0] = np.nan

    vmin = 0
    vmax = 40  # ajustable

    plt.figure(figsize=(6, 5))

    plt.imshow(
        data,
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

    # -----------------------
    # TXT traçabilitat
    # -----------------------
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(f"Daily accumulation for {day}\n")
        f.write("Input files used:\n")
        for nc in sorted_files:
            f.write(nc.name + "\n")

    print("   ✅ Saved", out_file.name)
    print("   ✅ Saved", png_file.name)
    print("   ✅ Saved", txt_file.name)

    # =========================
    # 🗑️ ELIMINAR FITXERS 6H UTILITZATS
    # =========================
    print("   🗑️ Eliminant fitxers 6H utilitzats...")

    # tanquem datasets abans d'esborrar
    ds_all.close()
    for ds in ds_list:
        ds.close()

    for f in sorted_files:
        try:
            os.remove(f)
            print(f"      ✔ Esborrat {f.name}")
        except Exception as e:
            print(f"      ⚠️ No s'ha pogut esborrar {f.name}: {e}")

print("🏁 Procés completat")




# =========================================================
# =========================================================
# =========================================================
# =========================================================
# 📅 ACUMULAT SETMANAL (executa cada dia però només actua dilluns)
# =========================================================

import datetime

print("\n🗓️ Comprovant si cal generar acumulat setmanal...")

today = datetime.date.today()

# Només dilluns (weekday(): dilluns=0)
if today.weekday() == 0:

    print("📆 Avui és dilluns — comprovant setmana anterior")

    # diumenge anterior (ahir)
    last_sunday = today - datetime.timedelta(days=1)

    # dilluns de la setmana anterior
    last_monday = last_sunday - datetime.timedelta(days=6)

    print(f"   Setmana objectiu: {last_monday} → {last_sunday}")

    # noms esperats
    expected_days = [
        last_monday + datetime.timedelta(days=i)
        for i in range(7)
    ]

    weekly_files = []
    missing = []

    for d in expected_days:
        f = DATA_DIR / f"GLD_RNN24H_{d.strftime('%Y%m%d')}.nc"
        if f.exists():
            weekly_files.append(f)
        else:
            missing.append(f.name)

    # ----------------------------
    # Si falten dies → no fem res
    # ----------------------------
    if missing:
        print("⚠️ No es pot generar setmanal. Falten:")
        for m in missing:
            print("   -", m)

    else:
        print("✅ Tots els dies disponibles — generant setmanal")

        week_id = last_sunday.strftime("%Y%m%d")

        weekly_nc = DATA_DIR / f"GLD_RNN7D_{week_id}.nc"
        weekly_png = PNG_DIR / f"GLD_RNN7D_{week_id}.png"
        weekly_txt = DATA_DIR / f"GLD_RNN7D_{week_id}.txt"

        if weekly_nc.exists():
            print("ℹ️ Setmanal ja existeix — skip")
        else:

            # =========================
            # Obrir i sumar
            # =========================
            ds_list = [xr.open_dataset(f) for f in weekly_files]
            ds_all = xr.concat(ds_list, dim="time")

            valid_count = ds_all["precipitation_mm_24h"].notnull().sum(dim="time")
            weekly_sum = ds_all["precipitation_mm_24h"].sum(dim="time", skipna=True)
            weekly_sum = weekly_sum.where(valid_count > 0)

            out_week = xr.Dataset({"precipitation_mm_7d": weekly_sum})
            out_week.to_netcdf(weekly_nc)

            # =========================
            # PNG setmanal
            # =========================
            print("   🎨 Generant PNG setmanal...")

            data = weekly_sum.values.astype(float)
            data[data <= 0] = np.nan

            vmin = 0
            vmax = 100  # 👈 ajusta per setmanal

            plt.figure(figsize=(6, 5))

            plt.imshow(
                data,
                origin="upper",
                cmap="turbo",
                vmin=vmin,
                vmax=vmax,
            )

            plt.axis("off")

            plt.savefig(
                weekly_png,
                dpi=150,
                bbox_inches="tight",
                pad_inches=0,
                transparent=True,
            )

            plt.close()

            # =========================
            # TXT traçabilitat
            # =========================
            with open(weekly_txt, "w", encoding="utf-8") as f:
                f.write(
                    f"Weekly accumulation {last_monday} to {last_sunday}\n"
                )
                f.write("Input daily files used:\n")
                for nc in weekly_files:
                    f.write(nc.name + "\n")

            # tancar datasets
            ds_all.close()
            for ds in ds_list:
                ds.close()

            print("   ✅ Setmanal generat")

else:
    print("📅 Avui no és dilluns — skip setmanal")

