import requests
import tarfile
import io
import os
import re

URL = "https://www.aemet.es/es/api-eltiempo/radar/download/RNN"
OUT_DIR = "data"

os.makedirs(OUT_DIR, exist_ok=True)

print("Descarregant...")

r = requests.get(URL, timeout=300)
r.raise_for_status()

print("Processant tar.gz...")

with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tar:
    for member in tar.getmembers():
        name = member.name

        # filtrem només Gelida acumulat 6h
        if "GLD" in name and "RNN.6HR_CAPPI" in name and name.endswith(".tif"):
            print("Trobat:", name)

            extracted = tar.extractfile(member)
            if extracted is None:
                continue

            # -------------------------------------------------
            # 🔴 EXTREURE DATA REAL DEL NOM ORIGINAL
            # -------------------------------------------------
            # busca patró YYMMDDHHMMSS després de GLD
            m = re.search(r"GLD(\d{12})", name)

            if m:
                ts = m.group(1)  # YYMMDDHHMMSS

                # convertir a AAAAMMDD_HHmm
                year = "20" + ts[0:2]
                month = ts[2:4]
                day = ts[4:6]
                hour = ts[6:8]
                minute = ts[8:10]

                timestamp = f"{year}{month}{day}_{hour}{minute}"
            else:
                # fallback molt rar
                from datetime import datetime
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")

            # -------------------------------------------------
            # Nom final
            # -------------------------------------------------
            out_name = f"GLD_RNN6H_{timestamp}.tif"
            out_path = os.path.join(OUT_DIR, out_name)

            with open(out_path, "wb") as f:
                f.write(extracted.read())

            print("Guardat com:", out_path)
