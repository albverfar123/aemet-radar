import requests
import tarfile
import io
import os
from datetime import datetime

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

            # extreiem el fitxer a memòria
            extracted = tar.extractfile(member)
            if extracted is None:
                continue

            # intentem obtenir hora del nom
            # exemple: ...202602131200...
            timestamp = None
            for part in name.split("_"):
                if part.isdigit() and len(part) >= 12:
                    timestamp = part[:12]
                    break

            if timestamp is None:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
            else:
                timestamp = f"{timestamp[:8]}_{timestamp[8:12]}"

            out_name = f"GLD_RNN6H_{timestamp}.tif"
            out_path = os.path.join(OUT_DIR, out_name)

            with open(out_path, "wb") as f:
                f.write(extracted.read())

            print("Guardat com:", out_path)