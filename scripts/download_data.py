
from pathlib import Path
from urllib.parse import urljoin, urlparse
import time

import requests
from bs4 import BeautifulSoup


URL = "https://mibici.net/es/datos-abiertos/"
DATA_DIR = Path("data/raw")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "es-MX,es;q=0.9,en;q=0.8",
}

DATA_DIR.mkdir(parents=True, exist_ok=True)

response = requests.get(
    URL,
    headers=HEADERS,
    timeout=30,
)

print("Código HTTP:", response.status_code)
print("Página consultada:", response.url)

response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

csv_links = set()

for link in soup.find_all("a", href=True):
    file_url = urljoin(URL, link["href"])

    if urlparse(file_url).path.lower().endswith(".csv"):
        csv_links.add(file_url)

if not csv_links:
    raise RuntimeError(
        "No se encontraron enlaces CSV en la página oficial."
    )

print(f"Se encontraron {len(csv_links)} archivos CSV.")

downloaded = 0
failed = []

for file_url in sorted(csv_links):
    filename = Path(urlparse(file_url).path).name
    destination = DATA_DIR / filename

    if destination.exists() and destination.stat().st_size > 0:
        print(f"Ya existe, se omite: {filename}")
        continue

    try:
        file_headers = HEADERS.copy()
        file_headers["Referer"] = URL
        
        file_response = requests.get(
            file_url,
            headers=file_headers,
            timeout=120,
        )
        file_response.raise_for_status()

        if not file_response.content:
            raise ValueError("El archivo está vacío.")

        destination.write_bytes(file_response.content)

        print(f"Descargado: {filename}")
        downloaded += 1
        time.sleep(0.2)

    except Exception as error:
        print(f"Error con {filename}: {error}")
        failed.append(file_url)

print("\nProceso terminado.")
print(f"Archivos nuevos descargados: {downloaded}")
print(f"Errores: {len(failed)}")

if failed:
    print("\nEnlaces que no se pudieron descargar:")
    for file_url in failed:
        print(file_url)
