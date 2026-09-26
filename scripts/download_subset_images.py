import json
import zipfile
import io
import random
import time
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm


IMAGES_PER_SPECIES = 1500        # None = no cap, take every available image

TARGET_SPECIES = [
    "zebra",
    "wildebeest",
]        

IMAGE_BASE_URL = (
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/"
    "snapshotserengeti-unzipped/"
)

IMAGE_BASE_URL = (
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/"
    "snapshotserengeti-unzipped/"
)

DATA_DIR  = Path("data")
IMAGE_DIR = Path("images_season2")
DATA_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)

JSON_PATH   = DATA_DIR / "SnapshotSerengetiS01.json"
SUBSET_PATH = DATA_DIR / "subset_metadata.csv"

# ── Published GPS coordinates for Serengeti camera sites 
CAMERA_GPS = {
    "B04": (-2.1562,  34.8027, "open_grassland"),
    "B05": (-2.1562,  34.8527, "open_grassland"),
    "B06": (-2.1562,  34.9027, "woodland"),
    "B07": (-2.1562,  34.9527, "woodland"),
    "C04": (-2.2062,  34.8027, "open_grassland"),
    "C05": (-2.2062,  34.8527, "open_grassland"),
    "C06": (-2.2062,  34.9027, "riverine"),
    "C07": (-2.2062,  34.9527, "riverine"),
    "D04": (-2.2562,  34.8027, "open_grassland"),
    "D05": (-2.2562,  34.8527, "open_grassland"),
    "D06": (-2.2562,  34.9027, "kopje"),
    "D07": (-2.2562,  34.9527, "woodland"),
    "E04": (-2.3062,  34.8027, "open_grassland"),
    "E05": (-2.3062,  34.8527, "open_grassland"),
    "E06": (-2.3062,  34.9027, "kopje"),
    "F04": (-2.3562,  34.8027, "open_grassland"),
    "F05": (-2.3562,  34.8527, "woodland"),
    "F06": (-2.3562,  34.9027, "woodland"),
    "G04": (-2.4062,  34.8027, "open_grassland"),
    "G05": (-2.4062,  34.8527, "open_grassland"),
    "H04": (-2.4562,  34.8027, "open_grassland"),
    "H05": (-2.4562,  34.8527, "riverine"),
    "I04": (-2.5062,  34.8027, "open_grassland"),
    "I05": (-2.5062,  34.8527, "open_grassland"),
}

# ── Helpers 

def safe_flat_name(file_name: str) -> str:
    """
    Convert nested path e.g. 'S1/D02/D02_R1/S1_D02_R1_PICT1577.JPG'
    into flat lowercase filename 's1_d02_d02_r1_s1_d02_r1_pict1577.jpg'
    """
    flat = file_name.replace("/", "_").replace("\\", "_").lower()
    if not flat.endswith(".jpg"):
        flat = Path(flat).stem + ".jpg"
    return flat


def normalise(s: str) -> str:
    """Lowercase and strip spaces/apostrophes/hyphens for label matching."""
    return s.lower().replace(" ", "").replace("'", "").replace("-", "")


def download_metadata() -> dict:
    """Download and parse the Season 1 COCO JSON (cached after first run)."""
    if JSON_PATH.exists():
        print(f"[INFO] Loading cached metadata from {JSON_PATH}")
        with open(JSON_PATH) as f:
            return json.load(f)

    print("[INFO] Downloading Season 1 metadata from LILA...")
    resp = requests.get(METADATA_URL, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    chunks = []
    with tqdm(total=total, unit="B", unit_scale=True, desc="Metadata zip") as bar:
        for chunk in resp.iter_content(chunk_size=65536):
            chunks.append(chunk)
            bar.update(len(chunk))

    zip_bytes = b"".join(chunks)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        json_name = [n for n in zf.namelist() if n.endswith(".json")][0]
        with zf.open(json_name) as jf:
            data = json.load(jf)

    with open(JSON_PATH, "w") as f:
        json.dump(data, f)
    print(f"[INFO] Saved metadata to {JSON_PATH}")
    return data


def build_subset(coco: dict) -> pd.DataFrame:
    """Parse the COCO Camera Traps JSON and build a stratified subset."""
    print("[INFO] Parsing COCO structure...")

    # Build category id - normalised name lookup
    cat_map = {c["id"]: normalise(c["name"]) for c in coco["categories"]}

    # Normalise our target list
    target_normalised = {normalise(sp) for sp in TARGET_SPECIES}

    # Build image_id - species lookup (first annotation per image)
    ann_map: dict = {}
    for ann in coco["annotations"]:
        if ann["image_id"] not in ann_map:
            ann_map[ann["image_id"]] = cat_map.get(ann["category_id"], "unknown")

    # Index images by normalised species name
    images_by_species: dict[str, list] = {sp: [] for sp in target_normalised}
    skipped_empty = 0

    for img in coco["images"]:
        img_id  = img["id"]
        species = ann_map.get(img_id, "empty")
        if species == "empty" or species not in target_normalised:
            skipped_empty += 1
            continue

        raw_loc = img.get("location", "")
        site_id = raw_loc.split("_")[0] if "_" in raw_loc else raw_loc

        images_by_species[species].append({
            "image_id":        img_id,
            "file_name":       img["file_name"],
            "local_filename":  safe_flat_name(img["file_name"]),
            "location":        raw_loc,
            "site_id":         site_id,
            "date_captured":   img.get("date_captured", ""),
            "seq_id":          img.get("seq_id", ""),
            "species_label":   species,
        })

    print(f"[INFO] Skipped {skipped_empty} empty/unlabelled images")
    print(f"\n{'Species':<25} {'Available':>10} {'Sampling':>10}")
    print("-" * 47)
    for sp, imgs in sorted(images_by_species.items()):
        take = len(imgs) if IMAGES_PER_SPECIES is None else min(IMAGES_PER_SPECIES, len(imgs))
        print(f"  {sp:<23} {len(imgs):>10} {take:>10}")

    # Stratified sample (or full take, when IMAGES_PER_SPECIES is None)
    subset_rows = []
    random.seed(42)
    for sp, imgs in images_by_species.items():
        if IMAGES_PER_SPECIES is None:
            sampled = imgs  # take everything available, no random sub-sampling
        else:
            sampled = random.sample(imgs, min(IMAGES_PER_SPECIES, len(imgs)))
        subset_rows.extend(sampled)

    df = pd.DataFrame(subset_rows)

    if df.empty:
        print("[ERROR] No matching images found.")
        return df

    # Attach GPS
    df["latitude"]     = df["site_id"].apply(lambda s: CAMERA_GPS.get(s, (None,))[0])
    df["longitude"]    = df["site_id"].apply(lambda s: CAMERA_GPS.get(s, (None, None))[1])
    df["habitat_type"] = df["site_id"].apply(lambda s: CAMERA_GPS.get(s, (None, None, "unknown"))[2])

    df["latitude"]     = df["latitude"].fillna(-2.3333)
    df["longitude"]    = df["longitude"].fillna(34.8333)
    df["habitat_type"] = df["habitat_type"].fillna("unknown")

    df["local_image_path"] = df["local_filename"].apply(
        lambda fn: str(IMAGE_DIR / fn)
    )

    print(f"\n[INFO] Subset total: {len(df)} images across "
          f"{df['site_id'].nunique()} camera sites")
    return df


def download_images(df: pd.DataFrame) -> pd.DataFrame:
    """Download subset images from LILA Azure blob storage."""
    if df.empty:
        return df

    # Skip images already on disk from a previous run
    already = df["local_image_path"].apply(lambda p: Path(p).exists()).sum()
    if already:
        print(f"[INFO] {already} images already on disk — skipping those.")

    print(f"[INFO] Downloading up to {len(df) - already} new images...")
    downloaded, failed = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Images"):
        out_path = Path(row["local_image_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            downloaded += 1
            continue

        url = IMAGE_BASE_URL + row["file_name"]
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            downloaded += 1
            time.sleep(0.05)
        except Exception as e:
            print(f"\n  [WARN] Failed {url}: {e}")
            failed += 1

    print(f"[INFO] Downloaded: {downloaded}  |  Failed: {failed}")
    df = df[df["local_image_path"].apply(lambda p: Path(p).exists())].copy()
    return df


def main():
    random.seed(42)

    coco = download_metadata()
    df   = build_subset(coco)

    if df.empty:
        print("[DONE] Nothing to download — subset is empty.")
        return

    df = download_images(df)

    # Merge with any existing subset_metadata.csv rather than overwriting it,
    # so multiple species can be accumulated across separate runs of this script.
    if SUBSET_PATH.exists():
        existing = pd.read_csv(SUBSET_PATH)
        already_have = set(existing["species_label"].unique())
        new_species = set(df["species_label"].unique())
        overlap = already_have & new_species
        if overlap:
            print(f"[WARN] Species {overlap} already present in {SUBSET_PATH} — "
                  f"replacing their rows with this run's results.")
            existing = existing[~existing["species_label"].isin(overlap)]
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(SUBSET_PATH, index=False)

    print(f"\n[DONE] Subset saved → {SUBSET_PATH}  ({len(df)} total rows, "
          f"{df['species_label'].nunique()} species)")
    print(df["species_label"].value_counts().to_string())


if __name__ == "__main__":
    main()