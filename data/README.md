# data/

Place your prepared dataset here:

- `subset_metadata.csv` — one row per image, with columns:
  `image_id, local_image_path, species_label, date_captured,
   site_id, latitude, longitude, habitat_type`
- `images/` — the camera trap JPEG files referenced by
  `local_image_path`.

`scripts/setup_data.py` checks that these are present.
