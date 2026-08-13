# models/

Model checkpoints live here.

- `sam_vit_b_01ec64.pth` — SAM 1 (auto-downloaded by setup_data.py)
- `yolo11n.pt` — YOLOv11 nano (auto-downloaded by Ultralytics)
- `sam3.pt` — SAM 3 (OPTIONAL, gated). Request access and download from
  https://huggingface.co/facebook/sam3 then place the file here to enable
  true text-prompted concept segmentation. Without it the SAM 3 pipeline
  falls back to SAM 1.
