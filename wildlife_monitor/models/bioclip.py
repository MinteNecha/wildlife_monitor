"""
BioCLIP retrieval model (Package P2).

This wraps the BioCLIP vision-language model and exposes a single method,
:meth:`rank_by_species`, that scores and ranks images by how well they
match a species text prompt. Both the BioCLIP+SAM and BioCLIP+YOLO
pipelines use this one class, so the retrieval logic exists in exactly
one place instead of being copied into each pipeline.
"""

from __future__ import annotations

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from wildlife_monitor.config import BIOCLIP_MODEL, PROMPT_TEMPLATE, SystemConfig


class BioCLIPModel:
    """Ranks images by cosine similarity to a species text prompt.

    The model and tokenizer are loaded once when the object is created.
    Text embeddings are computed once per call, before the image loop, so
    the cost of encoding the prompt is paid a single time no matter how
    many images are scored.
    """

    def __init__(self) -> None:
        import open_clip

        self.device = SystemConfig.instance().device
        print(f"[INFO] Loading BioCLIP on {self.device} ...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            BIOCLIP_MODEL, pretrained=None
        )
        self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(BIOCLIP_MODEL)

    def _encode_prompt(self, species: str) -> torch.Tensor:
        """Encode the species prompt into a normalised text embedding."""
        prompt = PROMPT_TEMPLATE.format(species=species)
        tokens = self.tokenizer([prompt]).to(self.device)
        with torch.no_grad():
            text_feat = self.model.encode_text(tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        return text_feat

    def rank_by_species(
        self, frame: pd.DataFrame, species: str, top_n: int
    ) -> pd.DataFrame:
        """Score every image in ``frame`` and return the ``top_n`` best.

        A min-max normalised ``confidence`` column in ``[0, 1]`` is added
        alongside the raw ``bioclip_score`` so downstream consumers have a
        calibrated value to threshold on.
        """
        text_feat = self._encode_prompt(species)

        scores: list[float] = []
        for _, row in tqdm(frame.iterrows(), total=len(frame),
                           desc="BioCLIP scoring"):
            try:
                image = self.preprocess(
                    Image.open(row["local_image_path"]).convert("RGB")
                ).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    image_feat = self.model.encode_image(image)
                    image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
                score = float((image_feat @ text_feat.T).squeeze())
            except Exception:
                score = 0.0
            scores.append(score)

        ranked = frame.copy()
        ranked["bioclip_score"] = scores

        low, high = min(scores), max(scores)
        if high > low:
            ranked["confidence"] = (ranked["bioclip_score"] - low) / (high - low)
        else:
            ranked["confidence"] = 1.0

        top = ranked.nlargest(top_n, "bioclip_score").reset_index(drop=True)
        print(f"[INFO] Selected top {len(top)} images by BioCLIP score.")
        return top

    def release(self) -> None:
        """Free GPU memory held by the model (call before loading SAM)."""
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
