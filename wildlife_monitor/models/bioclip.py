"""
BioCLIP retrieval model (Package P2).

This wraps the BioCLIP vision-language model and exposes it two ways.

:meth:`rank_by_species` is *retrieval*: given one species, rank images by how
well they match it. That is what the detection pipelines use when the caller
already knows what they are looking for.

:meth:`classify` is *identification*: given one image, score it against a list
of candidate species and return the best match. This is what a user with
unlabelled photographs needs, because they do not know what is in them.

Both share the same machinery — an image embedding and a text embedding,
compared by cosine similarity — and differ only in which side is held fixed.

One property of classification is worth stating plainly, because it shapes how
the results must be used: the candidate list is closed. BioCLIP will always
return whichever species on the list is nearest, even when the photograph
contains a bird, a vehicle, or nothing at all. Confidence does not reliably
catch this — the project's own evaluation found the wrong answers are not
necessarily low-confidence. That is why the classification pipeline runs an
animal detector first and only asks BioCLIP about frames that contain an
animal.
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

    def encode_prompts(self, species_list: list[str]) -> torch.Tensor:
        """Encode many species prompts at once into normalised embeddings.

        Encoding the text side once for the whole candidate list is the same
        saving that matters in retrieval: thirteen prompts encoded once rather
        than once per image.
        """
        prompts = [PROMPT_TEMPLATE.format(species=species)
                   for species in species_list]
        tokens = self.tokenizer(prompts).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def encode_image(self, image_path: str) -> torch.Tensor | None:
        """Normalised embedding for one image, or None when unreadable."""
        try:
            tensor = self.preprocess(
                Image.open(image_path).convert("RGB")
            ).unsqueeze(0).to(self.device)
        except Exception:
            return None
        with torch.no_grad():
            features = self.model.encode_image(tensor)
            return features / features.norm(dim=-1, keepdim=True)

    def classify(self, image_path: str, candidates: list[str],
                 top_k: int = 5,
                 text_features: torch.Tensor | None = None
                 ) -> list[tuple[str, float]]:
        """Identify one image against a candidate species list.

        Returns ``(species, probability)`` pairs, best first. Probabilities are
        a softmax over the candidates, so they sum to one and express *which
        of these* rather than *whether any of these* — a photograph of
        something not on the list still produces a confident-looking answer.
        """
        if not candidates:
            return []
        if text_features is None:
            text_features = self.encode_prompts(candidates)

        image_features = self.encode_image(image_path)
        if image_features is None:
            return []

        with torch.no_grad():
            similarity = (image_features @ text_features.T).squeeze(0)
            probabilities = (similarity * 100.0).softmax(dim=-1)

        scored = sorted(zip(candidates, probabilities.tolist()),
                        key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def classify_frame(self, frame: pd.DataFrame, candidates: list[str],
                        path_column: str = "local_image_path",
                        top_k: int = 5) -> pd.DataFrame:
        """Identify every image in a frame, adding prediction columns.

        Adds ``predicted_species``, ``confidence`` and ``top_k`` — the last
        being the ranked runners-up, which is what a reviewer needs in order
        to judge a borderline call.
        """
        text_features = self.encode_prompts(candidates)

        predictions, confidences, alternatives = [], [], []
        for _, row in tqdm(frame.iterrows(), total=len(frame),
                           desc="BioCLIP classifying"):
            ranked = self.classify(str(row[path_column]), candidates,
                                    top_k, text_features)
            if ranked:
                predictions.append(ranked[0][0])
                confidences.append(round(float(ranked[0][1]), 4))
                alternatives.append("; ".join(
                    f"{name}:{score:.3f}" for name, score in ranked))
            else:
                predictions.append("")
                confidences.append(0.0)
                alternatives.append("")

        classified = frame.copy()
        classified["predicted_species"] = predictions
        classified["confidence"] = confidences
        classified["top_k"] = alternatives
        return classified

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
