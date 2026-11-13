"""Deep Learning appearance Re-ID embedding extractor using MobileNetV3-Small."""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ReIDExtractor:
    """Extracts L2-normalized 576-d appearance embeddings for detected object crops."""

    def __init__(
        self,
        input_size: tuple[int, int] = (128, 128),
        device: str = "cpu",
        pretrained: bool = True,
    ) -> None:
        self.input_size = input_size
        self.device = torch.device(device)
        self.pretrained = pretrained
        self._model = None

    def _ensure_model(self) -> torch.nn.Module:
        if self._model is None:
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

            weights = MobileNet_V3_Small_Weights.DEFAULT if self.pretrained else None
            model = mobilenet_v3_small(weights=weights)
            model.eval()
            model.to(self.device)
            self._model = model
        return self._model

    def _prepare_crop(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = np.round(bbox).astype(int)
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1, min(width, x2))
        y2 = max(y1, min(height, y2))

        if (x2 - x1) < 4 or (y2 - y1) < 4:
            return None

        crop = frame[y1:y2, x1:x2]
        resized = cv2.resize(crop, self.input_size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        # HWC -> CHW
        return np.transpose(normalized, (2, 0, 1))

    def extract(self, frame: np.ndarray, bboxes: list[np.ndarray]) -> list[np.ndarray | None]:
        """Compute L2-normalized embeddings in a single batched forward pass."""
        if not bboxes:
            return []

        tensors: list[np.ndarray] = []
        valid_indices: list[int] = []
        for idx, bbox in enumerate(bboxes):
            prepared = self._prepare_crop(frame, bbox)
            if prepared is not None:
                tensors.append(prepared)
                valid_indices.append(idx)

        results: list[np.ndarray | None] = [None] * len(bboxes)
        if not tensors:
            return results

        model = self._ensure_model()
        batch = torch.from_numpy(np.stack(tensors, axis=0)).to(self.device)
        with torch.no_grad():
            features = model.features(batch)
            pooled = model.avgpool(features)
            flattened = torch.flatten(pooled, 1)
            normalized = F.normalize(flattened, p=2, dim=1)
            vectors = normalized.cpu().numpy().astype(np.float32)

        for out_idx, vec in zip(valid_indices, vectors):
            results[out_idx] = vec
        return results
