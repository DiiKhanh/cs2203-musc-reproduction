"""Dataset adapter for one validated, unlabeled image category."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from PIL import Image, ImageOps, UnidentifiedImageError

from .models import ValidatedImage


class ImageDecodeError(RuntimeError):
    """Raised when a previously validated image can no longer be decoded."""

    def __init__(self, image_id: str, source_name: str) -> None:
        self.image_id = image_id
        self.source_name = source_name
        super().__init__(f"Unable to decode {source_name} ({image_id})")


class UnlabeledImageDataset:
    """Lazily load validated images and apply the upstream preprocessing."""

    def __init__(
        self,
        images: Sequence[ValidatedImage],
        transform: Callable[[Image.Image], object],
    ) -> None:
        self.images = tuple(images)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self.images[index]
        try:
            with Image.open(item.path) as source:
                oriented = ImageOps.exif_transpose(source)
                rgb = oriented.convert("RGB")
                transformed = self.transform(rgb)
        except (OSError, UnidentifiedImageError, ValueError) as error:
            raise ImageDecodeError(item.image_id, item.source_name) from error

        return {
            "image": transformed,
            "image_id": item.image_id,
            "source_name": item.source_name,
            "path": item.path,
        }
