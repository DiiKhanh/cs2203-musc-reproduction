from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from musc_demo.dataset_adapter import ImageDecodeError, UnlabeledImageDataset
from musc_demo.models import ValidatedImage


def validated(path: Path, image_id: str, source_name: str) -> ValidatedImage:
    return ValidatedImage(
        image_id=image_id,
        source_name=source_name,
        path=path,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def test_dataset_preserves_stable_order_and_metadata(tmp_path: Path) -> None:
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    Image.new("RGB", (4, 3), (1, 2, 3)).save(first_path)
    Image.new("RGB", (5, 2), (4, 5, 6)).save(second_path)
    images = (
        validated(first_path, "id-first", "line-a/first.png"),
        validated(second_path, "id-second", "line-b/second.png"),
    )
    dataset = UnlabeledImageDataset(images, transform=lambda image: image.size)

    assert len(dataset) == 2
    assert dataset[0] == {
        "image": (4, 3),
        "image_id": "id-first",
        "source_name": "line-a/first.png",
        "path": first_path,
    }
    assert dataset[1]["image_id"] == "id-second"
    assert dataset[1]["source_name"] == "line-b/second.png"


@pytest.mark.parametrize(
    ("mode", "suffix"),
    [("L", ".png"), ("RGBA", ".png"), ("CMYK", ".jpg")],
)
def test_converts_grayscale_rgba_and_cmyk_to_rgb(
    tmp_path: Path, mode: str, suffix: str
) -> None:
    path = tmp_path / f"source{suffix}"
    color = 42 if mode == "L" else tuple(42 for _ in mode)
    Image.new(mode, (3, 2), color).save(path)
    seen_modes: list[str] = []

    def transform(image: Image.Image) -> str:
        seen_modes.append(image.mode)
        return image.mode

    dataset = UnlabeledImageDataset(
        (validated(path, "image-id", path.name),), transform=transform
    )

    assert dataset[0]["image"] == "RGB"
    assert seen_modes == ["RGB"]


def test_applies_exif_orientation_before_transform(tmp_path: Path) -> None:
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (2, 3), (20, 40, 60)).save(path, exif=exif)
    seen_sizes: list[tuple[int, int]] = []

    def transform(image: Image.Image) -> tuple[int, int]:
        seen_sizes.append(image.size)
        return image.size

    dataset = UnlabeledImageDataset(
        (validated(path, "rotated-id", path.name),), transform=transform
    )

    assert dataset[0]["image"] == (3, 2)
    assert seen_sizes == [(3, 2)]


def test_duplicate_basenames_remain_distinct(tmp_path: Path) -> None:
    first_path = tmp_path / "a" / "image.png"
    second_path = tmp_path / "b" / "image.png"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    Image.new("RGB", (2, 2), (1, 2, 3)).save(first_path)
    Image.new("RGB", (2, 2), (4, 5, 6)).save(second_path)
    dataset = UnlabeledImageDataset(
        (
            validated(first_path, "hash-a-001", "a/image.png"),
            validated(second_path, "hash-b-001", "b/image.png"),
        ),
        transform=lambda image: image.getpixel((0, 0)),
    )

    assert [dataset[index]["image_id"] for index in range(2)] == [
        "hash-a-001",
        "hash-b-001",
    ]
    assert [dataset[index]["source_name"] for index in range(2)] == [
        "a/image.png",
        "b/image.png",
    ]


def test_transform_receives_a_pillow_rgb_image(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    Image.new("L", (3, 3), 80).save(path)
    received: list[tuple[type[Image.Image], str]] = []

    def transform(image: Image.Image) -> str:
        received.append((type(image), image.mode))
        return "transformed"

    dataset = UnlabeledImageDataset(
        (validated(path, "image-id", path.name),), transform=transform
    )

    assert dataset[0]["image"] == "transformed"
    assert received[0][1] == "RGB"
    assert issubclass(received[0][0], Image.Image)


def test_raises_typed_decode_error_when_validated_file_disappears(
    tmp_path: Path,
) -> None:
    path = tmp_path / "image.png"
    Image.new("RGB", (3, 3), (1, 2, 3)).save(path)
    item = validated(path, "image-id", "nested/image.png")
    path.unlink()
    dataset = UnlabeledImageDataset((item,), transform=lambda image: image)

    with pytest.raises(ImageDecodeError) as error:
        dataset[0]

    assert error.value.image_id == "image-id"
    assert error.value.source_name == "nested/image.png"
