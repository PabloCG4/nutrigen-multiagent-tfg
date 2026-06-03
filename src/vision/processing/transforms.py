import torchvision.transforms.functional as vision_functional


class SquarePadding:
    """
    Shared image transformation that pads a PIL image to a square canvas using
    the ImageNet mean colour (R=123, G=116, B=103).

    Using the channel means as fill value ensures that, after the standard
    ImageNet normalisation step (mean subtraction / std division), the padded
    border pixels are mapped to exactly (0, 0, 0) — the statistically neutral
    activation for DINOv2.  Filling with black (0, 0, 0) before normalisation
    produces extreme negative activations (~-2.1) that are out-of-distribution
    for the pre-trained ViT and can corrupt the [CLS] token embedding.

    This class is the single source of truth for the padding transform.
    Both the database population script (populate_vector_db_padding.py) and the
    inference engine (semantic_engine.py) import it from here so that the
    preprocessing applied at index time and at query time is always identical.
    """

    # ImageNet channel means in uint8 space: mean * 255 ≈ (123, 116, 103).
    # After ToTensor + Normalize these values map to exactly (0, 0, 0).
    FILL_COLOR = (123, 116, 103)

    def __call__(self, pil_image):
        image_width, image_height = pil_image.size
        maximum_dimension = max(image_width, image_height)

        # Calculate symmetric padding to centre the image within the square canvas
        padding_left = (maximum_dimension - image_width) // 2
        padding_top = (maximum_dimension - image_height) // 2
        padding_right = maximum_dimension - image_width - padding_left
        padding_bottom = maximum_dimension - image_height - padding_top

        return vision_functional.pad(
            pil_image,
            (padding_left, padding_top, padding_right, padding_bottom),
            fill=self.FILL_COLOR,
            padding_mode='constant',
        )
