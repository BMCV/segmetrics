import numpy as np

LabelImage = np.typing.NDArray[np.integer]

BinaryImage = np.typing.NDArray[np.bool_]

Image = LabelImage | BinaryImage
