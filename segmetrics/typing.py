import numpy as np
import numpy.typing as npt

LabelImage = npt.NDArray[np.integer]

BinaryImage = npt.NDArray[np.bool_]

Image = LabelImage | BinaryImage
