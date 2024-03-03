from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:

    LabelImage = npt.NDArray[np.integer]

    BinaryImage = npt.NDArray[np.bool_]

    Image = LabelImage | BinaryImage

else:

    LabelImage = None
    BinaryImage = None
    Image = None
