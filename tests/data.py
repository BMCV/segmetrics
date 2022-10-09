import numpy as np
import re


# Four square objects uniformly distributed.

img1 = np.zeros((500, 500), 'uint8')
img1[100:200, 100:200] = 1
img1[300:400, 100:200] = 2
img1[100:200, 300:400] = 3
img1[300:400, 300:400] = 4


# Like img1, but slightly different in position and size.

img2 = np.zeros((500, 500), 'uint8')
img2[120:220, 120:220] = 1
img2[340:440, 100:200] = 2
img2[100:200, 250:400] = 3
img2[300:480, 300:400] = 4


# Like img2, but without the object in the bottom right.

img3 = np.zeros((500, 500), 'uint8')
img3[120:220, 120:220] = 1
img3[340:440, 100:200] = 2
img3[100:200, 250:400] = 3


# Like img1, but the bottom right object is significantly larger.

img4 = np.zeros((500, 500), 'uint8')
img4[100:200, 100:200] = 1
img4[300:400, 100:200] = 2
img4[100:200, 300:400] = 3
img4[250:500, 250:500] = 4


# Like img2, but without the objects on the right.

img5 = np.zeros((500, 500), 'uint8')
img5[120:220, 120:220] = 1
img5[340:440, 100:200] = 2


# Like img1, but the objects on the right are touching.

img6 = np.zeros((500, 500), 'uint8')
img6[100:200, 100:200] = 1
img6[300:400, 100:200] = 2
img6[100:250, 300:400] = 3
img6[250:400, 300:400] = 4


# Like img1, but the objects on the right are merged.

img7 = np.zeros((500, 500), 'uint8')
img7[120:220, 120:220] = 1
img7[340:440, 100:200] = 2
img7[ 90:410, 280:420] = 3


images = [
    img1,
    img2,
    img3,
    img4,
    img5,
    img6,
    img7,
]


class CrossSampler:

    def __init__(self, images1, images2):
        self.images1 = images1
        self.images2 = images2

    def __len__(self):
        return len(self.images1) * len(self.images2)

    def __getitem__(self, pos):
        assert 0 <= pos < len(self)
        i, j = pos % len(self.images1), pos // len(self.images1)
        return f'sample-{i}-{j}', self.images1[i], self.images2[j]

    def all(self):
        for pos in range(len(self)):
            yield self[pos]

    @property
    def sample_ids(self):
        return [sid for sid, _, _ in self.all()]

    @property
    def img1_list(self):
        return [img1 for _, img1, _ in self.all()]

    @property
    def img2_list(self):
        return [img2 for _, _, img2 in self.all()]

    def img1(self, sample_id):
        i = int(re.match(r'^sample-([0-9]+)-[0-9]+$', sample_id).group(1))
        return self.images1[i]

    def img2(self, sample_id):
        j = int(re.match(r'^sample-[0-9]+-([0-9]+)$', sample_id).group(1))
        return self.images2[j]

