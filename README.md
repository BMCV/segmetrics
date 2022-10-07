# segmetrics.py

[![Anaconda-Server Badge](https://anaconda.org/bioconda/segmetrics/badges/version.svg?)](https://anaconda.org/bioconda/segmetrics)
[![Anaconda-Server Badge](https://anaconda.org/bioconda/segmetrics/badges/latest_release_date.svg?)](https://anaconda.org/bioconda/segmetrics)
[![Anaconda-Server Badge](https://anaconda.org/bioconda/segmetrics/badges/downloads.svg?)](https://anaconda.org/bioconda/segmetrics)
[![Anaconda-Server Badge](https://anaconda.org/bioconda/segmetrics/badges/installer/conda.svg)](https://conda.anaconda.org/bioconda)

A Python package implementing image segmentation and object detection performance measures, for biomedical image analysis and beyond.

The following *region-based* performance measures are currently implemented:

 - `regional.Dice`: Dice similarity coefficient
 - `regional.ISBIScore`: ISBI SEG Score [1]
 - `regional.JaccardSimilarityIndex`: [Jaccard coefficient](https://en.wikipedia.org/wiki/Jaccard_index)
 - `regional.JaccardIndex`: Jaccard index [2]
 - `regional.RandIndex`: Rand index [2]
 - `regional.AdjustedRandIndex`: [Adjusted Rand index](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)

The following *contour-based* performance measures are currently implemented:

 - `boundary.Hausdorff`: Hausdorff distance (HSD) [3]
 - `boundary.NSD`: Normalized sum of distances (NSD) [2]
 - `boundary.ObjectBasedDistance`: Adapter for object-based HSD and NSD

The following *detection-based* performance measures [2] are currently implemented:

 - `detection.FalseSplit`: Falsely split objects per image
 - `detection.FalseMerge`: Falsely merged objects per image
 - `detection.FalsePositive`: Falsely detected objects per image
 - `detection.FalseNegative`: Undetected objects per image

## Usage

Segmentation performance evaluation is driven by the `Study` class. The general procedure is to instantiate a `Study` object, add the required performance measures, and then to process the segmentation results. A simple example is:

```python
import segmetrics as sm

study = sm.study.Study()
study.add_measure(sm.regional.Dice(), 'Dice')
study.add_measure(sm.regional.ISBIScore(), 'SEG')

for gt_img, seg_img in zip(gt_list, seg_list):
    study.set_expected(gt_img)
    study.process(seg_img)

study.print_results()
```

In the example above, it is presumed that `gt_list` and `seg_list` are two iterables of ground truth segmentation and segmentation result images, respectively.

The following code can be used to include *object-based* distance measures:

```python
study.add_measure(sm.boundary.ObjectBasedDistance(sm.boundary.NSD()), 'NSD')
study.add_measure(sm.boundary.ObjectBasedDistance(sm.boundary.Hausdorff()), 'HSD')
```

The object correspondences between the ground truth objects and the segmented objects are established by choosing the closest object according to the respective distance function.

See [test.ipynb](test.ipynb) for more examples.

## References

[1] M. Maska et al., "A benchmark for comparison of cell tracking 1609 algorithms," Bioinformatics, vol. 30, no. 11, pp. 1609–1617, 2014.

[2] L. Coelho, A. Shariff, and R. Murphy, "Nuclear segmentation in microscope cell images: A hand-segmented dataset and comparison of algorithms," in Proc. Int. Symp. Biomed. Imag., 2009, pp. 518–521.

[3] P. Bamford, "Empirical comparison of cell segmentation algorithms using an annotated dataset," in Proc. Int. Conf. Image Proc., 1612 vol. 2, 2003, pp. II-1073–1076.
