segmetrics
==========

.. image:: https://img.shields.io/badge/Install%20with-conda-%2387c305
    :target: https://anaconda.org/bioconda/segmetrics

.. image:: https://img.shields.io/conda/v/bioconda/segmetrics.svg?label=Version
    :target: https://anaconda.org/bioconda/segmetrics

.. image:: https://img.shields.io/conda/dn/bioconda/segmetrics.svg?label=Downloads
    :target: https://anaconda.org/bioconda/segmetrics

The goal of this package is to provide a low-threshold and standardized way of evaluating the performance of segmentation methods in biomedical image analysis and beyond, and to fasciliate the comparison of different methods. This package currently only supports 2-D image data, which may be extended to 3-D in the future.

The following *region-based* performance measures are currently implemented:

 - ``Dice``: Dice similarity coefficient
 - ``ISBIScore``: ISBI SEG Score [1]_
 - ``JaccardSimilarityIndex``: `Jaccard coefficient`_
 - ``JaccardIndex``: Jaccard index [2]_
 - ``RandIndex``: Rand index [2]_
 - ``AdjustedRandIndex``: `Adjusted Rand index`_

.. _`Jaccard coefficient`: https://en.wikipedia.org/wiki/Jaccard_index
.. _`Adjusted Rand index`: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html

The following *contour-based* performance measures are currently implemented:

 - ``Hausdorff``: Hausdorff distance (HSD) [3]_
 - ``NSD``: Normalized sum of distances (NSD) [2]_

The following *detection-based* performance measures [2]_ are currently implemented:

 - ``FalseSplit``: Falsely split objects per image
 - ``FalseMerge``: Falsely merged objects per image
 - ``FalsePositive``: Falsely detected objects per image
 - ``FalseNegative``: Undetected objects per image

Use ``python -m tests.all`` to run the test suite.

Usage
-----

Segmentation performance evaluation is driven by the ``Study`` class. The general procedure is to instantiate a ``Study`` object, add the required performance measures, and then to process the segmentation results. A simple example is:

.. code-block:: python

    import segmetrics as sm
    
    study = sm.Study()
    study.add_measure(sm.Dice(), 'Dice')
    study.add_measure(sm.ISBIScore(), 'SEG')
    
    for gt_img, seg_img in zip(gt_list, seg_list):
        study.set_expected(gt_img)
        study.process(seg_img)
    
    study.print_results()

In the example above, it is presumed that ``gt_list`` and ``seg_list`` are two iterables of ground truth segmentation and segmentation result images, respectively.

The following code can be used to include *object-based* distance measures:

.. code-block:: python

    study.add_measure(sm.NSD().object_based(), 'NSD')
    study.add_measure(sm.Hausdorff().object_based(), 'HSD')

The object correspondences between the ground truth objects and the segmented objects are established by choosing the closest object according to the respective distance function.

References
----------

.. [1] M. Maska et al., "A benchmark for comparison of cell tracking 1609 algorithms," Bioinformatics, vol. 30, no. 11, pp. 1609–1617, 2014.

.. [2] L. Coelho, A. Shariff, and R. Murphy, "Nuclear segmentation in microscope cell images: A hand-segmented dataset and comparison of algorithms," in Proc. Int. Symp. Biomed. Imag., 2009, pp. 518–521.

.. [3] P. Bamford, "Empirical comparison of cell segmentation algorithms using an annotated dataset," in Proc. Int. Conf. Image Proc., 1612 vol. 2, 2003, pp. II-1073–1076.
