User guide
==========

.. _installation:

Installation
------------

To use segmetrics, first install it using conda:

.. code-block:: console

   conda install -c bioconda segmetrics

Usage
-----

Segmentation performance evaluation is driven by the ``Study`` class. The general procedure is to instantiate a ``Study`` object, add the required performance measures, and then to process the segmentation results. A simple example is:

.. code-block:: python

    import segmetrics as sm
    
    study = sm.Study()
    study.add_measure(sm.Dice(), 'Dice')
    study.add_measure(sm.ISBIScore(), 'SEG')
    
    for file_idx, (gt_img, seg_img) in enumerate(zip(gt_list, seg_list):
        study.set_expected(gt_img)
        study.process(f'file-{file_idx}', seg_img)
    
    study.print_results()

In the example above, it is presumed that ``gt_list`` and ``seg_list`` are two iterables of ground truth segmentation and segmentation result images, respectively (they contain numpy arrays which represent segmentation masks).

The method ``study.process`` computes the performance measures for the segmentation ``seg_img`` with respect to the ground truth segmentation ``gt_img``. The first argument is an arbitrary indentifier of the segmentation image (e.g., the file name). Supplying the same identifier multiple times overrides any previously computed results for that identifier. This is particularily handy in an interactive environment, such as Jupyter notebooks. The identifier is also used in the detailed output of the study (e.g., ``study.tocsv()``).

Implemented measures
********************

Region-based performance measures:

- :class:`segmetrics.regional.Dice`
- :class:`segmetrics.regional.ISBIScore`
- :class:`segmetrics.regional.JaccardCoefficient`
- :class:`segmetrics.regional.JaccardIndex`
- :class:`segmetrics.regional.RandIndex`
- :class:`segmetrics.regional.AdjustedRandIndex`

Contour-based performance measures:

- :class:`segmetrics.boundary.Hausdorff`
- :class:`segmetrics.boundary.NSD`

Detection-based performance measures:

- :class:`segmetrics.detection.FalseSplit`
- :class:`segmetrics.detection.FalseMerge`
- :class:`segmetrics.detection.FalsePositive`
- :class:`segmetrics.detection.FalseNegative`

Object-based distance measures
******************************

The following code can be used to include *object-based* distance measures:

.. code-block:: python

    study.add_measure(sm.NSD().object_based(), 'NSD')
    study.add_measure(sm.Hausdorff().object_based(), 'HSD')

The object correspondences between the ground truth objects and the segmented objects are established by choosing the closest object according to the respective distance function.

Parallel computing
******************

It is also easy to exploit the computational advantages of multi-core systems by evaluating multiple images in parallel via the ``parallel`` interface:

.. code-block:: python

    sample_ids = list(range(len(seg_list)))
    for sample_id in sm.parallel.process(study, seg_list.__getitem__, gt_list.__getitem__, sample_ids, num_forks=2):
        print(f'Finished processing: {sample_id}')
    
Or even more simply:

.. code-block:: python

    sample_ids = list(range(len(seg_list)))
    sm.parallel.process_all(study, seg_list.__getitem__, gt_list.__getitem__, sample_ids, num_forks=2)

Command line interface
**********************

For example, assume the following directory structure:

.. code-block::

    ./seg/t02.png
    ./seg/t04.png
    ./seg/t12.png
    ./gt/man_seg02.tif
    ./gt/man_seg04.tif
    ./gt/man_seg12.tif

Then, an evaluation of the segmentation performance can be performed using the following command:

.. code-block:: bash

    python -m segmetrics.cli ./seg ".*t([0-9]+).png" ./gt/man_seg\\1.tif GOWT1-1.csv "sm.ISBIScore()" "sm.FalseMerge()" "sm.FalseSplit()"

This will write the results to the file ``GOWT1-1.csv``. The list of performance measures is arbitrary. Refer to ``python -m segmetrics.cli --help`` for details.
