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
    study.add_measure(sm.Dice())
    study.add_measure(sm.ISBIScore())
    
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

The choice of the suitable performance measaures for evaluation should depend on the application and the methods which are used for comparison (and the performance measures which were reported for those methods). In addition, the following considerations should be kept in mind when choosing suitable performance measures.

One of the most widely used performance measures is the ``Dice`` score. This is sensitive to false-positive detections, but invariant to falsely split/merged objects. On the other hand, ``ISBIScore`` is sensitive to falsely split/merged but invariant to false-positive detections. Thus, using ``Dice`` in combination with ``ISBIScore`` well reflects the overall segmentation performance from a region-based point of view.

The ``Hausdorff`` distance is overly sensitive to outliers (e.g., few objects which yield very high distance values). In fact, the sensitivity is higher than it is probably suitable in most applications. One solution is to use the object-based variant instead, which means that such outliers will be averaged out. Another, more simple solution, is to use the quantile-based variant of the ``Hausdorff`` distance instead, which cuts off the outliers based on a carefully chosen quantile value. Suitable choices for the quantile should be between ``0.9`` and ``0.99``, and should be chosen equal for all methods within a comparison. The ``NSD`` measure does not suffer from outliers. Using the quantile-based variant of the ``Hausdorff`` distance in combination with ``NSD`` thus well reflects the overall segmentation performance from a contour-based point of view.

Including the ``FalseSplit`` and ``FalseMerge`` measures is always useful in applications where the main challenge is the separation of the individual objects (e.g., cluster splitting in cell segmentation).

Object-based distance measures
******************************

The following code can be used to include *object-based* distance measures:

.. code-block:: python

    study.add_measure(sm.NSD().object_based())
    study.add_measure(sm.Hausdorff().object_based())

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

    python -m segmetrics.cli ./seg ".*t([0-9]+).png" ./gt/man_seg\\1.tif results.csv \
        "sm.ISBIScore()" "sm.FalseMerge()" "sm.FalseSplit()"

This will write the results to the file ``results.csv``. The list of performance measures is arbitrary. Refer to ``python -m segmetrics.cli --help`` for details.
