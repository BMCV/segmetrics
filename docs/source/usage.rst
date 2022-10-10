User guide
==========

.. _installation:

Installation
------------

To use segmetrics, first install it using conda:

.. code-block:: console

   conda install segmetrics -c bioconda

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
