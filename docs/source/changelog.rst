Change log
==========

0.11.3
------

First public release

1.0.0
-----

Drop compatibility with Python 2.7.

Add ``replace`` argument for ``segmetrics.study.Study`` class.

Mark functions in ``segmetrics.boundary`` as private.

Mark ``segmetrics.boundary.ObjectBasedDistance.obj_mapping`` as private.

Remove ``segmetrics.legacy``.

Rename ``segmetrics.metric.Metric`` to ``segmetrics.measure.Measure``.

Rename ``segmetrics.boundary.ObjectBasedDistance`` to ``segmetrics.boundary.ObjectBasedDistanceMeasure``.

1.1.0
-----

Add ``segmetrics.cli``.

Add ``segmetrics.Measure.accumulative`` parameter and makes detection-based measures non-accumulative by default.

Remove the ``segmetrics.Measure.ACCUMULATIVE`` field.

1.2.0
-----

Change default names of performance measures in ``segmetrics.study.Study.add_measure()``.

Add ``segmetrics.measure.Measure.default_name()`` method.

1.2.1
-----

Add quantile-based implementation of ``segmetrics.boundary.Hausdorff``.

1.2.2
-----

Pass positional and keyword arguments for ``segmetrics.boundary.DistanceMeasure.object_based`` through to ``ObjectBasedDistanceMeasure``.

1.2.3
-----

Add ``--semicolon`` CLI option.

1.3
---

Add the ``aggregation`` keyword argument and member variable of the ``segmetrics.measure.Measure`` base class.

Remove the ``accumulative`` keyword argument and member variable of the ``segmetrics.measure.Measure`` base class.

Remove the ``FRACTIONAL`` class variable of the ``segmetrics.measure.Measure`` base class.

The method ``add_measure``` of the ``segmetrics.study.Study`` class now returns the name of the measure.

1.4
---

Change default name of ``segmetrics.regional.ISBIScore`` if ``min_ref_size`` is not the default.

Change default name of ``segmetrics.boundary.ObjectBasedDistanceMeasure`` if ``skip_fn`` is not the default.

1.5
---

Rename ``segmetrics.boundary`` to ``segmetrics.contour``.

Move ``segmetrics.detection.COCOmAP`` to ``segmetrics.deprecated.COCOmAP``.

Minimum supported Python version is now 3.8 due to type linting.

Replace the ``ObjectBasedDistanceMeasure`` class by the more general ``segmetrics.measure.ObjectMeasureAdapter`` class.

Add ``.object_based()`` method for all image-level measures (including region-based measures).
The method does not accept positional arguments any more, only keyword arguments.

Remove the argument and attribute ``mode`` from the Hausdorff distance.

Add ``.reversed()`` and ``.symmetric()`` methods for all asymmetric measures (e.g., object-based).

Add ``geometric-mean`` aggregation method.

Rename aggregation method ``obj-mean`` to ``object-mean``.