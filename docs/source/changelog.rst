Change log
==========

0.11.3
------

First public release

1.0.0
-----

Drops compatibility with Python 2.7.

Adds ``replace`` argument to ``segmetrics.study.Study`` class.

Marks functions in ``segmetrics.boundary`` as private.

Marks ``segmetrics.boundary.ObjectBasedDistance.obj_mapping`` as private.

Removes ``segmetrics.legacy``.

Renames ``segmetrics.metric.Metric`` to ``segmetrics.measure.Measure``.

Renames ``segmetrics.boundary.ObjectBasedDistance`` to ``segmetrics.boundary.ObjectBasedDistanceMeasure``.

1.1.0
-----

Adds ``segmetrics.cli``.

Adds ``segmetrics.Measure.accumulative`` parameter and makes detection-based measures non-accumulative by default.

Removes the ``segmetrics.Measure.ACCUMULATIVE`` field.

1.2.0
-----

Changes default names of performance measures in ``segmetrics.study.Study.add_measure()``.

Adds ``segmetrics.measure.Measure.default_name()`` method.

1.2.1
-----

Adds quantile-based implementation of ``segmetrics.boundary.Hausdorff``.

1.2.2
-----

Positional and keyword arguments passed to ``segmetrics.boundary.DistanceMeasure.object_based`` are passed through to ``ObjectBasedDistanceMeasure``.

1.2.3
-----

Adds ``--semicolon`` CLI option.

1.3
---

Removes the ``FRACTIONAL`` class variable of the ``segmetrics.measure.Measure`` base class.

