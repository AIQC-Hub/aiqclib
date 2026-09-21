Derived Values
===========================

The ``derived_values`` feature is an observation-level feature holding the
quantities that are computed from the measured variables rather than measured
themselves: density, depth and potential temperature.

Density is the one the feature proposal asks for by name. It is not a column a
CTD produces; it follows from temperature, salinity and pressure through the
equation of state. Depth is the same story in reverse: the instrument reports
pressure, and turning that into metres needs the latitude, because gravity
varies with it.

Configuration: Setup
-------------------------------------

.. code-block:: yaml

   feature_sets:
     - name: feature_set_1
       features:
         - derived_values

Configuration: Parameters
-------------------------------------

``outputs`` names the columns to emit; leaving it out emits all of them.

.. code-block:: yaml

   feature_param_sets:
     - name: feature_set_1_param_set_1
       params:
         - feature: derived_values
           outputs: [ sigma0, depth ]
           stats_set: { type: raw }

================================ =========================================
Output                           What it holds
================================ =========================================
``sigma0``                       Potential density anomaly in kg/m³: the
                                 density the parcel would have at the
                                 surface, less 1000. Comparing this rather
                                 than in-situ density removes the pressure
                                 effect, which would otherwise swamp the
                                 difference between two levels.
``depth``                        Metres below the surface, from pressure
                                 and latitude (UNESCO 1983).
``potential_temperature``        The temperature the parcel would have at
                                 the surface, in °C.
================================ =========================================

``params`` names the input columns, since only the dataset knows what they are
called:

================================ ================== =====================
Parameter                        Default            Meaning
================================ ================== =====================
``salinity_column``              ``psal``           Practical salinity.
``temperature_column``           ``temp``           In-situ temperature.
``pressure_column``              ``pres``           Pressure in decibars.
``latitude_column``              ``latitude``       Latitude in degrees.
``prefer_input_columns``         ``true``           Use a column of the
                                                    output's own name when
                                                    the input has one,
                                                    instead of recomputing.
================================ ================== =====================

Missing and impossible values
-------------------------------------

The equation of state is a fit to measurements of sea water, so a value that
cannot be one has no density. Inputs outside the accepted domain (see
:data:`~aiqclib.common.utils.seawater.SALINITY_LIMITS` and its neighbours)
produce null, exactly as a missing value does. A placeholder such as ``-999``
therefore yields no density rather than a finite number hundreds of units away
from sea water.

``prefer_input_columns`` exists because a dataset that already carries its own
density was given it by someone, and quietly disagreeing with them helps
nobody. Set it to ``false`` to always recompute.
