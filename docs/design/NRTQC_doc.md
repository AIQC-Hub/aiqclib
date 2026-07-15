# 3. Temperature and Salinity

In the following, automated RTQC will be listed for different types of temperature and salinity measurements, i.e. vertical profiles as well as time series. The automated QC procedures described here have been developed for the QC for the Argo data management (Argo, 2009) and have been extended to other profile data and time series. To improve the efficiency of some tests, specifications are incorporated into the validation process of regional measurements, depending on local water mass structures, statistics of data anomalies, the depth and gradient of the thermocline, as well as using regional enhanced bathymetry and climatology. If the salinity is calculated from the temperature and conductivity (CNDC) parameters, and the temperature is flagged ‘4’ (or ‘3’), then salinity will also be flagged ‘4’ (or ‘3’).

## 3.1. Required metadata

Detailed metadata are needed as guidelines to those involved in the collection, processing, QC and exchange of data. The quality controlled data set requires any data type (profiles, time series, trajectories, etc.) to be accompanied by key background information. A detailed metadata guideline for specific types of data including temperature and salinity measurements can be found in Eaton et al., 2009. Therefore only a short summary of required information is given below:

  1. **Position** of the measurement (latitude, longitude, depth).
  2. **Date** of the measurement (data and time in UTC or clearly specified local time zone).
  3. **Method** of the measurement (e.g. instrument types)
  4. **Specification** of the measurement (e.g. station numbers, cast numbers, platform code, name of the data distribution centre).
  5. **PI** of the measurement (name and institution of the data originator for traceability reasons).
  6. **Processing** of the measurement (e.g. details of processing and calibration already applied, algorithms used to compute derived parameters).
  7. **Comments** on measurement (e.g. problems encountered, comments on data quality, references to applied protocols).

## 3.2. RTQC for vertical profiles: Argo, CTD, XBT
Automated tests for vertical profiles are presented here, i.e. temperature and salinity measurements from Argo floats, CTDs and XBTs.

### RTQC1: Platform identification (applies only to GTS data and Argo)
Every centre handling GTS data and posting them to the GTS will need to prepare a metadata file for each float and in this is the WMO number that corresponds to each float ptt (platform transmitter terminal). There is no reason why, except because of a mistake, an unknown float ID should appear on the GTS.

**Action:** If the correspondence between the float ptt cannot be matched to the correct WMO number, none of the data from the profile should be distributed on the GTS.

### RTQC2: Impossible date test

The test requires that the observation date and time from the profile data are sensible.

  - Year greater than 1950
  - Month in range 1 to 12
  - Day in range expected for month
  - Hour in range 0 to 23
  - Minute in range 0 to 59

**Action:** If any one of the conditions fails, the date should be flagged as bad data.

### RTQC3: Impossible location test

The test requires that the observation latitude and longitude from the profile data be sensible.

  - Latitude in range –90 to 90
  - Longitude in range –180 to 180

**Action:** If either latitude or longitude fails, the position should be flagged as bad data.

### RTQC4: Position on land test

The test requires that the observation latitude and longitude from the profile measurement be located in an ocean. Use can be made of any file that allows an automatic test to see if data are located on land. We suggest use of at least the 2-minute bathymetry file that is generally available. This is commonly called and can be downloaded from http://www.ngdc.noaa.gov/mgg/global/etopo2.html. Action: If the data cannot be located in an ocean, the position should be flagged as bad data.

### RTQC5: Impossible speed test (applies only to GTS data and Argo)

Drift speeds for floats can be generated given the positions and times of the floats when they are at the surface and between profiles. In all cases we would not expect the drift speed to exceed 3 m/s. If it does, it means either a position or time is bad data, or a float is mislabelled. Using the multiple positions that are normally available for a float while at the surface, it is often possible to isolate the one position or time that is in error. Action: If an acceptable position and time can be used from the available suite, then the data can be distributed. Otherwise, flag the position, the time, or both as bad data.

### RTQC6: Global range test

This test applies a gross filter on observed values for temperature and salinity. It needs to accommodate all of the expected extremes encountered in the oceans.

 - Temperature in range -2.5°C to 40.0°C
 - Salinity in range 2 to 41.0

**Action:** If a value fails, it should be flagged as bad data. If temperature and salinity values at the same depth both fail, both values should be flagged as bad.

### RTQC7: Regional range test

This test applies only to certain regions of the world where conditions can be further qualified. In this case, specific ranges for observations from the Mediterranean and Red Seas further restrict what are considered sensible values. The Red Sea is defined by the region 10N, 40E; 20N, 50E; 30N, 30E; 10N, 40E and the Mediterranean Sea by the region 30N, 6W; 30N, 40E; 40N, 35E; 42N, 20E; 50N, 15E; 40N, 5E; 30N, 6W.

**Action:** Individual values that fail these ranges should be flagged as bad data.

#### Red Sea

  - Temperature in range 21.7°C to 40.0°C
  - Salinity in range 2.0 to 41.0

#### Mediterranean Sea

  - Temperature in range 10.0°C to 40.0°C
  - Salinity in range 2.0 to 40.0

#### North Western Shelves

  - Temperature in range –2.0°C to 24.0°C
  - Salinity in range 0.0 to 37.0

#### South West Shelves

  - Temperature in range –2.0°C to 30.0°C
  - Salinity in range 0.0 to 38.0

#### Arctic Sea

  - Temperature in range –1.92°C to 25.0°C
  - Salinity in range 2.0 to 40.0

### RTQC8: Pressure increasing test

This test requires that the profile has pressures that are monotonically increasing (assuming the pressures are ordered from smallest to largest).

**Action:** If there is a region of constant pressure, all but the first of a consecutive set of constant pressures should be flagged as bad data. If there is a region where pressure reverses, all of the pressures in the reversed part of the profile should be flagged as bad data.

### RTQC9: Spike test

A large difference between sequential measurements, where one measurement is quite different from adjacent ones, is a spike in both size and gradient. The test does not consider the differences in depth, but assumes a sampling that adequately reproduces the temperature and salinity changes with depth. The algorithm is used on both the temperature and salinity profiles:

  Test value = | V2 – (V3 + V1)/2 | – | (V3 – V1) / 2 |

where V2 is the measurement being tested as a spike, and V1 and V3 are the values above and below.

#### Temperature

The V2 value is flagged when

  - the test value exceeds 6.0°C for pressures less than 500 db or
  - the test value exceeds 2.0°C for pressures greater than or equal to 500 db

#### Salinity

The V2 value is flagged when

  - the test value exceeds 0.9 for pressures less than 500 db or
  - the test value exceeds 0.3 for pressures greater than or equal to 500 db

**Action:** Values that fail the spike test should be flagged as bad data. If temperature and salinity values at the same depth both fail, they should be flagged as bad data.

### RTQC10: Bottom Spike test (XBT only)

This is a special version of the spike test, which compares the measurements at the end of the profile to the adjacent measurement. Temperature at the bottom should not differ from the adjacent measurement by more than 1°C. Action: Values that fail the test should be flagged as bad data.

### RTQC11: Gradient test

This test is failed when the difference between vertically adjacent measurements is too steep. The test does not consider the differences in depth, but assumes a sampling that adequately reproduces the temperature and salinity changes with depth. The algorithm is used on both the temperature and salinity profiles:

  Test value = | V2 – (V3 + V1)/2 |

where V2 is the measurement being tested as a spike, and V1 and V3 are the values above and below.

#### Temperature

The V2 value is flagged when

 - the test value exceeds 9.0°C for pressures less than 500 db or
 - the test value exceeds 3.0°C for pressures greater than or equal to 500 db

#### Salinity

The V2 value is flagged when

  - the test value exceeds 1.5 for pressures less than 500 db or
  - the test value exceeds 0.5 for pressures greater than or equal to 500 db

**Action:** Values that fail the test (i.e. value V2) should be flagged as bad data. If temperature and salinity values at the same depth both fail, they should both be flagged as bad data.

### RTQC12: Digit rollover test

Only so many bits are allowed to store temperature and salinity values in a sensor. This range is not always large enough to accommodate conditions that are encountered in the ocean. When the range is exceeded, stored values roll over to the lower end of the range. This rollover should be detected and compensated for when profiles are constructed from the data stream from the instrument. This test is used to ensure the rollover was properly detected.

  - Temperature difference between adjacent depths > 10°C
  - Salinity difference between adjacent depths >5

**Action:** Values that fail the test should be flagged as bad data. If temperature and salinity values at the same depth both fail, both values should be flagged as bad data.

### RTQC13: Stuck value test

This test looks for all measurements of temperature or salinity in a profile being identical. 

**Action:** If this occurs, all of the values of the affected variable should be flagged as bad data. If temperature and salinity are affected, all observed values are flagged as bad data.

### RTQC14: Density inversion

This test uses values of temperature and salinity at the same pressure level and computes the density (sigma0). The algorithm published in UNESCO Technical Papers in Marine Science #44, 1983 should be used. Densities (sigma0) are compared at consecutive levels in a profile, in both directions, i.e. from top to bottom profile, and from bottom to top. Small inversion, below a threshold that can be region dependant, is allowed.

**Action:** from top to bottom, if the density (sigma0) calculated at the greater pressure is less than that calculated at the lesser pressure within the threshold, both the temperature and salinity values should be flagged as bad data. From bottom to top, if the density (sigma0) calculated at the lesser pressure is more than calculated at the greater pressure within the threshold, both the temperature and salinity values should be flagged as bad data.

#### RTQC15: Grey list (Argo only)

This test is implemented to stop the real-time dissemination of measurements from a sensor that is not working correctly. The grey list contains the following 7 items:

 - Float Id
 - Parameter: name of the grey listed parameter
 - Start date: from that date, all measurements for this parameter are flagged as bad or probably bad
 - End date: from that date, measurements are not flagged as bad or probably bad
 - Flag: value of the flag to be applied to all measurements of the parameter
 - Comment: comment from the PI on the problem
 - DAC: data assembly centre for this float

Each DAC manages a black list, sent to the GDACs. The merged black-list is available from the GDACs. The decision to insert a float parameter in the grey list comes from the PI.

#### RTQC16: Gross salinity or temperature sensor drift (Argo only)

This test is implemented to detect a sudden and significant sensor drift. It calculates the average salinity on the last 100 dbar on a profile and the previous good profile. Only measurements with good QC are used.

**Action:** if the difference between the two average values is more than 0.5 psu then all measurements for this parameter are flagged as probably bad data (flag ‘3’). The same test is applied for temperature: if the difference between the two average values is more than 1°C then all measurements for this parameter are flagged as probably bad data (flag ‘3’).

#### RTQC17: Frozen profile test

This test can detect an instrument that reproduces the same profile (with very small deviations) over and over again. Typically the differences between two profiles are of the order of 0.001 for salinity and of the order of 0.01 for temperature.

  A. Derive temperature and salinity profiles by averaging the original profiles to get mean values for each profile in 50 dbar slabs (Tprof, T_previous_prof and Sprof, S_previous_prof). This is necessary because the instruments do not sample at the same level for each profile.

  B. Subtract the two resulting profiles for temperature and salinity to get absolute difference profiles:

    - deltaT = abs(Tprof – T_previous_prof)
    - deltaS = abs(Sprof – S_previous_prof)

  C. Derive the maximum, minimum and mean of the absolute differences for temperature and salinity:

    - mean(deltaT), max(deltaT), min(deltaT)
    - mean(deltaS), max(deltaS), min(deltaS)

  D. To fail the test requires that:

    - max(deltaT) < 0.3
    - min(deltaT) < 0.001
    - mean(deltaT) < 0.02
    - max(deltaS) < 0.3
    - min(deltaS) < 0.001
    - mean(deltaS) < 0.004

**Action:** if a profile fails this test, all measurements for this profile are flagged as bad data (flag ‘4’). If the float fails the test on 5 consecutive cycles, it is inserted in the grey-list.

### RTQC18: Deepest pressure test (Argo only)

This test requires that the profile has pressures that are not higher than DEEPEST_PRESSURE plus 10%. The DEEPEST_PRESSURE value comes from the meta-data file of the instrument.

**Action:** If there is a region of incorrect pressures, all pressures and corresponding measurements should be flagged as bad data.
