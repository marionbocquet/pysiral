# -*- coding: utf-8 -*-
"""
Created on Tue Jul 07 14:10:34 2015

@author: Stefan

L1bdata is a data container that unifies radar altimeter L1b orbit data
from different missions. It allows subsetting and merging of adjacent
orbit segments. L1bdata can be stored as a netCDF file, thus allowing
faster access to pre-processed subsets of RA orbit data for L2 processing.

The scope of the L1bdata container comprises:

---------
Metadata:
---------

- descriptors of RA source data
- period and geographical location
- processing history (subsetted, merged)
- software version

-------------
Waveform Data
-------------

- waveform echo power
  dimension: (n_records, n_range_bins)
- range for each range bin to the satellite in meters
  dimension: (n_records, n_range_bins)
- radar mode flag for each waveform:
    0: LRM
    1: SAR
    2: SIN
  (this is necessary for merging CryoSat-2 SAR and SIN adjacent orbit segments)
- summarizing flag from source data
    0: invalid
    1: valid
- optional: Additional named flags


----------------------
Time-Orbit Information
----------------------

- timestamp in UTC
- longitude, latitude (of satellite/nadir point)
- altitude (of satellite above WGS84 reference ellipsoid)

All parameter are of dimension (n_records).


-----------------
Range Corrections
-----------------

A list of range corrections (usually from RA source data files). The list
of correction is not predefined, but usally contains range corrections for:

- dry troposphere
- wet troposphere
- ionosphere
- inverse barometric / dynamic atmosphere
- ocean tide
- solid earth tide
- long period tide
- pole tide
- tidal loading

All parameter are of dimension (n_records) and of unit meter


----------
Classifier
----------

A list of optional named parameters that can be used for waveform
classification in the L2 processor. (e.g. stack parameter from the
CryoSat-2 l1b files)

All parameter are of dimension (n_records)


------------
Surface Type
------------



"""

from pysiral.io_adapter import (L1bAdapterCryoSat, L1bAdapterEnvisat)
from pysiral.surface_type import SurfaceType

from collections import OrderedDict
import numpy as np
import os


class Level1bData(object):
    """
    Unified L1b Data Class
    """
    def __init__(self):
        self.info = L1bMetaData()
        self.waveform = L1bWaveforms(self.info)
        self.time_orbit = L1bTimeOrbit(self.info)
        self.correction = L1bRangeCorrections(self.info)
        self.classifier = L1bClassifiers(self.info)
        self.surface_type = SurfaceType()

    def trim_to_subset(self, subset_list):
        """ Create a subset from an indix list """
        data_groups = ["time_orbit", "correction", "classifier", "waveform"]
        for data_group in data_groups:
            content = getattr(self, data_group)
            content.set_subset(subset_list)
        self.update_data_limit_attributes()
        self.info.set_attribute("is_orbit_subset", True)
        self.info.set_attribute("n_records", len(subset_list))

    def apply_range_correction(self, correction):
        """  Apply range correction """
        range_delta = self.correction.get_parameter_by_name(correction)
        if range_delta is None:
            # TODO: raise warning
            return
        self.waveform.add_range_delta(range_delta)

    def extract_subset(self, subset_list):
        """ Same as trim_to_subset, except returns a new l1bdata instance """
        l1b = np.copy(self)
        l1b.trim_to_subset(subset_list)
        return l1b

    def update_data_limit_attributes(self):
        """
        Set latitude/longitude and timestamp limits in the metadata container
        """
        info = self.info
        info.set_attribute("lat_min", np.nanmin(self.time_orbit.latitude))
        info.set_attribute("lat_max", np.nanmax(self.time_orbit.latitude))
        info.set_attribute("lon_min", np.nanmin(self.time_orbit.longitude))
        info.set_attribute("lon_max", np.nanmax(self.time_orbit.longitude))
        info.set_attribute("start_time", self.time_orbit.timestamp[0])
        info.set_attribute("stop_time", self.time_orbit.timestamp[-1])

    @property
    def n_records(self):
        return self.info.n_records


class L1bConstructor(Level1bData):
    """
    Class to be used to construct a L1b data object from any mission
    L1b data files
    """

    _SUPPORTED_MISSION_LIST = ["cryosat2", "envisat"]

    def __init__(self, config):
        super(L1bConstructor, self).__init__()
        self._config = config
        self._mission = None
        self._mission_options = None
        self._filename = None

    @property
    def mission(self):
        return self._mission

    @mission.setter
    def mission(self, value):
        if value in self._SUPPORTED_MISSION_LIST:
            self._mission = value
        else:
            # XXX: An ErrorHandler is needed here
            raise ValueError("Unsupported mission type")
        # Get mission default options
        self._mission_options = self._config.get_mission_defaults(value)

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        if os.path.isfile(value):
            self._filename = value
        else:
            # XXX: An ErrorHandler is needed here
            raise IOError("Not a valid path")

    def set_mission_options(self, **kwargs):
        self._mission_options = kwargs

    def construct(self):
        """ Parse the file and construct the L1bData object """
        adapter = get_l1b_adapter(self._mission)(self._config)
        adapter.filename = self.filename
        adapter.construct_l1b(self)


class L1bMetaData(object):
    """
    Container for L1B Metadata information
    (see property attribute_list for a list of attributes)
    """

    _attribute_list = [
        "mission", "mission_data_version", "mission_data_source",
        "n_records", "orbit", "cycle", "is_orbit_subset", "is_merged_orbit",
        "start_time", "stop_time", "subset_region_name",
        "lat_min", "lat_max", "lon_min", "lon_max", "pysiral_version"]

    def __init__(self):
        # Init all fields
        for field in self.attribute_list:
            setattr(self, field, None)
        # Set some fields to False (instead of none)
        self.is_orbit_subset = False
        self.is_merged_orbit = False
        self.n_records = -1

    def __repr__(self):
        output = "pysiral.L1bdata object:\n"
        for field in self.field_list:
            output += "%22s: %s" % (field, getattr(self, field))
            output += "\n"
        return output

    @property
    def attribute_list(self):
        return self._attribute_list

    @property
    def attdict(self):
        """ Return attributes as dictionary (e.g. for netCDF export) """
        attdict = {}
        for field in self.attribute_list:
            attdict[field] = getattr(self, field)
        return attdict

    def set_attribute(self, tag, value):
        if tag not in self.attribute_list:
            raise ValueError("Unknown attribute: ", tag)
        setattr(self, tag, value)

    def check_n_records(self, n_records):
        # First time a data set is set: Store number of records as reference
        if self.n_records == -1:
            self.n_records = n_records
        else:  # n_records exists: verify consistenty
            if n_records == self.n_records:  # all good
                pass
            else:  # raise Erro
                raise ValueError("n_records mismatch, len must be: ",
                                 str(self.n_records))


class L1bTimeOrbit(object):
    """ Container for Time and Orbit Information of L1b Data """
    def __init__(self, info):
        self._info = info  # Pointer to metadata container
        self._timestamp = None
        self._longitude = None
        self._latitude = None
        self._altitude = None

    @property
    def longitude(self):
        return self._longitude

    @property
    def latitude(self):
        return self._latitude

    @property
    def altitude(self):
        return self._altitude

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        self._info.check_n_records(len(value))
        self._timestamp = value

    @property
    def parameter_list(self):
        return ["timestamp", "longitude", "latitude", "altitude"]

    @property
    def dimdict(self):
        """ Returns dictionary with dimensions"""
        dimdict = OrderedDict([("n_records", len(self._timestamp))])
        return dimdict

    def set_position(self, longitude, latitude, altitude):
        # Check dimensions
        self._info.check_n_records(len(longitude))
        self._info.check_n_records(len(latitude))
        self._info.check_n_records(len(altitude))
        # All fine => set values
        self._longitude = longitude
        self._latitude = latitude
        self._altitude = altitude

    def set_subset(self, subset_list):
        for parameter in self.parameter_list:
            data = getattr(self, "_"+parameter)
            data = data[subset_list]
            setattr(self,  "_"+parameter, data)


class L1bRangeCorrections(object):
    """ Container for Range Correction Information """

    def __init__(self, info):
        self._info = info  # Pointer to Metadate object
        self._parameter_list = []

    def set_parameter(self, tag, value):
        self._info.check_n_records(len(value))
        setattr(self, tag, value)
        self._parameter_list.append(tag)

    @property
    def parameter_list(self):
        return self._parameter_list

    @property
    def dimdict(self):
        """ Returns dictionary with dimensions"""
        dimdict = OrderedDict([("n_records",
                                len(self.get_parameter_by_index(0)))])
        return dimdict

    def get_parameter_by_index(self, index):
        name = self._parameter_list[index]
        return getattr(self, name), name

    def get_parameter_by_name(self, name):
        try:
            return getattr(self, name)
        except:
            return None

    def set_subset(self, subset_list):
        for parameter in self.parameter_list:
            data = getattr(self, parameter)
            data = data[subset_list]


class L1bClassifiers(object):
    """ Containier for parameters that can be used as classifiers """

    def __init__(self, info):
        self._info = info  # Pointer to Metadate object
        # Make a pre-selection of different classifier types
        self._list = {
            "surface_type": [],
            "warning": [],
            "error": []}

    def add(self, value, name, classifier_type="surface_type"):
        """ Add a parameter for a given classifier type """
        setattr(self, name, np.array(value))
        self._list[classifier_type].append(name)

    @property
    def parameter_list(self):
        parameter_list = []
        for key in self._list.keys():
            parameter_list.extend(self._list[key])
        return parameter_list

    @property
    def n_records(self):
        parameter_list = self.parameter_list
        if len(parameter_list) == 0:
            return 0
        else:
            return len(getattr(self, parameter_list[0]))

    @property
    def dimdict(self):
        """ Returns dictionary with dimensions"""
        dimdict = OrderedDict([("n_records", self.n_records)])
        return dimdict

    def set_subset(self, subset_list):
        for parameter in self.parameter_list:
            data = getattr(self, parameter)
            data = data[subset_list]


class L1bWaveforms(object):
    """ Container for Echo Power Waveforms """

    _valid_radar_modes = ["lrm", "sar", "sin"]
    _parameter_list = ["power", "range", "radar_mode", "is_valid"]
    _attribute_list = ["echo_power_unit"]

    def __init__(self, info):
        self._info = info  # Pointer to Metadate object
        # Attributes
        self.echo_power_unit = None
        # Parameter
        self._power = None
        self._range = None
        self._radar_mode = None
        self._is_valid = None

    @property
    def power(self):
        return self._power

    @property
    def range(self):
        return self._range

    @property
    def radar_mode(self):
        return self._radar_mode

    @property
    def is_valid(self):
        return self._is_valid

    @property
    def parameter_list(self):
        return self._parameter_list

    @property
    def n_range_bins(self):
        return self._get_wfm_shape(1)

    @property
    def n_records(self):
        return self._get_wfm_shape(0)

    @property
    def dimdict(self):
        """ Returns dictionary with dimensions"""
        shape = np.shape(self._power)
        dimdict = OrderedDict([("n_records", shape[0]), ("n_bins", shape[1])])
        return dimdict

    def set_waveform_data(self, power, range, radar_mode):
        # Validate input
        if power.shape != range.shape:
            raise ValueError("power and range must be of same shape",
                             power.shape, range.shape)
        if len(power.shape) != 2:
            raise ValueError("power and range arrays must be of dimension" +
                             " (n_records, n_bins)")
        # Validate number of records
        self._info.check_n_records(power.shape[0])
        # Assign values
        self._power = power
        self._range = range
        # Create radar mode arrays
        if type(radar_mode) is str and radar_mode in self._valid_radar_modes:
            self._radar_mode = np.repeat(radar_mode, self.n_records)
        else:
            raise ValueError("Invalid radar_mode: ", radar_mode)
        # Set valid flag (assumed to be valid for all waveforms)
        # Flag can be set separately using the set_valid_flag method
        if self._is_valid is None:
            self._is_valid = np.ones(shape=(self.n_records), dtype=bool)

    def set_valid_flag(self, valid_flag):
        # Validate number of records
        self._info.check_n_records(len(valid_flag))
        self._is_valid = valid_flag

    def set_subset(self, subset_list):
        self._power = self._power[subset_list, :]
        self._range = self._range[subset_list, :]
        self._radar_mode = self._radar_mode[subset_list]
        self._is_valid = self._is_valid[subset_list]

    def add_range_delta(self, range_delta):
        range_delta_reshaped = np.repeat(range_delta, self.n_range_bins)
        range_delta_reshaped = range_delta_reshaped.reshape(
            self.n_records, self.n_range_bins)
        self._range += range_delta_reshaped

    def _get_wfm_shape(self, index):
        shape = np.shape(self._power)
        return shape[index]


def get_l1b_adapter(mission):
    """ XXX: Early development state only """
    if mission == "cryosat2":
        return L1bAdapterCryoSat
    if mission == "envisat":
        return L1bAdapterEnvisat
