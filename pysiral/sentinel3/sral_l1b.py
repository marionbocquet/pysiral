# -*- coding: utf-8 -*-

from pysiral.sentinel3.iotools import get_sentinel3_sral_l1_from_l2
from pysiral.errorhandler import FileIOErrorHandler
from pysiral.path import folder_from_filename

import numpy as np
import os


class Sentinel3SRALL1b(object):

    def __init__(self, raise_on_error=False):

        # Error Handling
        self._init_error_handling(raise_on_error)
        self.product_info = Sentinel3SRALProductInfo()
        self._radar_mode = "sar"
        self._filename = None
        self.n_records = 0
        self.range_bin_width = 0.234212857813 * 2.
        self.nominal_tracking_bin = 60
        self._xml_header_file = "xfdumanifest.xml"
        self._xml_metadata_object_index = {
            "processing": 0,
            "acquisitionPeriod": 1,
            "platform": 2,
            "generalProductInformation": 3,
            "measurementOrbitReference": 4,
            "measurementQualityInformation": 5,
            "measurementFrameSet": 6,
            "sralProductInformation": 7}

    def parse(self):
        self._parse_xml_header()
        self._parse_measurement_nc()

    def _parse_xml_header(self):
        """
        Parse the Sentinel-3 XML header file and extract key attributes
        for filtering
        """

        filename_header = os.path.join(
            folder_from_filename(self.filename), self._xml_header_file)
        self._xmlh = parse_sentinel3_l1b_xml_header(filename_header)

        # Extract Metadata
        metadata = self._xmlh["metadataSection"]["metadataObject"]

        # Extract Product Info
        index = self._xml_metadata_object_index["sralProductInformation"]
        sral_product_info = metadata[index]["metadataWrap"]["xmlData"]
        sral_product_info = sral_product_info["sralProductInformation"]

        # Save in
        sar_mode_percentage = sral_product_info["sral:sarModePercentage"]
        self.product_info.sar_mode_percentage = float(sar_mode_percentage)

        open_ocean_percentage = sral_product_info["sral:openOceanPercentage"]
        self.product_info.open_ocean_percentage = float(open_ocean_percentage)

        # print "sar_mode_percentage = %.1f" % int(sar_mode_percentage)

    def _parse_measurement_nc(self):

        from pysiral.iotools import ReadNC
        self._validate()

        # Read the L2 netCDF file
        self.nc = ReadNC(self.filename)

        # Additionally read the L1 netCDF file for additional
        # waveform and stack parameters
        # L1 filename cannot be directly derived from L2 filename, thus
        # it has to be estimated and looked for potential matches
        # -> if l1nc_filename is None, corresponding L1 file does not exist
#        l1nc_filename = get_sentinel3_sral_l1_from_l2(self.filename)
#        if l1nc_filename is not None:
#
#            self.l1nc = ReadNC(l1nc_filename)
#
#            if len(self.l1nc.time_l1b_echo_sar_ku) == 0:
#                print ". Warning - l1 ku data empty"
#            else:
#                # Verify time overlap between l1 and l2 data
#                start_l1 = np.amin(self.l1nc.time_l1b_echo_sar_ku)
#                end_l1 = np.amax(self.l1nc.time_l1b_echo_sar_ku)
#
#                start_l2 = np.amin(self.nc.time_20_ku)
#                end_l2 = np.amax(self.nc.time_20_ku)
#
#                l1_l2_overlap = (start_l1 <= end_l2) and (end_l1 >= start_l2)
#                if not l1_l2_overlap:
#                    raise IOError("L1/L2 files - no overlap")
#        else:
#            print ". Warning - no l1 file match"

#        for attribute in self.nc.attributes:
#            print "attribute: %s = %s" % (
#                attribute, str(getattr(self.nc, attribute)))
#        for parameter in self.nc.parameters:
#            print parameter, getattr(self.nc, parameter).shape

    def get_status(self):
        # XXX: Not much functionality here
        return False

    def post_processing(self):
        """
        The SGDR data structure needs to be streamlined, so that it
        is easy to grab the relevant parameters as indiviual arrays
        """
        self._prepare_waveform_power_and_range()

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, filename):
        """ Save and validate filenames for header and product file """
        # Test if valid file first
        self._error.file_undefined = not os.path.isfile(filename)
        if self._error.file_undefined:
            return
        self._filename = filename

    @property
    def radar_mode(self):
        return self._radar_mode

    def _init_error_handling(self, raise_on_error):
        self._error = FileIOErrorHandler()
        self._error.raise_on_error = raise_on_error
        self._error.file_undefined = True

    def _prepare_waveform_power_and_range(self):
        """
        reforms the waveform to computes the corresponding range for each
        range bin
        """

        # self.wfm_power = self.nc.waveform_20_ku
        self.wfm_power = self.nc.i2q2_meas_ku_l1b_echo_sar_ku
        n_records, n_range_bins = np.shape(self.wfm_power)
        self.n_records = n_records
        # Get the window delay
        # "The tracker_range_20hz is the range measured by the onboard tracker
        #  as the window delay, corrected for instrumental effects and
        #  CoG offset"
        tracker_range_20hz = self.nc.range_ku_l1b_echo_sar_ku


        self.wfm_range = np.ndarray(shape=self.wfm_power.shape,
                                    dtype=np.float32)
        range_bin_index = np.arange(n_range_bins)
        for record in np.arange(n_records):
            self.wfm_range[record, :] = tracker_range_20hz[record] + \
                (range_bin_index*self.range_bin_width) - \
                (self.nominal_tracking_bin*self.range_bin_width)

    def _validate(self):
        pass


class Sentinel3SRALProductInfo(object):

    def __init__(self):

        self.sar_mode_percentage = None
        self.open_ocean_percentage = None
        self.start_time = None
        self.stop_time = None
        self.lat_min = None
        self.lat_max = None
        self.lon_min = None
        self.lon_max = None


def parse_sentinel3_l1b_xml_header(filename):
    """
    Reads the XML header file of a Sentinel 3 L1b Data set
    and returns the contents as an OrderedDict
    """
    import xmltodict
    with open(filename) as fd:
        content_odereddict = xmltodict.parse(fd.read())
    return content_odereddict[u'xfdu:XFDU']