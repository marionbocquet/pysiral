# -*- coding: utf-8 -*-

""" """

__all__ = ["auxdata", "bnfunc", "cryosat2", "envisat", "ers", "esa", "icesat", "sentinel3", "classifier", "clocks",
           "config", "datahandler", "errorhandler", "filter", "flag", "frb", "grid",
           "iotools", "l1bdata", "l1preproc", "l2data", "l2preproc", "l2proc", "l3proc",
           "logging", "mask", "orbit", "output", "proj", "retracker", "roi",
           "sit", "surface_type", "validator", "waveform"]


import warnings
warnings.filterwarnings("ignore")

import re
import sys
import yaml
import shutil
from pathlib import Path
from attrdict import AttrDict
from datetime import datetime
from distutils import log, dir_util
log.set_verbosity(log.INFO)
log.set_threshold(log.INFO)
import importlib

# Get version from VERSION in package root
PACKAGE_ROOT_DIR = Path(__file__).absolute().parent
try:
    version_file = open(PACKAGE_ROOT_DIR / "VERSION")
    with version_file as f:
        version = f.read().strip()
except IOError:
    sys.exit("Cannot find VERSION file in package (expected: %s" % version_file)

# Package Metadata
__version__ = version
__author__ = "Stefan Hendricks"
__author_email__ = "stefan.hendricks@awi.de"

# Get the config directory of the package
# NOTE: This approach should work for a local script location of distributed package
PACKAGE_CONFIG_PATH = PACKAGE_ROOT_DIR / "resources" / "pysiral-cfg"


# Get an indication of the location for the pysiral configuration path
# NOTE: In its default version, the text file `PYSIRAL-CFG-LOC` does only contain the
#       string `USER_HOME`. In this case, pysiral will expect the a .pysiral-cfg subfolder
#       in the user home. The only other valid option is an absolute path to a specific
#       directory with the same content as .pysiral-cfg. This was introduced to enable
#       fully encapsulated pysiral installation in virtual environments


# Get the home directory of the current user
CURRENT_USER_HOME_DIR = Path.home()


# Read pysiral config location indicator file
cfg_loc_file = PACKAGE_ROOT_DIR / "PYSIRAL-CFG-LOC"
try:
    with open(cfg_loc_file) as f:
        cfg_loc = f.read().strip()
except IOError:
    sys.exit("Cannot find PYSIRAL-CFG-LOC file in package (expected: %s)" % cfg_loc_file)


# Case 1 (default): pysiral config path is in user home
if cfg_loc == "USER_HOME":

    # NOTE: This is where to expect the pysiral configuration files
    USER_CONFIG_PATH = CURRENT_USER_HOME_DIR / ".pysiral-cfg"

# Case 2: package specific config path
else:
    # This should be an existing path, but in the case it is not, it is created
    USER_CONFIG_PATH = cfg_loc

# Check if pysiral configuration exists in user home directory
# if not: create the user configuration directory
# Also add an initial version of local_machine_def, so that pysiral does not raise an exception
if not USER_CONFIG_PATH.is_dir():
    print("Creating pysiral config directory: %s" % USER_CONFIG_PATH)
    dir_util.copy_tree(PACKAGE_CONFIG_PATH, USER_CONFIG_PATH, verbose=1)
    print("Init local machine def")
    template_filename = PACKAGE_CONFIG_PATH / "templates" / "local_machine_def.yaml.template"
    target_filename = USER_CONFIG_PATH / "local_machine_def.yaml"
    shutil.copy(template_filename, target_filename)


# The config data is now present and can be parsed

class MissionDefinitionCatalogue(object):
    """
    Container for storing and querying information from mission_def.yaml
    """

    def __init__(self, filepath):
        """
        Create a catalogue for all altimeter missions definitions
        :param file_path:
        """

        # Store Argument
        self._filepath = filepath

        # Read the file and store the content
        self._content = None
        with open(self._filepath) as fh:
            self._content = AttrDict(yaml.safe_load(fh))

    @property
    def content(self):
        """
        The content of the definition file as Attrdict
        :return: attrdict.AttrDict
        """
        return self._content

    @property
    def platform_ids(self):
        """
        A list of id's for each platforms
        :return: list with platform ids
        """
        return list(self.content.platforms.keys())


class PysiralPackageConfiguration(object):
    """
    Container for the content of the pysiral definition files
    (in pysiral/configuration) and the local machine definition file
    (local_machine_definition.yaml)
    """

    # Global variables
    _DEFINITION_FILES = {
        "mission": "mission_def.yaml",
        "auxdata": "auxdata_def.yaml",
    }

    _LOCAL_MACHINE_DEF_FILE = "local_machine_def.yaml"

    VALID_SETTING_TYPES = ["proc", "output", "grid"]
    VALID_DATA_LEVEL_IDS = ["l1", "l2", "l2i", "l2p", "l3", None]

    def __init__(self):
        """
        Collect package configuration data from the various definition files and provide an interface
        to pysiral processor, output and grid definition files
        """

        # --- Get information of supported platforms ---
        # The general information for supported radar altimeter missions (mission_def.yaml for historical reasons)
        # provides general metadata for each altimeter missions that can be used to sanity checks
        # and queries for sensor names etc.
        self.mission_def_filepath = USER_CONFIG_PATH / self._DEFINITION_FILES["mission"]
        if not self.mission_def_filepath.is_file():
            error_msg = "Cannot load pysiral package files: \n %s" % self.mission_def_filepath
            print(error_msg)
            sys.exit(1)
        self.mission_def = MissionDefinitionCatalogue(self.mission_def_filepath)

        # read the local machine definition file
        self._read_local_machine_file()

    def _read_config_files(self):
        """
        Read the
        :return:
        """
        for key in self._DEFINITION_FILES.keys():
            filename = USER_CONFIG_PATH / self._DEFINITION_FILES[key]
            setattr(self, key, self.get_yaml_config(filename))

    @staticmethod
    def get_yaml_config(filename):
        """
        Read a yaml file and return it content as an attribute-enabled dictionary
        :param filename: path to the yaml file
        :return: attrdict.AttrDict
        """
        with open(filename) as fileobj:
            settings = AttrDict(yaml.safe_load(fileobj))
        return settings

    def get_mission_info(self, mission):
        mission_info = self.mission[mission]
        if mission_info.data_period.stop is None:
            mission_info.data_period.stop = datetime.utcnow()
        return mission_info

    def get_setting_ids(self, type, data_level=None):
        lookup_directory = self.get_local_setting_path(type, data_level)
        ids, files = self.get_yaml_setting_filelist(lookup_directory)
        return ids

    def get_settings_file(self, type, data_level, setting_id_or_filename):
        """ Returns a processor settings file for a given data level.
        (data level: l2 or l3). The second argument can either be an
        direct filename (which validity will be checked) or an id, for
        which the corresponding file (id.yaml) will be looked up in
        the default directory """

        if type not in self.VALID_SETTING_TYPES:
            return None

        if data_level not in self.VALID_DATA_LEVEL_IDS:
            return None

        # Check if filename
        if setting_id_or_filename.is_file():
            return setting_id_or_filename

        # Get all settings files in settings/{data_level} and its
        # subdirectories
        lookup_directory = self.get_local_setting_path(type, data_level)
        ids, files = self.get_yaml_setting_filelist(lookup_directory)

        # Test if ids are unique and return error for the moment
        if len(set(ids)) != len(ids):
            msg = "Non-unique %-%s setting filename" % (type, str(data_level))
            print("ambiguous-setting-files: %s" % msg)
            sys.exit(1)

        # Find filename to setting_id
        try:
            index = ids.index(setting_id_or_filename)
            return files[index]
        except:
            return None

    def get_yaml_setting_filelist(self, directory, ignore_obsolete=True):
        """ Retrieve all yaml files from a given directory (including
        subdirectories). Directories named "obsolete" are ignored if
        ignore_obsolete=True (default) """
        setting_ids = []
        setting_files = []
        for filepath in directory.rglob("*.yaml"):
            if "obsolete" in filepath.parts:
                continue
            setting_ids.append(filepath.name.replace(".yaml", ""))
            setting_files.append(filepath)
        return setting_ids, setting_files

    def get_local_setting_path(self, type, data_level):
        if type in self.VALID_SETTING_TYPES and data_level in self.VALID_DATA_LEVEL_IDS:
            args = [type]
            if data_level is not None:
                args.append(data_level)
            return Path(USER_CONFIG_PATH, *args)
        else:
            return None

    def _read_local_machine_file(self):
        filename = USER_CONFIG_PATH / self._LOCAL_MACHINE_DEF_FILE
        try:
            local_machine_def = self.get_yaml_config(filename)
        except IOError:
            msg = "local_machine_def.yaml not found (expected: %s)" % filename
            print("local-machine-def-missing: %s" % msg)
            sys.exit(1)
        setattr(self, "local_machine", local_machine_def)

    @property
    def mission_ids(self):
        return self.mission_def.missions

# Create a package configuration object as global variable
psrlcfg = PysiralPackageConfiguration()


def get_cls(module_name, class_name, relaxed=True):
    """ Small helper function to dynamically load classes"""
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        if relaxed:
            return None
        else:
            raise ImportError("Cannot load module: %s" % module_name)
    try:
        return getattr(module, class_name)
    except AttributeError:
        if relaxed:
            return None
        else:
            raise NotImplementedError("Cannot load class: %s.%s" % (module_name, class_name))


