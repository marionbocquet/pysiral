# -*- coding: utf-8 -*-

"""
This is the SAMOSA+ retracker variant that uses `samosa-waveform-model`, a
re-implementation of the SAMPy package. The objective of the re-implementation
is better performance (code optimization and multiprocessing) and a greater flexibility
for retracking settings (sub-waveform retracking, limiting parameters of the fit)

NOTES:
======

List of fit modes:

- single fit - all parameters
- double fit - all parameters
- single fit - fixed mean square slope

All variants with options for single core and multi-processing

TODO: Refine workflow (first extraction of all variables and than retracking in another class)?
TODO: Implement sigma0/windspeed computation (per switch in config file)
TODO: Implement waveform mask (sub-waveform tracking)
TODO: Computation of residuals should be configurable method
TODO: Add thermal noise to computation of residuals
TODO: Combined method to extract first guess and parameter bounds from waveform
TODO: Allow to add waveforms to l2 data

Workflow

1. Colocate all input variables in dedicated list of input data classes
2. Initialize specified fit method (one class per fit method?)
3. Process waveforms with/without multiprocessing
4. Organize output

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

import multiprocessing
import numpy as np
from functools import partial

from loguru import logger
from typing import Tuple, List, Dict, Optional, Any, Literal, Callable, get_args
from dataclasses import dataclass

from scipy.optimize import least_squares, OptimizeResult
from scipy.signal import argrelmin

from samosa_waveform_model import (
    ScenarioData, SensorParameters, SAMOSAWaveformModel, PlatformLocation, SARParameters,
    WaveformModelParameters
)
from samosa_waveform_model.dataclasses import WaveformModelOutput

from pysiral import psrlcfg
from pysiral.core.config import RadarModes
from pysiral.retracker import BaseRetracker
from pysiral.l1data import Level1bData
from pysiral.l2data import Level2Data

# NOTE:
# There are three fit methods that may be chosen for different surface types.
# E.g., for open ocean it has been standard practice to have a first fit only
# for mean square slope to estimate mean square slope and then a second fit
# with invariable mean square slope ("two_fits_mss_first_swh_second").
# This implementation also supports fitting all parameters in single fit
# ("single_fit_mss_swh") of only of significant wave height with mean square
# slope sourced from waveform parameter ("single_fit_mss_preset").
# (Epoch is always fitted).
VALID_METHOD_LITERAL = Literal["samosap_standard", "samosap_specular"]
VALID_METHODS = get_args(VALID_METHOD_LITERAL)

DEFAULT_FIT_KWARGS = dict(
    loss="linear",
    method="trf",
    ftol=1e-2,
    xtol=2*1e-3,
    gtol=1e-2,
    max_nfev=None,
)

SWH_FIRST_GUESS = dict(
    lead=0.0,
    sea_ice=2.0
)

SWH_DEFAULT_BOUNDS = dict(
    lead=[-0.2, 0.2],    # Significant wave height does not affect leads very much, mss more important
    sea_ice=[-0.5, 10]   # default values for sampy
)

NU_FIRST_GUESS = dict(
    lead=2.0e6,
    sea_ice=1.0e4
)

NU_DEFAULT_BOUNDS = dict(
    lead=[1e6, 1e9],
    sea_ice=[1e3, 1e7]
)


@dataclass
class NormedWaveform:
    """
    Data container for the waveform fit
    """
    power: np.ndarray
    range_bins: np.ndarray
    tau: np.ndarray
    scaling_factor: float
    radar_mode_flag: int
    window_delay: float
    transmit_power: float
    look_angles: np.ndarray
    surface_type: str
    surface_class: str
    thermal_noise: float = 0.0
    first_maximum_index: int = None


@dataclass
class WaveformFitData:
    """
    Dataclass that contains all data from l1b and l2 data objects
    required for the SAMOSA waveform model fit
    """
    idx: int
    waveform_data: NormedWaveform
    scenario_data: ScenarioData


@dataclass
class SAMOSAWaveformFitResult:
    """

    """
    epoch: float = np.nan
    retracker_range: float = np.nan
    retracker_range_standard_error: float = np.nan
    significant_wave_height: float = np.nan
    significant_wave_height_standard_error: float = np.nan
    mean_square_slope: float = np.nan
    mean_square_slope_standard_error: float = np.nan
    thermal_noise: float = np.nan
    misfit: float = np.nan
    fit_mode: str = "n/a"
    waveform: np.ndarray = None
    waveform_model: np.ndarray = None
    misfit_sub_waveform: float = None
    number_of_model_evaluations: int = -1
    fit_return_status: int = None
    sigma0: float = 0.0

    @property
    def is_sub_waveform_fit(self) -> bool:
        return self.misfit_sub_waveform is not None

class SAMOSAWaveformFit(object):
    """
    Class for fitting the SAMOSA waveform model to a single
    waveform.

    Usage:

        fit_cls = SAMOSAWaveformFit(scenario_data, normed_waveform, ...)
        fit_result = scipy.optimize.least_squares(fit_cls.fit_func, ...)
    """

    def __init__(
            self,
            scenario_data: ScenarioData,
            normed_waveform: NormedWaveform,
            waveform_model: Optional[SAMOSAWaveformModel] = None,
            sub_waveform_mask: Optional[np.ndarray] = None,
            waveform_model_kwargs: Optional[Dict] = None,
            step1_fixed_nu_value: float = 0.0,
            step2_fixed_swh_value: float = 0.0,
            method: str = None
    ) -> None:

        # Default waveform model kwargs to empty dict
        waveform_model_kwargs = waveform_model_kwargs if isinstance(waveform_model_kwargs, dict) else {}

        # initialize waveform mode if no instance has been provided.
        self.samosa_waveform_model = (
            waveform_model if isinstance(waveform_model, SAMOSAWaveformModel) else
            SAMOSAWaveformModel(scenario_data, **waveform_model_kwargs)
        )
        self.normed_waveform = normed_waveform

        # The first fit step in the SAMOSA+ retracker uses a fixed nu value
        self.step1_fixed_nu_value = step1_fixed_nu_value
        self.step2_fixed_swh_value = step2_fixed_swh_value

        # The second fit step in the SAMOSA+ retracker uses a fixed swh value

        # Mask is empty by default (mask=True -> point does not generate a residual value)
        self.sub_waveform_mask = sub_waveform_mask
        self.method = method

    def fit_func(self, fit_args: List[float], *_) -> np.ndarray:
        """
        Fit function for the least square algorithm. Computes and returns
        the residual between waveform and SAMOSA+ waveform model.

        Also stores the last waveform model in class instance, which then
        can be accessed later after the optimization is complete.

        :param fit_args: Waveform model parameter

        :return: Difference between waveform and waveform model.
            be minimized by least square process.
        """
        epoch, significant_wave_height, nu = fit_args
        waveform_model = get_model_from_args(
            self.samosa_waveform_model,
            [epoch * 1e-9, significant_wave_height, nu],
            thermal_noise=self.normed_waveform.thermal_noise
        )

        # Compute the residuals
        pr = waveform_model.power  # sampy notation
        # power_scale = np.nanmax(pr) / self.normed_waveform.power[self.normed_waveform.first_maximum_index]
        waveform_model_scaled_power = (pr / np.nanmax(pr)) + self.normed_waveform.thermal_noise
        residuals = waveform_model_scaled_power - self.normed_waveform.power

        if self.sub_waveform_mask is None:
            return residuals

        # Filter residuals
        x = np.arange(residuals.size)
        valid_entries = np.logical_not(self.sub_waveform_mask)
        residuals_masked = np.interp(x, x[valid_entries], residuals[valid_entries])

        # # x = np.arange(256)
        # plt.figure("residuals")
        # plt.plot(waveform_model.tau * 1e9, residuals)
        # plt.plot(waveform_model.tau * 1e9, residuals_masked)
        #
        # plt.figure("waveforms")
        # plt.plot(waveform_model.tau * 1e9, waveform_model_scaled_power)
        # plt.scatter(waveform_model.tau[valid_entries] * 1e9, self.normed_waveform.power[valid_entries])
        # plt.show()

        return residuals_masked

    def fit_func_samosap_standard_step1(self, fit_args: List[float], *_) -> np.ndarray:
        """
        Fit of the first step in the SAMOSA+ fitting process.

        :param fit_args:
        :param _:

        :return: Masked residual vector
        """

        epoch, significant_wave_height, power_scale = fit_args
        nu = self.step1_fixed_nu_value  # The default value for nu (1/mss) is 0 in sampy 1/np.inf -> 0.0
        waveform_model = get_model_from_args(
            self.samosa_waveform_model,
            [epoch * 1e-9, significant_wave_height, nu],
            thermal_noise=self.normed_waveform.thermal_noise
        )

        # Compute the residuals
        pr = waveform_model.power  # sampy notation
        waveform_model_scaled_power = power_scale * (pr / np.nanmax(pr)) + self.normed_waveform.thermal_noise
        residuals = waveform_model_scaled_power - self.normed_waveform.power

        if self.sub_waveform_mask is None:
            return residuals

        # Filter residuals
        x = np.arange(residuals.size)
        valid_entries = np.logical_not(self.sub_waveform_mask)
        residuals_masked = np.interp(x, x[valid_entries], residuals[valid_entries])

        return residuals_masked

    def fit_func_samosap_standard_step2(self, fit_args: List[float], *_) -> np.ndarray:
        """
        Fit of the second step in the SAMOSA+ fitting process.

        :param fit_args:
        :param _:

        :return: Masked residual vector
        """

        epoch_ns, nu, power_scale = fit_args
        significant_wave_height = self.step2_fixed_swh_value
        waveform_model = get_model_from_args(
            self.samosa_waveform_model,
            [epoch_ns * 1e-9, significant_wave_height, nu]
        )

        # Compute the residuals
        pr = waveform_model.power  # sampy notation
        waveform_model_scaled_power = power_scale * (pr / np.nanmax(pr)) + self.normed_waveform.thermal_noise
        residuals = waveform_model_scaled_power - self.normed_waveform.power

        if self.sub_waveform_mask is None:
            return residuals

        # Filter residuals
        x = np.arange(residuals.size)
        valid_entries = np.logical_not(self.sub_waveform_mask)
        residuals_masked = np.interp(x, x[valid_entries], residuals[valid_entries])

        # x = np.arange(256)
        # plt.figure()
        # plt.plot(waveform_model.tau * 1e9, residuals)
        # plt.plot(waveform_model.tau * 1e9, residuals_masked)
        #
        # plt.figure()
        # plt.plot(waveform_model.tau * 1e9, waveform_model_scaled_power)
        # plt.scatter(waveform_model.tau[valid_entries] * 1e9, self.normed_waveform.power[valid_entries])
        # plt.show()

        return residuals_masked


class SAMOSAWaveformCollectionFit(object):
    """
    Handles the SAMOSA waveform model fitting of a waveform
    collection. Can be configured to use several fitting
    strategies with or without multiprocessing.

    The notation convention for class methods is `_fit_{fit_method}{_mp: use_multiprocessing: True}`.
    Therefore, each fit method needs to class methods: One for single process and one for parallel
    processes.

    :param fit_method: Name of the fitting method (choices: "mss_swh", "mss_preset", "mss_seperate")
    :param predictor_method:
    :param use_multiprocessing: Flag if to use multiprocessing (default: False)
    :param num_processes: Number of multiprocessing workers
    :param predictor_kwargs: Keyword arguments for the parameter predictor class
    :param least_squares_kwargs: Keyword arguments for scipy.optimize.least_squares
    """

    def __init__(
        self,
        fit_method: VALID_METHOD_LITERAL,
        predictor_method: str,
        use_multiprocessing: bool = False,
        num_processes: Optional[int] = None,
        predictor_kwargs: Optional[Dict] = None,
        least_squares_kwargs: Optional[Dict] = None,
    ) -> None:
        self.fit_method = fit_method
        self.predictor_method = predictor_method
        self.least_squares_kwargs = DEFAULT_FIT_KWARGS if least_squares_kwargs is None else least_squares_kwargs
        self.predictor_kwargs = {} if predictor_kwargs is None else predictor_kwargs
        self.use_multiprocessing = use_multiprocessing
        self.num_processes = num_processes
        self.fit_method = self._select_fit_method()

    def fit(self, waveform_collection: List[WaveformFitData]) -> List[SAMOSAWaveformFitResult]:
        """
        Main entry method for the waveform collection fitting.

        :param waveform_collection: Waveform collection data

        :return: List of waveform fit results
        """
        return self.fit_method(waveform_collection)

    def _select_fit_method(self) -> Callable:
        """
        Selects to appropiate fit method based on class configuration.

        :raises AttributeError": Invalid configuration

        :return: callable fit method
        """
        mp_string = "_mp" if self.use_multiprocessing else ""
        fit_method_name = f"_fit_{self.fit_method}{mp_string}"
        return getattr(self, fit_method_name)

    def _fit_samosap_standard(self, waveform_collection: List[WaveformFitData]) -> List[SAMOSAWaveformFitResult]:
        """
        Computes Samosa waveform model fit in the main process (no multiprocessing)

        :param waveform_collection: List of fit input

        :return: List of fit outputs
        """
        return [
            samosa_fit_samosap_standard(
                fit_data,
                least_squares_kwargs=self.least_squares_kwargs,
                predictor_kwargs=self.predictor_kwargs
            )
            for fit_data in waveform_collection
        ]

    def _fit_samosap_standard_mp(self, waveform_collection: List[WaveformFitData]) -> List[SAMOSAWaveformFitResult]:
        """
        Computes Samosa waveform model fit with multiprocessing

        :param waveform_collection: List of fit input

        :return: List of fit outputs
        """
        pool = multiprocessing.Pool(self.num_processes)
        fit_func = partial(
            samosa_fit_samosap_standard,
            least_squares_kwargs=self.least_squares_kwargs,
            predictor_kwargs=self.predictor_kwargs
        )
        fit_results = pool.map(fit_func, waveform_collection)
        pool.close()
        pool.join()
        return fit_results

    # def _fit_mss_swh(self, waveform_collection: List[WaveformFitData]) -> List[SAMOSAWaveformFitResult]:
    #     """
    #     Computes Samosa waveform model fit in the main process (no multiprocessing)
    #
    #     :param waveform_collection: List of fit input
    #
    #     :return: List of fit outputs
    #     """
    #     return [
    #         samosa_fit_swh_mss(
    #             fit_data,
    #             least_squares_kwargs=self.least_squares_kwargs,
    #             predictor_kwargs=self.predictor_kwargs
    #         )
    #         for fit_data in waveform_collection
    #     ]

    # def _fit_mss_swh_mp(self, waveform_collection: List[WaveformFitData]) -> List[SAMOSAWaveformFitResult]:
    #     """
    #     Computes Samosa waveform model fit with multiprocessing
    #
    #     :param waveform_collection: List of fit input
    #
    #     :return: List of fit outputs
    #     """
    #     pool = multiprocessing.Pool(self.num_processes)
    #     fit_func = partial(
    #         samosa_fit_swh_mss,
    #         least_squares_kwargs=self.least_squares_kwargs,
    #         predictor_kwargs=self.predictor_kwargs
    #     )
    #     fit_results = pool.map(fit_func, waveform_collection)
    #     pool.close()
    #     pool.join()
    #     return fit_results


class SAMOSAModelParameterPrediction(object):
    """
    Class to get initial guess and bounds of SAMOSA waveform fit parameter
    """

    def __init__(
        self,
        method: VALID_METHOD_LITERAL,
        surface_type: str,
        bounds: Dict[str, Tuple[float, float]],
        earliest_epoch: float = 0.0,
    ):
        self.surface_type = surface_type
        self.earliest_epoch = earliest_epoch
        self.method = method
        self.bounds = bounds
        self.first_guess_method = getattr(self, f"_get_first_guess_{method}")
        self.bounds_method = getattr(self, f"_get_bounds_{method}")

    def get(self, waveform: NormedWaveform, **kwargs) -> Tuple[Tuple, Tuple, Tuple]:
        first_guess = self.first_guess_method(waveform, **kwargs)
        lower_bounds, upper_bounds = self.bounds_method(waveform.tau, first_guess, **kwargs)
        return first_guess, lower_bounds, upper_bounds

    @staticmethod
    def _get_first_guess_fit_mss_swh(waveform: NormedWaveform) -> Tuple:
        """
        Estimate the first guess

        :param waveform:

        :return:
        """
        epoch_first_guess = waveform.tau[waveform.first_maximum_index] * 1e9
        swh_first_guess = SWH_FIRST_GUESS[waveform.surface_type]
        nu_first_guess = NU_FIRST_GUESS[waveform.surface_type]
        return epoch_first_guess, swh_first_guess, nu_first_guess

    def _get_bounds_fit_mss_swh(self, tau, first_guess) -> Tuple[Tuple, Tuple]:
        epoch_bounds_range_gate = get_epoch_bounds(tau, 0.02, 0.8)
        tau_fmi_offset = (first_guess[0] * 1e-9) + 20. * (tau[1]-tau[0])
        epoch_bounds = (
            epoch_bounds_range_gate[0],
            np.min([np.max(tau), tau_fmi_offset])
        )
        swh_bounds = self.bounds["swh"]
        nu_bounds = self.bounds["nu"]
        lower_bounds = epoch_bounds[0] * 1e9, float(swh_bounds[0]), float(nu_bounds[0])
        upper_bounds = epoch_bounds[1] * 1e9, float(swh_bounds[1]), float(nu_bounds[1])
        return lower_bounds, upper_bounds

    @staticmethod
    def _get_first_guess_samosap_standard(waveform: NormedWaveform, mode: int = 1) -> Tuple:
        """
        Estimate the first guess

        First fit step (mode=1): [epoch, significant_waveheight, amplitude scale]
        Seconds fit step (mode=2): [epoch, mean square slope, amplitude scale]

        :param waveform:

        :return:
        """
        epoch_first_guess = waveform.tau[waveform.first_maximum_index]
        nu_first_guess = NU_FIRST_GUESS[waveform.surface_type]
        if mode == 1:
            return epoch_first_guess * 1e9, SWH_FIRST_GUESS[waveform.surface_type], 1
        elif mode == 2:
            return epoch_first_guess * 1e9,  nu_first_guess, 1
        else:
            raise ValueError(f"mode={mode} not in [1, 2]")

    def _get_bounds_samosap_standard(self, tau, first_guess, mode: int = 1) -> Tuple[Tuple, Tuple]:
        epoch_bounds_range_gate = get_epoch_bounds(tau, 0.02, 0.8)
        tau_fmi_offset = (first_guess[0] * 1e-9) + 20. * (tau[1]-tau[0])
        epoch_bounds = (
            epoch_bounds_range_gate[0],
            np.min([np.max(tau), tau_fmi_offset])
        )
        swh_bounds = self.bounds["swh"]
        nu_bounds = self.bounds["nu"]
        pu_bounds = [0.2, 1.5]  # TODO: copied over from sampy
        if mode == 1:
            lower_bounds = epoch_bounds[0] * 1e9, float(swh_bounds[0]), pu_bounds[0]
            upper_bounds = epoch_bounds[1] * 1e9, float(swh_bounds[1]), pu_bounds[1]
        elif mode == 2:
            lower_bounds = epoch_bounds[0] * 1e9, float(nu_bounds[0]), pu_bounds[0]
            upper_bounds = epoch_bounds[1] * 1e9, float(nu_bounds[1]), pu_bounds[1]
        else:
            raise ValueError(f"mode={mode} not in [1, 2]")
        return lower_bounds, upper_bounds


class SAMOSAPlusRetracker(BaseRetracker):

    def __init__(self) -> None:
        super(SAMOSAPlusRetracker, self).__init__()
        self._retracker_params = {}

    def l2_retrack(
            self,
            rng: np.ndarray,
            wfm: np.ndarray,
            indices: np.ndarray,
            radar_mode: np.ndarray,
            is_valid: np.ndarray
    ) -> None:
        """
         API method for the retracker interface in the Level-2 processor.

        :param rng: The range per waveform samples for all waveforms in the Level-1 data object
        :param wfm: All waveforms in the Level-1 data object
        :param indices: List of waveforms for the retracker
        :param radar_mode: Flag indicating the radar mode for all waveforms
        :param is_valid: Error flag for all waveforms

        :return: None (Output is added to the instance)
        """

        # Run the retracker
        # NOTE: Output is a SAMOSAFitResult dataclass for each index in indices.
        fit_results = self._samosa_plus_retracker(rng, wfm, indices, radar_mode)

        # Store retracker properties (including range)
        self._store_retracker_properties(fit_results, indices)

        # Set/compute uncertainty
        # self._set_range_uncertainty()

        # Add range biases (when set in config file)
        # self._set_range_bias(radar_mode)

        # Add retracker parameters to the l2 data object
        self._l2_register_retracker_parameters()

    def create_retracker_properties(self, n_records: int) -> None:
        """
        Initialize retracker properties with correct arrays shapes (shape = (n_records, )).
        The list of parameters depends on whether the SAMOSA_DEBUG_MODE flag is set.

        NOTE: The properties are set to an array, but can be accessed as `self.{property_name}`
        via the __getattr__ method.

        :param n_records: Number of Level-2 data points
        """

        parameter = [
            "misfit",
            "swh",
            "mean_square_slope",
            "wind_speed",
            "epoch",
            "guess",
            "fit_num_func_eval",
            "fit_return_status",
            "Pu",
            "rval",
            "kval",
            "pval",
            "cval",
        ]
        for parameter_name in parameter:
            self._retracker_params[parameter_name] = np.full(n_records, np.nan, dtype=np.float32)

        parameter = [
            "waveform",
            "waveform_model"
        ]
        n_range_gates = self._l1b.waveform.power.shape[1]
        for parameter_name in parameter:
            self._retracker_params[parameter_name] = np.full((n_records, n_range_gates), np.nan, dtype=np.float32)

    def _samosa_plus_retracker(
            self,
            rng: np.ndarray,
            wfm: np.ndarray,
            indices: np.ndarray,
            radar_mode: np.ndarray
    ) -> List[SAMOSAWaveformFitResult]:
        """
        Performs the retracking via SAMOSA waveform model fits for
        all target waveforms.

        :param rng: Range window arrays
        :param wfm: Waveform arrays
        :param indices: Target waveform indices
        :param radar_mode: Radar modes

        :return: List of SAMOSA plus waveform fit results
        """

        # Aggregate all necessary input data for the SAMOSA waveform model fit
        waveform_collection = self._get_waveform_fit_data(rng, wfm, indices, radar_mode)

        # Configure and execute waveform fitting for all target waveforms
        args, kwargs = self._get_waveform_model_fit_configuration()
        waveform_fits = SAMOSAWaveformCollectionFit(*args, **kwargs)
        return waveform_fits.fit(waveform_collection)

    def _get_waveform_fit_data(
            self,
            rng: np.ndarray,
            wfm: np.ndarray,
            indices: np.ndarray,
            radar_mode: np.ndarray
    ) -> List[WaveformFitData]:
        """
        Collects all the data required for the waveform model fits.
        Whenever possible,

        :param rng:
        :param wfm:
        :param indices:
        :param radar_mode:
        :return:
        """

        waveform_collection = []
        for idx in indices:
            # Create the SAMOSA Waveform scenario data
            look_angles = get_look_angles(self._l1b, idx)
            scenario_data = self._get_scenario_data(self._l1b, self._l2, idx, look_angles)
            waveform_data = self._get_normed_waveform(
                rng[idx, :],
                wfm[idx, :],
                scenario_data.rp.tau,
                radar_mode[idx],
                self._l1b.classifier.window_delay[idx],
                self._l1b.classifier.transmit_power[idx],
                look_angles,
                self._l1b.classifier.first_maximum_index[idx]
            )
            waveform_collection.append(WaveformFitData(idx, waveform_data, scenario_data))

        return waveform_collection

    def _get_scenario_data(
            self,
            l1: Level1bData,
            l2: Level2Data,
            idx: int,
            look_angles
    ) -> ScenarioData:
        """
        Creates the scenario data for the SAMOSA waveform model. This dataclass
        is needed for the initialization of the SAMOSA waveform model class and
        contains information on the radar mode and radar processing settings as
        well as the geometry (location, altitude & altitude rate, attitude and
        velocity of the altimeter platform).

        The specific radar parameters are expected to be included in the
        `samosa_waveform_model` and are retrieved by specifying platform
        and radar mode.

        Geometry parameters are available in the Level-1 and Level-2 data
        containers.

        :param l1: Level-1 data container
        :param l2: Level-2 data container
        :param idx: Target Waveform index
        :param look_angles:

        :return: SAMOSA waveform model scenario data for specific waveform
        """

        radar_mode_name = RadarModes.get_name(l1.waveform.radar_mode[idx])
        platform = l2.info.mission

        sp = SensorParameters.get(platform, radar_mode_name)

        location_data = dict(
            latitude=l2.latitude[idx],
            longitude=l2.longitude[idx],
            altitude=l2.altitude[idx],
            height_rate=l1.time_orbit.altitude_rate[idx],
            pitch=np.radians(l1.time_orbit.antenna_pitch[idx]),
            roll=np.radians(l1.time_orbit.antenna_roll[idx]),
            velocity=total_velocity_from_vector(
                self._l1b.classifier.satellite_velocity_x[idx],
                self._l1b.classifier.satellite_velocity_y[idx],
                self._l1b.classifier.satellite_velocity_z[idx]
            )
        )
        geo = PlatformLocation(**location_data)
        sar = SARParameters(look_angles=look_angles)
        sar.compute_multi_look_parameters(geo=geo, sp=sp)

        return ScenarioData(sp, geo, sar)

    def _get_normed_waveform(
            self,
            rng: np.ndarray,
            wfm: np.ndarray,
            tau: np.ndarray,
            radar_mode: int,
            window_delay: float,
            transmit_power: float,
            look_angles: np.ndarray,
            first_maximum_index: int
    ) -> NormedWaveform:
        """
        Return a normed representation of the input waveform

        :param rng:
        :param wfm:
        :param radar_mode:

        :return:
        """
        scaling_factor = 1.0 / np.nanmax(wfm)
        normed_waveform = wfm * scaling_factor
        return NormedWaveform(
            normed_waveform,
            rng,
            tau,
            radar_mode,
            scaling_factor,
            window_delay,
            transmit_power,
            look_angles,
            self._options.get("surface_type"),
            self._options.get("surface_class"),
            first_maximum_index=first_maximum_index
        )

    def _get_waveform_model_fit_configuration(self) -> Tuple[List, Dict]:
        """
        Constructs the arguments and keywords for the SAMOSAWaveformCollectionFit class,
        defining the fit method and its configration
        :return:
        """

        fit_method = self._options.get("fit_method")
        if fit_method is None:
            raise ValueError("Mandatory configuration parameter `fit_method not specified")

        predictor_method = self._options.get("predictor_method")
        if fit_method is None:
            raise ValueError("Mandatory configuration parameter `predictor_method not specified")
        args = [fit_method, predictor_method]

        num_processes = self._options.get("num_processes")
        num_processes = psrlcfg.CPU_COUNT if num_processes == "pysiral-cfg" else int(num_processes)
        kwargs = {
            "use_multiprocessing": self._options.get("use_multiprocessing", False),
            "num_processes": num_processes,
            "least_squares_kwargs": self._options.get("fit_kwargs", {}),
            "predictor_kwargs": self._options.get("predictor_kwargs", {}),
        }
        return args, kwargs

    def _store_retracker_properties(
        self,
        fit_results: List[SAMOSAWaveformFitResult],
        indices: np.ndarray
    ) -> None:
        """
        Store the output of the SAMOSA+ retracker in the class and maps
        the fit result to the corresponding indices in the Level-2 data.

        :param fit_results:
        :param indices:

        :return: None
        """

        for index, fit_result in zip(indices, fit_results):

            # Main retracker properties
            self._range[index] = fit_result.retracker_range
            self._power[index] = fit_result.sigma0

            # Derived physical parameters
            # self.wind_speed[index] = func_wind_speed([fit_result.sigma0])

            # Waveform model fit parameters
            self.swh[index] = fit_result.significant_wave_height
            self.misfit[index] = fit_result.misfit
            self.mean_square_slope[index] = fit_result.mean_square_slope
            self.epoch[index] = fit_result.epoch

            # Store waveform and waveform model for debugging
            self.waveform[index, :] = fit_result.waveform
            self.waveform_model[index, :] = fit_result.waveform_model

            # Fit method statistics
            self.fit_num_func_eval[index] = fit_result.number_of_model_evaluations
            self.fit_return_status[index] = fit_result.fit_return_status

    def _l2_register_retracker_parameters(self) -> None:
        """
        Add retracker variables to the L2 data object. The specific variables
        depend on surface type.
        """

        # General auxiliary variables
        self.register_auxdata_output("sammf", "samosa_misfit", self.misfit)
        self.register_auxdata_output("sammss", "samosa_mean_square_slope", self.mean_square_slope)
        self.register_auxdata_output("samfnfe", "samosa_fit_num_func_eval", self.fit_num_func_eval)
        self.register_auxdata_output("samfrs", "samosa_fit_return_status", self.fit_return_status)

        # Waveform and waveform model
        # Register results as auxiliary data variable
        num_range_gates = self.waveform.shape[1]
        dims = {"new_dims": (("range_gates", num_range_gates),),
                "dimensions": ("time", "range_gates"),
                "add_dims": (("range_gates", np.arange(num_range_gates)),)}
        self._l2.set_multidim_auxiliary_parameter("samiwfm", "waveform", self.waveform, dims, update=True)
        self._l2.set_multidim_auxiliary_parameter("samwfm", "waveform_model", self.waveform_model, dims, update=True)

        # Lead and open ocean surfaces
        surface_class = self._options.get("surface_class", "undefined")
        if surface_class == "polar_ocean":
            self.register_auxdata_output("samswh", "samosa_swh", self.swh)
            self.register_auxdata_output("samwsp", "samosa_wind_speed", self.wind_speed)

        # Sea ice surface types
        elif surface_class == "sea_ice":
            self.register_auxdata_output("samshsd", "samosa_surface_height_standard_deviation", self.swh / 4.)

        else:
            logger.warning("No specific surface type set for SAMOSA+: No auxiliary variables added to L2")

    def __getattr__(self, item: str) -> Any:
        """
        Direct attribute access to the retracker properties dictionary

        :param item: parameter name

        :raise AttributeError: item not in self._retracker_params (see self.create_retracker_properties)

        :return:
        """
        if item in self._retracker_params:
            return self._retracker_params[item]
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {item}")


# def samosa_fit_swh_mss(
#         fit_data: WaveformFitData,
#         predictor_kwargs: Dict = None,
#         least_squares_kwargs: Dict = None,
#         sub_waveform_method: Optional[Literal["filter_trailing_edge", "specular_only"]] = None
# ) -> SAMOSAWaveformFitResult:
#     """
#     Fits the SAMOSA waveform model with all free parameters (epoch, swh, mss)
#
#     :param fit_data: Input parameters for waveform fitting process. Mainly
#         contains waveform model scenario data and waveform data.
#     :param predictor_kwargs: Input parameter for parameter first guess and fit bounds
#     :param least_squares_kwargs: Keyword arguments to `scipy.optimize.least_squares`
#     :param sub_waveform_method:
#
#     :return: SAMOSA+ waveform model fit result
#     """
#
#     # Input validation
#     if least_squares_kwargs is None:
#         least_squares_kwargs = {}
#
#     if predictor_kwargs is None:
#         predictor_kwargs = {}
#
#     # Unpack for readability.
#     scenario_data, waveform_data = fit_data.scenario_data, fit_data.waveform_data
#     waveform_data.thermal_noise = compute_thermal_noise(waveform_data.power)
#
#     # Compute the lower envelope of the trailing edge
#     if sub_waveform_method is not None:
#         sub_waveform_mask = get_trailing_edge_lower_envelope_mask(
#             waveform_data.power,
#             waveform_data.first_maximum_index
#         )
#         sub_waveform_mask = np.logical_not(sub_waveform_mask)
#     else:
#         sub_waveform_mask = None
#     # Get first guess of fit parameters and fit bounds
#     predictor = SAMOSAModelParameterPrediction(
#         "fit_mss_swh",
#         waveform_data.surface_type,
#         **predictor_kwargs
#     )
#     first_guess, lower_bounds, upper_bounds = predictor.get(waveform_data)
#
#     # Update least square kwargs with fit bounds
#     # (Fit bounds are dynamic per waveform).
#     fit_kwargs = dict(bounds=(lower_bounds, upper_bounds))
#     fit_kwargs.update(least_squares_kwargs)
#
#     # The fitting process is happening here:
#     fit_cls = SAMOSAWaveformFit(
#         scenario_data, waveform_data,
#         sub_waveform_mask=sub_waveform_mask,
#         waveform_model_kwargs=dict(mode=2)
#     )
#     fit_result = least_squares(fit_cls.fit_func, first_guess, **fit_kwargs)
#
#     # Unpack parameters for better readability
#     epoch_ns, swh, mss = fit_result.x
#     epoch = epoch_ns*1.0e-9
#
#     # Recompute the selected waveform model. Required to store the fitted waveform model.
#     # NOTE: The waveform model cannot (easily) be retrieved from the fit class
#     # because it is not always the last model.
#     fitted_model = get_model_from_args(fit_cls.samosa_waveform_model, [epoch, swh, mss])
#
#     scale_model = 1. / np.nanmax(fitted_model.power)
#     scale = 1. / waveform_data.power[waveform_data.first_maximum_index]
#
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(9, 6))
#     plt.title(f"swh={swh:0.2f}, nu={mss:.2f}")
#     plt.plot(waveform_data.tau, waveform_data.power * scale, color="black", lw=0.5)
#     plt.scatter(waveform_data.tau[sub_waveform_mask], waveform_data.power[sub_waveform_mask] * scale, color="red", s=10)
#     plt.scatter(waveform_data.tau, waveform_data.power * scale, color="black", s=2)
#     # plt.plot(waveform_data.tau, fitted_model_step1.power * scale_1 + waveform_data.thermal_noise, label="step1")
#     plt.plot(waveform_data.tau, fitted_model.power * scale_model + waveform_data.thermal_noise, label="step2")
#     # plt.plot(waveform_data.tau, fitted_model_combined.power * scale_c + waveform_data.thermal_noise, label="combined")
#     plt.legend()
#     plt.show()
#     breakpoint()
#
#     # Compute the misfit from residuals in SAMPy fashion
#     misfit = sampy_misfit(fit_result.fun)
#
#     # Convert epoch to range (excluding range corrections)
#     # TODO: apply radar mode range bias here?
#     retracker_range = epoch2range(epoch, fit_data.waveform_data.window_delay)
#
#     return SAMOSAWaveformFitResult(
#         epoch,
#         retracker_range,
#         retracker_range_uncertainty,
#         swh,
#         mss,
#         0.0,  # placeholder for thermal noise
#         misfit,
#         "single_fit_mss_swh",
#         waveform_data.power,
#         fitted_model.power,
#         number_of_model_evaluations=fit_result.nfev,
#         fit_return_status=fit_result.status
#     )


def samosa_fit_samosap_standard(
        fit_data: WaveformFitData,
        trailing_edge_sub_waveform_filter: bool = True,
        predictor_kwargs: Dict = None,
        least_squares_kwargs: Dict = None,
        filter_trailing_edge_kwargs: Dict = None
) -> SAMOSAWaveformFitResult:
    """
    Fits the SAMOSA waveform model with all free parameters (epoch, swh, mss, amplitude) using
    the two-step standard fitting approach used for open ocean waveforms with SAMOSA+.

    :param fit_data: Input parameters for waveform fitting process. Mainly
        contains waveform model scenario data and waveform data.
    :param trailing_edge_sub_waveform_filter: Flag if trailing edge filter should be used
    :param predictor_kwargs: Input parameter for parameter first guess and fit bounds
    :param least_squares_kwargs: Keyword arguments to `scipy.optimize.least_squares`
    :param filter_trailing_edge_kwargs: Keyword arguments to

    :return: SAMOSA+ waveform model fit result
    """

    # Input validation
    predictor_kwargs = {} if predictor_kwargs is None else predictor_kwargs
    least_squares_kwargs = {} if least_squares_kwargs is None else least_squares_kwargs
    filter_trailing_edge_kwargs = {} if filter_trailing_edge_kwargs is None else filter_trailing_edge_kwargs

    # Unpack for readability
    scenario_data, waveform_data = fit_data.scenario_data, fit_data.waveform_data
    waveform_data.thermal_noise = compute_thermal_noise(waveform_data.power)

    # Get the sub-waveform mask
    # (unless explicitly disabled by `trailing_edge_sub_waveform_filter=False` config file)
    if trailing_edge_sub_waveform_filter is not None:
        sub_waveform_mask = get_sub_waveform_mask(waveform_data, filter_trailing_edge_kwargs)
    else:
        sub_waveform_mask = None

    # Get first guess of fit parameters and fit bounds
    predictor = SAMOSAModelParameterPrediction("samosap_standard", waveform_data.surface_type, **predictor_kwargs)

    # --- SAMOSA+ Fit Step 1 ---
    # This step fits a waveform with fixed nu and variable swh
    model_parameters_step1, fitted_model_step1, optimize_result_step1 = samosa_fit_samosap_standard_step1(
        waveform_data, scenario_data, predictor,
        least_squares_kwargs=least_squares_kwargs,
        sub_waveform_mask=sub_waveform_mask
    )

    # --- SAMOSA+ Fit Step 2 ---
    # This step fits a waveform with fixed swh and variable nu
    # This fit will be used
    model_parameters_step2, fitted_model_step2, optimize_result_step2 = samosa_fit_samosap_standard_step2(
        waveform_data, scenario_data, predictor,
        least_squares_kwargs=least_squares_kwargs,
        sub_waveform_mask=sub_waveform_mask
    )

    # --- Summarize the result from two fits ---
    # Note that the waveform model from the combination of swh and nu will provide the best fit

    # Compute the misfit from residuals in SAMPy fashion
    misfit_subwaveform = sampy_misfit(optimize_result_step2.fun)
    misfit = sampy_misfit(fitted_model_step2.power - waveform_data.power)

    # Count number of model evaluations (performance tuning metric)
    number_of_model_evaluations = optimize_result_step1.nfev + optimize_result_step2.nfev

    # Convert epoch to range (excluding range corrections)
    retracker_range = epoch2range(model_parameters_step2.epoch, fit_data.waveform_data.window_delay)
    retracker_range_standard_error = 0.5 * 299792458. * model_parameters_step2.epoch_sdev

    # # --- Debug Plot ---
    # plot_parameters = [
    #     ["range", f"{retracker_range:.2f} +/- {retracker_range_standard_error:.2f}m"],
    #     ["swh", f"{swh:.2f} +/- {model_parameters_step1.significant_wave_height_sdev:.2f}m"],
    #     ["nu", f"{nu:.2f} +/- {model_parameters_step2.nu_sdev:.2f}"],
    #     ["misfit (sub)", f"{misfit:.2f} ({misfit_subwaveform:.2f})"],
    # ]
    #
    # import matplotlib.pyplot as plt
    # import os
    # from pathlib import Path
    # idx = int(os.environ["PYSIRAL_SAMOSA_TEST_WAVEFORM_IDX"])
    # output_dir = os.environ["PYSIRAL_SAMOSA_TEST_OUTPUT_DIR"]
    # fig = plt.figure(figsize=(9, 6))
    # plt.title(f"Waveform index: {idx:05g}")
    # plt.axvline(epoch, color="violet", lw=0.5, ls="dashed", label="epoch")
    # plt.plot(waveform_data.tau, waveform_data.power, color="0.5", lw=0.5, zorder=20)
    # plt.scatter(waveform_data.tau, waveform_data.power, color="0.5", s=6, marker="o", zorder=20)
    # plt.scatter(
    #     waveform_data.tau[sub_waveform_mask],
    #     waveform_data.power[sub_waveform_mask],
    #     color="red", marker="x", s=20,
    #     label="Rejected Range Gates",
    #     zorder=30
    # )
    # # plt.plot(waveform_data.tau, fitted_model_step1.power * scale_1 + waveform_data.thermal_noise, label="step1")
    # plt.plot(waveform_data.tau, fitted_model_step2.power, label="Fitted Model (SAMOSA step 2)",
    #          color="violet", alpha=0.85, zorder=40)
    # plt.legend(loc="upper right")
    # y = 0.7
    # y_step = 0.05
    # for label, value in plot_parameters:
    #     plt.annotate(f"{label} =", (0.7, y), xycoords="axes fraction", ha="right")
    #     plt.annotate(f" {value}", (0.7, y), xycoords="axes fraction", ha="left")
    #     y -= y_step
    # plt.ylim(0, 1.25)
    # plt.xlabel("Range Gate (seconds)")
    # plt.ylabel("Normed Waveform Power")
    # plt.savefig(Path(output_dir) / f"waveform_data_{idx:06g}_waveform_fit.png", dpi=300)
    # fig.clear()
    # plt.close(fig)

    return SAMOSAWaveformFitResult(
         epoch=model_parameters_step2.epoch,
         retracker_range=retracker_range,
         retracker_range_standard_error=retracker_range_standard_error,
         significant_wave_height=model_parameters_step1.significant_wave_height,
         significant_wave_height_standard_error=model_parameters_step1.significant_wave_height_sdev,
         mean_square_slope=model_parameters_step2.mean_square_slope,
         mean_square_slope_standard_error=1. / model_parameters_step2.nu_sdev,
         thermal_noise=model_parameters_step2.thermal_noise,
         misfit=misfit,
         misfit_sub_waveform=misfit_subwaveform,
         fit_mode="samosap_standard",
         waveform=waveform_data.power,
         waveform_model=fitted_model_step2.power,
         number_of_model_evaluations=number_of_model_evaluations,
         fit_return_status=optimize_result_step2.status
    )


def samosa_fit_samosap_standard_step1(
        waveform_data: NormedWaveform,
        scenario_data: ScenarioData,
        predictor: SAMOSAModelParameterPrediction,
        step1_fixed_nu_value: float = 0.0,
        least_squares_kwargs: Optional[Dict] = None,
        sub_waveform_mask: Optional[np.ndarray] = None,
) -> Tuple[WaveformModelParameters, WaveformModelOutput, OptimizeResult]:
    """
    Performs fit step 1 of the SAMOSAPlus retracker (as implemented in SAMPY)

    :param waveform_data:
    :param scenario_data:
    :param predictor:
    :param step1_fixed_nu_value:
    :param least_squares_kwargs:
    :param sub_waveform_mask:

    :return: Waveform model parameters,
    """

    # Get the fit parameters
    first_guess, lower_bounds, upper_bounds = predictor.get(waveform_data, mode=1)

    # Update least square kwargs with fit bounds
    # (Fit bounds are dynamic per waveform).
    fit_kwargs = dict(bounds=(lower_bounds, upper_bounds))
    fit_kwargs.update(least_squares_kwargs)

    # First fit step in SAMOSA+ two-stage fits, which fits
    # three parameters: 1. epoch, 2. significant wave height, 3. Amplitude
    fit_cls = SAMOSAWaveformFit(
        scenario_data,
        waveform_data,
        waveform_model_kwargs=dict(mode=1),
        step1_fixed_nu_value=step1_fixed_nu_value,
        sub_waveform_mask=sub_waveform_mask
    )
    fit_result_step1 = least_squares(fit_cls.fit_func_samosap_standard_step1, first_guess, **fit_kwargs)
    parameter_vars = get_least_squares_parameter_sdev(fit_result_step1)

    # --- Collect output ---
    # Summarize the fit result parameters
    model_parameters_step1 = WaveformModelParameters(
        epoch=fit_result_step1.x[0] * 1e-9,
        epoch_sdev=float(parameter_vars[0] * 1e-9),
        significant_wave_height=fit_result_step1.x[1],
        significant_wave_height_sdev=float(parameter_vars[1]),
        nu=step1_fixed_nu_value,
        nu_sdev=0.0,
        amplitude_scale=fit_result_step1.x[2],
        thermal_noise=waveform_data.thermal_noise
    )

    # Compute the waveform model with the fit parameters
    fitted_model_step1 = get_model_from_args(
        fit_cls.samosa_waveform_model,
        model_parameters_step1.args_list,
        amplitude_scale=model_parameters_step1.amplitude_scale,
        thermal_noise=model_parameters_step1.thermal_noise
    )
    return model_parameters_step1, fitted_model_step1, fit_result_step1


def samosa_fit_samosap_standard_step2(
        waveform_data: NormedWaveform,
        scenario_data: ScenarioData,
        predictor: SAMOSAModelParameterPrediction,
        step2_fixed_swh_value: float = 0.0,
        least_squares_kwargs: Optional[Dict] = None,
        sub_waveform_mask: Optional[np.ndarray] = None,
) -> Tuple[WaveformModelParameters, WaveformModelOutput, OptimizeResult]:
    """
    Performs fit step 2 of the SAMOSAPlus retracker (as implemented in SAMPY)

    :param waveform_data:
    :param scenario_data:
    :param predictor:
    :param step2_fixed_swh_value:
    :param least_squares_kwargs:
    :param sub_waveform_mask:

    :return: Fit result waveform mode
    """

    # Get the fit parameters
    first_guess, lower_bounds, upper_bounds = predictor.get(waveform_data, mode=2)

    # Update least square kwargs with fit bounds
    # (Fit bounds are dynamic per waveform).
    fit_kwargs = dict(bounds=(lower_bounds, upper_bounds))
    fit_kwargs.update(least_squares_kwargs)

    # First fit step in SAMOSA+ two-stage fits, which fits
    # three parameters: 1. epoch, 2. significant wave height, 3. Amplitude
    fit_cls = SAMOSAWaveformFit(
        scenario_data,
        waveform_data,
        waveform_model_kwargs=dict(mode=2),
        step2_fixed_swh_value=step2_fixed_swh_value,
        sub_waveform_mask=sub_waveform_mask
    )
    fit_result_step2 = least_squares(fit_cls.fit_func_samosap_standard_step2, first_guess, **fit_kwargs)
    parameter_sdev = get_least_squares_parameter_sdev(fit_result_step2)

    # --- Collect output ---
    # Summarize the fit result parameters
    model_parameters_step2 = WaveformModelParameters(
        epoch=fit_result_step2.x[0] * 1e-9,
        epoch_sdev=float(parameter_sdev[0] * 1e-9),
        significant_wave_height=step2_fixed_swh_value,
        significant_wave_height_sdev=0.0,
        nu=fit_result_step2.x[1],
        nu_sdev=float(parameter_sdev[1]),
        amplitude_scale=fit_result_step2.x[2],
        thermal_noise=waveform_data.thermal_noise
    )

    # Compute the waveform model with the fit parameters
    fitted_model_step2 = get_model_from_args(
        fit_cls.samosa_waveform_model,
        model_parameters_step2.args_list,
        amplitude_scale=model_parameters_step2.amplitude_scale,
        thermal_noise=model_parameters_step2.thermal_noise
    )

    # Compute uncertainties (Standard devi
    return model_parameters_step2, fitted_model_step2, fit_result_step2


def get_model_from_args(
        samosa_waveform_model,
        args,
        amplitude_scale: float = 1.0,
        thermal_noise: float = 0.0,
) -> "WaveformModelOutput":
    """
    Compute waveform model with final fit parameters. This function
    allows to compute the waveform model from fit parameter triplets.

    :param samosa_waveform_model: Initialized and configured SAMOSA+
        Waveform model instance.

    :param args: list of [epoch, swh, mss]
    :param amplitude_scale: Optional scaling factor
    :param thermal_noise: Optional thermal noise

    :return: Waveform model
    """
    model_parameter = WaveformModelParameters(
        epoch=args[0],
        significant_wave_height=args[1],
        nu=args[2],
        amplitude_scale=amplitude_scale,
        thermal_noise=thermal_noise
    )
    return samosa_waveform_model.generate_delay_doppler_waveform(model_parameter)


def total_velocity_from_vector(
        velocity_x: float,
        velocity_y: float,
        velocity_z: float
) -> float:
    """
    Compute total velocity from vector

    :param velocity_x: velocity x-component
    :param velocity_y: velocity y-component
    :param velocity_z: velocity z-component

    :return: Total velocity
    """
    return np.sqrt(velocity_x**2. + velocity_y**2. + velocity_z**2.)


def get_epoch_bounds(
        epoch: np.ndarray,
        earliest_fraction: float = 0.0,
        latest_fraction: float = 1.0
) -> Tuple[float, float]:
    """
    Get bounds for epoch based on the valid fraction range in the range window.

    :param epoch: epoch array
    :param earliest_fraction: first valid epoch as fraction of the range window
    :param latest_fraction: last valid epoch as fraction of the range window

    :raises ValueError: fraction not in interval [0-1] or earliest > latest

    :return: (epoch lower bound, epoch upper bound)
    """

    if earliest_fraction > latest_fraction:
        raise ValueError(f"{earliest_fraction} not smaller {latest_fraction}")

    for value in [earliest_fraction, latest_fraction]:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{value} out of bounds [0-1]")

    window_size = epoch[-1] - epoch[0]
    return float(epoch[0] + earliest_fraction * window_size), float(epoch[0] + latest_fraction * window_size)


def sampy_misfit(residuals: np.ndarray, waveform_mask: Optional[np.ndarray] = None) -> float:
    """
    Computes the SAMOSA waveform model misfit parameter according to SAMPy with optional
    misfit computation on sub-waveform.

    :param residuals: difference between waveform and waveform model
    :param waveform_mask: numpy index array of sub-waveform mask

    :return: SAMOSA waveform model misfit
    """
    waveform_mask = waveform_mask if waveform_mask is not None else np.arange(residuals.size)
    return np.sqrt(1. / residuals[waveform_mask].size * np.nansum(residuals[waveform_mask] ** 2)) * 100.


def compute_thermal_noise(
        waveform: np.ndarray,
        start_index: int = 4,
        end_index: int = 12,
) -> float:
    """
    Compute thermal noise for waveform collection (from SAMPy). The strategy is
    to compute the media in a slice of the lowest values of the first half
    of the waveform (apparently the end of the waveform contains zeros).

    :param waveform: The waveform array
    :param start_index: start of the slice of sorted waveform power values (forced to valid range)
    :param end_index: end of the slice of sorted waveform power values (forced to valid range)

    :return: thermal noise in waveform power units
    """
    waveform_sorted = np.sort(waveform[:waveform.shape[0] // 2])
    start_index = max(start_index, 0)
    end_index = waveform_sorted.size - 1 if start_index >= waveform.size else end_index
    return np.nanmedian(waveform_sorted[start_index:end_index+1])


def epoch2range(epoch: float, window_delay: float) -> float:
    """
    Computes the retracker range, defined as range of spacecraft center or mass
    to retracked elevation.

    :param epoch: retracker epoch in seconds

    :param window_delay:

    :return: retracker range in meter.
    """
    factor = 0.5 * 299792458.
    return epoch * factor + window_delay * factor


def get_look_angles(l1, index: int) -> np.ndarray:
    """
    Compute the lookangles based on l1b stack information
    # TODO: This functions raises a ValueError for NaN values in the classifiers (LRM?)

    :param l1: The level 1 data object
    :param index: Waveform index

    :return:
    """
    return 90. - np.linspace(np.rad2deg(l1.classifier.look_angle_start[index]),
                             np.rad2deg(l1.classifier.look_angle_stop[index]),
                             num=int(l1.classifier.stack_beams[index]))


def get_least_squares_parameter_sdev(optimize_result: "OptimizeResult") -> np.ndarray:
    """
    Compute standard deviation of parameter from Jacobian.

    Solution from Stack Overflow: https://stackoverflow.com/questions/42388139

    :param optimize_result: Return object from `scipy.optimize.least_squares`

    :return: List of fit parameter standard deviations
    """
    jac = optimize_result.jac
    cov = np.linalg.inv(jac.T.dot(jac))
    variance = np.sqrt(np.diagonal(cov))
    return np.sqrt(variance)


def get_sub_waveform_mask(waveform_data: NormedWaveform, filter_trailing_edge_kwargs: Dict) -> np.ndarray:
    """
    Estimates a sub-waveform mask by flagging off-nadir backscatter elements
    that manifests as peaks on the trailing edge. This is done by estimating
    the lower envelope of the trailing edge and exclude all waveform power
    values that

    :param waveform_data: Waveform data
    :param filter_trailing_edge_kwargs: Configuration keyword arguments for the
        trailing edge filter.

    :return: sub-waveform mask (True: masked)
    """
    sub_waveform_mask = get_trailing_edge_lower_envelope_mask(
        waveform_data.power,
        waveform_data.first_maximum_index,
        **filter_trailing_edge_kwargs
    )
    return np.logical_not(sub_waveform_mask)


def get_trailing_edge_lower_envelope_mask(
        waveform_power: np.ndarray,
        first_maximum_index: int,
        max_minimum_cleaning_passes: int = 5,
        noise_level_normed: float = 0.02,
        return_type: Literal["bool", "indices"] = "bool"
) -> np.ndarray:
    """
    Compute the indices of the waveforms trailing edge lower envelope.
    This method relies on the computation of relative minima via scipy.
    with added filtering.

    :param waveform_power: Waveform power values (any unit)
    :param first_maximum_index: Range gate index of the first maximum.
    :param max_minimum_cleaning_passes: Local minima on the trailing edge
        may not be in strictly decreasing in power depending on the
        off-nadir backscatter contanimation of the trailing edge.
        Local minima are iteratively removed that are not decreasing in
        power.
    :param noise_level_normed: Points within the noise level of the linear
        fit between successive local minima are added to the trailing edge
        mask. Noise level unit is the first maximum power.
    :param return_type: Options are a boolean array with the same dimensions
        as the waveform (default), or an index array.

    :return:
    """

    # Perform operation on trailing edge only normalized by
    # first maximum power (trailing edge is defined as
    # everything after the first maximum).
    wfm_trailing_edge = waveform_power[first_maximum_index:]
    wfm_trailing_edge /= np.nanmax(wfm_trailing_edge)

    # Get local minima and add first maximum and last range gate
    # the list (important for fitting waveform power).
    idx_lmin = argrelmin(wfm_trailing_edge)[0]
    idx_lmin = np.insert(idx_lmin, 0, 0)
    idx_lmin = np.insert(idx_lmin, len(idx_lmin), wfm_trailing_edge.size-1)

    # Throw out local minima that have greater power then the previous
    # one in several iterations
    for _ in np.arange(max_minimum_cleaning_passes):
        wfm_power_change = [
            wfm_trailing_edge[idx_lmin[1:]] - wfm_trailing_edge[idx_lmin[:-1]]
        ][0]
        wfm_change = np.insert(wfm_power_change, 0, -1)
        valid_local_minima = wfm_change < 0.0
        if valid_local_minima.all():
            break
        idx_lmin = idx_lmin[wfm_change < 0.0]

    # Create a leading edge mask. This is required for the
    # following operation
    mask_le = np.full(wfm_trailing_edge.size, False)
    mask_le[idx_lmin] = True

    # Take in again waveform points that are within the noise level
    # (ensures that are enough points for the fit, without deviating
    # from the lower envelope too much).
    # This is done by computing a linear fit between successive
    # local minima and checking the difference of actual power
    # values to the linear fit.
    n_range_gates = len(waveform_power)
    for i in np.arange(idx_lmin.size-1):

        # --- Method 1: Distance to POCA of linear fit ---
        idx_lmin_scaled = idx_lmin.astype(float)/ n_range_gates
        k = wfm_trailing_edge[idx_lmin[i]]
        m = ((wfm_trailing_edge[idx_lmin[i+1]] - wfm_trailing_edge[idx_lmin[i]]) /
             (idx_lmin_scaled[i+1] - idx_lmin_scaled[i]))

        for x, idx in enumerate(np.arange(idx_lmin[i]+1, idx_lmin[i+1])):

            x0 = float(x + 1) / n_range_gates
            y0 = wfm_trailing_edge[idx]

            x_poca = (x0 + m * y0 - m * k) / (m ** 2. + 1)
            y_poca = m * (x0 + m * y0 - m * k) / (m ** 2. + 1) + k

            dist = np.sqrt((x_poca - x0)**2. + (y_poca - y0)**2)
            mask_le[idx] = dist <= noise_level_normed

        # # --- Method 2: Vertical Distance to linear fit ---
        # linear_fit = np.linspace(
        #     wfm_trailing_edge[idx_lmin[i]],
        #     wfm_trailing_edge[idx_lmin[i+1]],
        #     idx_lmin[i + 1] - idx_lmin[i]
        # )
        #
        # power_delta = wfm_trailing_edge[idx_lmin[i]:idx_lmin[i+1]] - linear_fit
        # in_noise_level = np.absolute(power_delta) <= noise_level_normed
        # mask_le[idx_lmin[i]:idx_lmin[i+1]] = in_noise_level

    # Final step: Pad trailing edge mask to full waveform mask
    mask = np.full(waveform_power.size, True)
    mask[first_maximum_index:] = mask_le

    # # --- Debug plot ---
    # inverted_mask = np.logical_not(mask)
    # import os
    # from pathlib import Path
    # idx = int(os.environ["PYSIRAL_SAMOSA_TEST_WAVEFORM_IDX"])
    # output_dir = os.environ["PYSIRAL_SAMOSA_TEST_OUTPUT_DIR"]
    # x = np.arange(waveform_power.size)
    # fig = plt.figure(figsize=(9, 6))
    # plt.title(f"Waveform Index: {idx:05g}")
    # plt.plot(x, waveform_power, color="black", lw=0.75, zorder=20)
    # plt.scatter(x[mask], waveform_power[mask], color="0.5", s=6, marker="o", zorder=20)
    # plt.scatter(x[inverted_mask], waveform_power[inverted_mask], color="red", marker="x", s=30, zorder=30,
    #             label="Rejected Range Gates")
    # plt.scatter(x[idx_lmin+first_maximum_index], waveform_power[idx_lmin+first_maximum_index],
    #             edgecolors="greenyellow", facecolors="none", marker="s", s=30, zorder=30,
    #             label="Detected Local Minima")
    # plt.legend(loc="upper right")
    # plt.ylim(0, 1.25)
    # plt.xlabel("Range Gate")
    # plt.ylabel("Normed Waveform Power")
    # plt.savefig(Path(output_dir) / f"waveform_data_{idx:06g}_trailing_edge_filter.png", dpi=300)
    # fig.clear()
    # plt.close(fig)

    return mask if return_type is "bool" else np.where(mask)[0]

# def calc_sigma0(
#         Latm,
#         Pu,
#         CST,
#         RDB,
#         GEO,
#         epoch_sec,
#         window_del_20_hr_ku_deuso,
#         lat_20_hr_ku,
#         alt_20_hr_ku,
#         sat_vel_vec_20_hr_ku,
#         transmit_pwr_20_ku
# ):
#     atmospheric_attenuation = 10. ** (np.zeros(np.shape(alt_20_hr_ku)) / 10.)  ### IMP!!! ===>  Atmospheric Correction, set to zero dB
#     # Atm_Atten    = 10.**(Latm/10.)
#     Range = epoch_sec * CST.c0 / 2 + CST.c0 / 2 * window_del_20_hr_ku_deuso  #### this should be the retracked range without geo-corrections
#     Pout = Pu
#     earth_radius = np.sqrt(CST.R_e ** 2.0 * (np.cos(np.deg2rad(lat_20_hr_ku))) ** 2 + CST.b_e ** 2.0 * (
#         np.sin(np.deg2rad(lat_20_hr_ku))) ** 2)
#     kappa = (1. + alt_20_hr_ku / earth_radius)
#
#     Lx = CST.c0 * Range / (2. * sat_vel_vec_20_hr_ku * RDB.f_0 * RDB.Np_burst * 1. / RDB.PRF_SAR)
#     Ly = np.sqrt(CST.c0 * Range * (1. / RDB.Bs) / kappa)
#     A_SAR = Lx * (2. * Ly)
#     C = ((4. * np.pi) ** 3. * (GEO.Height ** 4) * atmospheric_attenuation) / (((CST.c0 / RDB.f_0) ** 2) * (RDB.G_0 ** 2) * A_SAR)
#
#     sigma0 = 10. * np.log10(Pout / transmit_pwr_20_ku) + 10. * np.log10(C) + RDB.bias_sigma0
#
#     return sigma0, np.log10(Pout / transmit_pwr_20_ku), np.log10(C), Range, kappa
