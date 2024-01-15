import numpy as np
import json
import multiprocessing as mp

import functools

from scipy.ndimage.filters import gaussian_filter
from scipy.special import expit

from skimage.transform import resize

from tensorflow.keras.models import load_model

from .component import Component, SeriesSummary
from ..configs import (
    DEFAULT_SLICE_PARAMS,
)
from ..utils import default


class SliceSelector(Component):
    def __init__(
        self,
        slice_selection_weights=None,
        slice_params=DEFAULT_SLICE_PARAMS,
        sigmoid_output=True,
        num_threads=8,
    ):

        super().__init__()

        self.slice_smoothing_kernel = 2.0
        if slice_selection_weights is None:
            slice_selection_weights = (
                "/body_comp/body_comp/models/regression_densenet_l12_k12.hdf5"
            )
        self.slice_selection_model = load_model(slice_selection_weights, compile=False)

        self.slice_params = slice_params.copy()

        self.slice_selection_input_shape = self.slice_selection_model.input_shape[1:3]
        self.sigmoid_scale = 10.0
        self.sigmoid_output = sigmoid_output
        self.num_threads = num_threads

        self.cohort_summary = {}

    @staticmethod
    def find_zero_crossings(predictions):
        # Find zero-crossings to choose the slice
        zero_crossings = []
        for s in range(len(predictions) - 1):
            if (predictions[s] < 0.0) != (predictions[s + 1] < 0.0):
                if abs(predictions[s]) < abs(predictions[s + 1]):
                    zero_crossings.append(s)
                else:
                    zero_crossings.append(s + 1)

        return zero_crossings

    def get_cohort_summary(self):
        return self.cohort_summary

    def save_cohort_summary(self, filename):
        with open(filename, "w") as outfile:
            json.dump(self.cohort_summary, outfile, default=default)

    def slice_selection_post_process_sigmoids(self, smoothed_predictions, z_locations):

        zero_crossings = self.find_zero_crossings(smoothed_predictions)

        L = len(smoothed_predictions)

        # If there are no zero-crossings, just choose the slice with the closest value to zero
        if len(zero_crossings) == 0:
            chosen_index = np.argmin(abs(smoothed_predictions))
        # Ideally there will be one zero-crossing, and this will be the chosen slice
        elif len(zero_crossings) == 1:
            chosen_index = zero_crossings[0]
        # For now just choose the first or the last, but this is quite dumb
        else:
            # Construct the ideal curve at the correct slice spacing to perform a correlation
            slice_spacing = abs(np.median(np.diff(z_locations)))
            if slice_spacing <= 0.0:
                # The slice spacings in this series are very odd
                # Fall back to simply choosing the last zero crossing
                chosen_index = zero_crossings[-1]
            else:
                ideal_array_half_length = np.floor(
                    (5.0 * self.sigmoid_scale) / slice_spacing
                )
                if (2 * ideal_array_half_length + 1) > L:
                    ideal_array_half_length = (L - 1) // 2
                z_lim = slice_spacing * ideal_array_half_length
                z = np.arange(-z_lim, z_lim, slice_spacing)
                ideal_curve = 2.0 * (expit(z) - 0.5)

                # Perform the correlation
                corr = np.correlate(smoothed_predictions, ideal_curve, mode="same")

                # Look for zero-crossing in the original curve that has the highest correlation value
                max_corr = -np.inf
                for x in zero_crossings:
                    if corr[x] > max_corr:
                        chosen_index = x
                        max_corr = corr[x]

        results_dict = {
            "index": int(chosen_index),
            "regression_val": float(smoothed_predictions[chosen_index]),
            "num_zero_crossings": len(zero_crossings),
        }

        return results_dict

    def slice_selection_post_process(self, smoothed_predictions, z_locations, z_pos):

        # Subtract the desired z position to get the predicted offset
        predicted_offsets = smoothed_predictions - z_pos
        n = len(smoothed_predictions)

        zero_crossings = self.find_zero_crossings(predicted_offsets)

        # If there are no zero-crossing, just choose the slice with the closest
        # value to zero
        if len(zero_crossings) == 0:
            chosen_index = np.argmin(abs(predicted_offsets))
        # Ideally there will be one zero-crossing, and this will be the chosen slice
        elif len(zero_crossings) == 1:
            chosen_index = zero_crossings[0]
        # For now just choose the first or the last, but this is quite dumb
        else:
            # Construct the ideal curve at the correct slice spacing to perform
            # a correlation
            slice_spacing = abs(np.median(np.diff(z_locations)))
            if slice_spacing <= 0.0:
                # The slice spacings in this series are very odd
                # Fall back to simply choosing the last zero crossing
                chosen_index = zero_crossings[-1]
            else:
                ideal_array_half_length = 10
                if (2 * ideal_array_half_length + 1) > n:
                    ideal_array_half_length = (n - 1) // 2
                z_lim = slice_spacing * ideal_array_half_length
                ideal_curve = np.linspace(
                    -z_lim, z_lim, 2 * ideal_array_half_length + 1
                )

                # Check all zero crossings
                min_rms_error = np.inf
                for zc in zero_crossings:
                    # Find area of overlap between filter and signal centered
                    # at this zero-crossing
                    start_filter = max(0, -(zc - ideal_array_half_length))
                    if n - zc - 1 < ideal_array_half_length:
                        end_filter = -ideal_array_half_length + n - 1 - zc
                    else:
                        end_filter = len(ideal_curve)
                    start_signal = max(0, zc - ideal_array_half_length)
                    end_signal = min(n, zc + ideal_array_half_length + 1)

                    # Find RMS error between the signal and the ideal
                    diff = (
                        predicted_offsets[start_signal:end_signal]
                        - ideal_curve[start_filter:end_filter]
                    )
                    rms = (diff**2).sum() ** 0.5
                    if rms < min_rms_error:
                        chosen_index = zc
                        min_rms_error = rms

        results_dict = {
            "index": int(chosen_index),
            "regression_val": float(smoothed_predictions[chosen_index]),
            "num_zero_crossings": len(zero_crossings),
        }

        return results_dict

    def slice_selection(self, series_list, z_locations, slices):

        # Resize each image and stack into a np array
        resize_func = functools.partial(
            resize,
            output_shape=self.slice_selection_input_shape,
            preserve_range=True,
            anti_aliasing=True,
            mode="constant",
        )
        if self.num_threads > 1:
            pool = mp.Pool(self.num_threads)
            series = pool.map(resize_func, series_list)
            series = np.dstack(series)
            pool.close()
        else:
            series = np.dstack([resize_func(im) for im in series_list])

        # Reshape the series for the network
        series = np.transpose(series[:, :, :, np.newaxis], [2, 0, 1, 3])

        # Predict the offsets for each image
        predictions = self.slice_selection_model.predict(series)
        if self.sigmoid_output:
            predictions = 2.0 * (predictions - 0.5)

        # Filter with a gaussian to smooth and take absolute value
        if self.slice_smoothing_kernel > 0.0:
            smoothing_kernel = (
                [self.slice_smoothing_kernel, 0.0]
                if self.sigmoid_output
                else self.slice_smoothing_kernel
            )
            smoothed_predictions = gaussian_filter(predictions, smoothing_kernel)
        else:
            smoothed_predictions = np.copy(predictions)

        if not self.sigmoid_output:
            smoothed_predictions = np.squeeze(smoothed_predictions)

        # Post procecheck_validct a single slice
        results_by_slice = {}
        for s in slices:
            if self.sigmoid_output:
                output_index = self.slice_params[s][
                    "slice_selection_model_output_index"
                ]
                results_by_slice[s] = self.slice_selection_post_process_sigmoids(
                    smoothed_predictions[:, output_index], z_locations
                )
            else:
                results_by_slice[s] = self.slice_selection_post_process(
                    smoothed_predictions, z_locations, self.slice_params[s]["z_pos"]
                )

        return results_by_slice, predictions, smoothed_predictions

    def F(self, series_summary: SeriesSummary):

        if not self.check_valid(series_summary):
            return series_summary

        slices = list(DEFAULT_SLICE_PARAMS.keys())

        series_results = {}

        results = {}

        # Perform slice selection
        results["slices"], raw_predictions, smoothed_predictions = self.slice_selection(
            series_summary["windowed_images"], series_summary["z_locations"], slices
        )

        series_results["results"] = results
        series_results["raw predictions"] = raw_predictions
        series_results["smoothed predictions"] = smoothed_predictions

        series_summary["slice selection"] = series_results

        return series_summary
