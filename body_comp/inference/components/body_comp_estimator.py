import importlib.resources
import os
from tempfile import TemporaryDirectory

from imageio import imsave, imread

from tensorflow.keras.models import load_model

from highdicom.content import AlgorithmIdentificationSequence
from highdicom.seg.content import SegmentDescription
from highdicom.seg.enum import SegmentAlgorithmTypeValues, SegmentationTypeValues
from highdicom.seg.sop import Segmentation

import numpy as np

import pydicom
from pydicom.sr.codedict import codes
from pydicom.uid import ExplicitVRLittleEndian

from skimage.transform import resize
from skimage.color.colorlabel import label2rgb
from skimage.color import gray2rgb

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


import body_comp
from .component import Component
from ..utils import get_dicom_metadata, apply_window
from ..configs import (
    INSTANCE_LEVEL_TAGS,
    MODEL_NAME,
    MANUFACTURER,
    SERIAL_NUMBER,
    KNOWN_SEGMENT_DESCRIPTIONS,
    MASK_COLOURS,
    DEFAULT_SLICE_PARAMS,
)


class BodyCompositionEstimator(Component):
    """A class that encapsulates the models and processes required to perform body composition analysis on
    CT images.
    """

    def __init__(
        self,
        slice_params=DEFAULT_SLICE_PARAMS,
        sigmoid_output=True,
        num_threads=8,
        algorithm_version="0.1.0",
        output_dir=None,
        segmentation_range=None,
        keep_existing=False,
        dicom_seg=False,
        slice_selection=True,
        win_centre=40.0,
        win_width=400,
    ):

        self.sigmoid_output = sigmoid_output
        self.num_threads = num_threads
        self.algorithm_version = algorithm_version
        self.output_dir = output_dir
        self.keep_existing = keep_existing
        self.segmentation_range = segmentation_range
        self.dicom_seg = dicom_seg
        self.slice_selection = slice_selection
        self.win_centre = win_centre
        self.win_width = win_width

        # Read in all the segmentation models required
        self.slice_params = slice_params.copy()
        for s in self.slice_params.keys():
            # If not specified, assume the default segmentation model installed
            # with the package
            if self.slice_params[s]["model_weights"] is None:
                self.slice_params[s][
                    "model_weights"
                ] = (
                    importlib.resources.files(body_comp) /
                    "models/segmentation_unet_d5_i16_c1.hdf5"
                )
        unique_seg_models = list(
            set([v["model_weights"] for v in self.slice_params.values()])
        )
        segmentation_models = [
            load_model(weights, compile=False) for weights in unique_seg_models
        ]

        for s in self.slice_params.keys():
            model_index = unique_seg_models.index(
                (self.slice_params[s]["model_weights"])
            )
            self.slice_params[s]["model"] = segmentation_models[model_index]
            self.slice_params[s]["segmentation_input_shape"] = segmentation_models[
                model_index
            ].input_shape[1:3]

        self.init_directories()

    def init_directories(self):

        # Make sure the output directories exist and are empty
        if os.path.exists(self.output_dir) and not self.keep_existing:
            if len(os.listdir(self.output_dir)) > 0:
                raise RuntimeError("Output directory is not empty. Exiting.")
        os.makedirs(self.output_dir, exist_ok=True)

        self.preview_output_dir = os.path.join(self.output_dir, "previews")
        os.makedirs(self.preview_output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "ready_for_qa"), exist_ok=True)

        if self.segmentation_range is not None:
            self.all_slices_output_dir = os.path.join(self.output_dir, "all_slices")
            os.makedirs(self.all_slices_output_dir, exist_ok=True)
        else:
            self.all_slices_output_dir = None

        if self.dicom_seg:
            self.seg_output_dir = os.path.join(self.output_dir, "dicom_seg")
            os.makedirs(self.seg_output_dir, exist_ok=True)

    # Perform the segmentation on a single image
    def segmentation(self, image, slice_name):

        # Resize and reshape
        orig_shape = image.shape
        req_shape = self.slice_params[slice_name]["segmentation_input_shape"]
        if image.shape != req_shape:
            image = resize(
                image,
                req_shape,
                preserve_range=True,
                anti_aliasing=True,
                mode="constant",
            )
        image = np.transpose(image[:, :, np.newaxis, np.newaxis], (2, 0, 1, 3))

        segmentation_predictions = self.slice_params[slice_name]["model"].predict(image)
        segmentation_mask = np.argmax(segmentation_predictions, axis=3)
        segmentation_mask = np.squeeze(segmentation_mask)

        # Resize the segmentation mask if needed
        if segmentation_mask.shape != orig_shape:
            segmentation_mask = resize(
                segmentation_mask, orig_shape, order=0, preserve_range=True
            ).astype(int)

        return segmentation_mask

    @staticmethod
    def check_boundary(mask, rec_radius_pix, num_classes):
        # Check the whether the segmentation touches the edge of the fov

        # Create an image of the radius from the centre
        m, n = mask.shape
        grid = np.mgrid[:m, :n]
        centre = np.array([[[(m - 1) / 2.0]], [[(n - 1) / 2.0]]])
        r = ((grid - centre) ** 2).sum(axis=0) ** 0.5

        # Threshold the image a few pixels inside the edge of the reconstruction area
        boundary_region = r > (rec_radius_pix - 2.0)

        # Edges of the image are always the boundary region
        boundary_region[0, :] = True
        boundary_region[-1, :] = True
        boundary_region[:, 0] = True
        boundary_region[:, -1] = True

        # Return the overlap between this boundary region and any positive class in the segmentation
        # mask
        per_class_boundary_checks = [
            int(np.logical_and(mask == c, boundary_region).sum())
            for c in range(1, num_classes + 1)
        ]
        return per_class_boundary_checks

    @staticmethod
    def find_pixel_statistics(pixels):
        results = {}
        if len(pixels) < 1:
            results["mean_hu"] = None
            results["median_hu"] = None
            results["std_hu"] = None
            results["iqr_hu"] = None
        else:
            results["mean_hu"] = float(pixels.mean())
            results["median_hu"] = float(np.median(pixels))
            results["std_hu"] = float(np.std(pixels))
            q75, q25 = np.percentile(pixels, [75, 25])
            results["iqr_hu"] = float(q75 - q25)
        return results

    def plot_regression(
        self,
        file_name,
        z_locations=None,
        predictions=None,
        smoothed_predictions=None,
        chosen_index=None,
        gt_location=None,
    ):

        fig = plt.figure(figsize=(4, 4))

        # if slice selection wasn't performed beforehand, save dummy fig instead
        if self.slice_selection:
            z_locations = np.array(z_locations)
            if self.sigmoid_output:
                for s in self.slice_params.keys():
                    index = self.slice_params[s]["slice_selection_model_output_index"]
                    colour = self.slice_params[s]["regression_plot_colour"]
                    plt.plot(
                        z_locations,
                        predictions[:, index],
                        c=colour,
                        linestyle=":",
                        label=s.upper() + " Prediction",
                    )
                    plt.plot(
                        z_locations,
                        smoothed_predictions[:, index],
                        c=colour,
                        label=s.upper() + " Smoothed Prediction",
                    )
            else:
                plt.plot(
                    z_locations,
                    predictions,
                    c="gray",
                    linestyle=":",
                    label="Prediction",
                )
                plt.plot(
                    z_locations,
                    smoothed_predictions,
                    c="gray",
                    label="Smoothed Prediction",
                )
            for s, ind in chosen_index.items():
                colour = self.slice_params[s]["regression_plot_colour"]
                plt.axvline(x=z_locations[ind], c=colour, linestyle="-", linewidth=0.3)
            if gt_location is not None:
                for s, zloc in gt_location.items():
                    colour = self.slice_params[s]["regression_plot_colour"]
                    plt.axvline(x=zloc, c=colour, linestyle="-.", linewidth=0.3)

            plt.legend()

        plt.grid(True)
        plt.ylabel("Prediction")
        plt.xlabel("Z-axis (mm)")

        fig.savefig(file_name, dpi=512 / 4)
        plt.close(fig)

    def make_dicom_seg(self, mask_list, dcm_list, class_names):

        if len(mask_list) > 1:
            mask = np.stack(mask_list, axis=0).astype(np.uint8)
        else:
            mask = mask_list[0].astype(np.uint8)

        # Describe the algorithm that created the segmentation
        algorithm_identification = AlgorithmIdentificationSequence(
            name=MODEL_NAME,
            version=self.algorithm_version,
            family=codes.cid7162.ArtificialIntelligence,
        )

        # Check that we have descriptions for all the segments
        for c in class_names:
            if c not in KNOWN_SEGMENT_DESCRIPTIONS:
                raise KeyError(f"There is no known segment description for class {c}")

        # Describe the segment
        segment_descriptions = [
            SegmentDescription(
                segment_number=i,
                segment_label=KNOWN_SEGMENT_DESCRIPTIONS[tis]["segment_label"],
                segmented_property_category=KNOWN_SEGMENT_DESCRIPTIONS[tis][
                    "segmented_property_category"
                ],
                segmented_property_type=KNOWN_SEGMENT_DESCRIPTIONS[tis][
                    "segmented_property_type"
                ],
                algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC,
                algorithm_identification=algorithm_identification,
            )
            for i, tis in enumerate(class_names, start=1)
        ]

        # Create the Segmentation instance
        seg_dataset = Segmentation(
            source_images=dcm_list,
            pixel_array=mask,
            segmentation_type=SegmentationTypeValues.BINARY,
            segment_descriptions=segment_descriptions,
            series_instance_uid=pydicom.uid.generate_uid(),
            series_number=100,
            sop_instance_uid=pydicom.uid.generate_uid(),
            instance_number=1,
            manufacturer=MANUFACTURER,
            manufacturer_model_name=MODEL_NAME,
            software_versions=self.algorithm_version,
            transfer_syntax_uid=ExplicitVRLittleEndian,
            device_serial_number=SERIAL_NUMBER,
        )

        return seg_dataset

    def save_image_results(self, study_name, series_summary, output_plot):

        if not self.check_valid(series_summary):
            return None

        # Create a preview image containing the central image and the regression plot
        reg_image = imread(output_plot)
        reg_image = reg_image[:, :, :3]
        os.remove(output_plot)

        # Place blank space next to the regression image
        reg_image_padded = np.hstack([reg_image, 255 * np.ones_like(reg_image)])
        preview_panels = [reg_image_padded]

        slices = list(series_summary["body composition"]["image_results"].keys())

        for s in slices:
            if self.segmentation_range is None:
                mask = series_summary["body composition"]["image_results"][s][
                    "seg_mask"
                ]
                image = series_summary["body composition"]["image_results"][s]["image"]
            else:
                chosen_slice_uid = series_summary["body composition"]["results"][
                    "slices"
                ][s]["sop_instance_uid"]
                # Need to find the correct element of the list for the chosen slice
                for slice_ind, slice_data in enumerate(
                    series_summary["body composition"]["results"]["slices"][s][
                        "individual"
                    ]
                ):
                    if slice_data["sop_instance_uid"] == chosen_slice_uid:
                        list_index = slice_ind
                        break
                mask = series_summary["body composition"]["image_results"][s][
                    "seg_mask"
                ][list_index]
                image = series_summary["body composition"]["image_results"][s]["image"][
                    list_index
                ]
            image = apply_window(image, self.win_centre, self.win_width)
            if image.shape != reg_image.shape[:2]:
                image = resize(
                    image, reg_image.shape[:2], preserve_range=True, clip=False
                )
            if mask.shape != reg_image.shape[:2]:
                mask = resize(
                    mask, reg_image.shape[:2], preserve_range=True, clip=False, order=0
                )
            colour_mask = (
                label2rgb(mask, colors=MASK_COLOURS, bg_label=-1) * 255
            ).astype(np.uint8)
            colour_image = gray2rgb(image).astype(np.uint8)
            output_image = np.hstack([colour_image, colour_mask])
            preview_panels.append(output_image)

        # Stack the panels for each slice
        preview_image_output = np.vstack(preview_panels).astype(np.uint8)
        image_path = os.path.join(
            self.preview_output_dir,
            "{}_{}_preview.png".format(study_name, series_summary["PrimaryUID"]),
        )

        imsave(image_path, preview_image_output)

        # Store all images and masks when multislice analysis was used
        if self.segmentation_range is not None:

            # Create a new subdirectory to hold images for this series
            series_output_dir = os.path.join(
                self.all_slices_output_dir,
                "{}_{}".format(study_name, series_summary["PrimaryUID"]),
            )
            os.makedirs(series_output_dir, exist_ok=True)

            for s in series_summary["body composition"]["image_results"].keys():

                masks_list = series_summary["body composition"]["image_results"][s][
                    "seg_mask"
                ]
                images = series_summary["body composition"]["image_results"][s]["image"]

                for j, (im, mask) in enumerate(zip(images, masks_list)):

                    # Change the mask to colour and chosen image to RGB
                    mask = (
                        label2rgb(mask, colors=MASK_COLOURS, bg_label=-1) * 255
                    ).astype(np.uint8)

                    im = apply_window(im, self.win_centre, self.win_width)
                    im = gray2rgb(im).astype(np.uint8)

                    composite_image = np.hstack([im, mask]).astype(np.uint8)

                    image_path = os.path.join(
                        series_output_dir, "{}_{}.png".format(s, j)
                    )
                    imsave(image_path, composite_image)

    def F(
        self, series_summary, slices=None, save_plot="", gt_index=None, gt_location=None
    ):

        if not self.check_valid(series_summary):
            return series_summary

        # Check that the requested slices are understood
        if slices is None:
            slices = list(self.slice_params.keys())
        else:
            for s in slices:
                if s not in self.slice_params:
                    raise ValueError(
                        "Unrecognised slice '{}', recognised values are: [{}]".format(
                            s, ", ".join(self.slice_params.keys())
                        )
                    )

        # Get the ground_truth z location (if needed)
        if gt_location is not None and gt_location is not None:
            raise ValueError("Provide either gt_location or gt_index, but not both")
        if gt_location is None and gt_index is not None:
            gt_location = {
                s: float(series_summary["datasets"][gti].ImagePositionPatient[2])
                for s, gti in gt_location.items()
            }

        dataset_list = series_summary["datasets"]

        images, windowed_images, z_locations = (
            series_summary["images"],
            series_summary["windowed_images"],
            series_summary["z_locations"],
        )

        results, image_results = {}, {}
        results["slices"] = {}

        # Iterate through the requested levels
        for s in slices:

            results["slices"][s] = {}
            image_results[s] = {}

            # Only expect chosen image index value if slice selection was already performed
            if self.slice_selection:
                chosen_image_index = series_summary["slice selection"]["results"][
                    "slices"
                ][s]["index"]
            else:
                chosen_image_index = 0

            results["slices"][s]["sop_instance_uid"] = dataset_list[
                chosen_image_index
            ].SOPInstanceUID
            results["slices"][s]["z_location"] = z_locations[chosen_image_index]
            num_classes = len(self.slice_params[s]["class_names"])

            # Perform segmentation
            if self.segmentation_range is not None:

                smoothed_predictions = series_summary["slice selection"][
                    "smoothed predictions"
                ]

                # Create a list to contain results for each slice within the range
                results["slices"][s]["individual"] = []
                results["slices"][s]["overall"] = {}
                results["slices"][s]["overall"]["tissues"] = {}

                image_results[s]["seg_mask"] = []
                image_results[s]["image"] = []

                chosen_z = z_locations[chosen_image_index]

                # Checks on the boundary region that will accumulate over the entire region
                overall_boundary_checks = [None] * num_classes

                chosen_datasets = []

                combined_seg_values = {
                    tis: [] for tis in self.slice_params[s]["class_names"]
                }

                # Initialise the 'overall' results
                for tis in self.slice_params[s]["class_names"]:
                    results["slices"][s]["overall"]["tissues"][tis] = {}
                    results["slices"][s]["overall"]["tissues"][tis]["area_cm2"] = 0.0

                # Loop through slices
                for ind, (dcm, z, im, wim) in enumerate(
                    zip(dataset_list, z_locations, images, windowed_images)
                ):
                    if abs(z - chosen_z) <= self.segmentation_range:
                        # Results dictionary for this individual slice
                        slice_results = {}
                        slice_results["z_location"] = z
                        slice_results["index"] = ind
                        slice_results["sop_instance_uid"] = dcm.SOPInstanceUID
                        slice_results["offset_from_chosen"] = z - chosen_z
                        slice_results["tissues"] = {
                            tis: {} for tis in self.slice_params[s]["class_names"]
                        }
                        chosen_datasets.append(dcm)

                        if self.sigmoid_output:
                            output_index = self.slice_params[s][
                                "slice_selection_model_output_index"
                            ]
                            slice_results["regression_val"] = float(
                                smoothed_predictions[ind, output_index]
                            )
                        else:
                            slice_results["regression_val"] = float(
                                smoothed_predictions[ind]
                            )

                        # Find the area of a single pixel
                        pixel_spacing = dcm.PixelSpacing
                        pixel_area = (
                            float(pixel_spacing[0]) * float(pixel_spacing[1]) / 100.0
                        )

                        # Run segmentation
                        mask = self.segmentation(wim, s)
                        image_results[s]["seg_mask"].append(mask)
                        image_results[s]["image"].append(im)

                        # Find the reconstruction radius in pixel units and use it to perform a boundary check if it's
                        # there
                        if "ReconstructionDiameter" in dcm:
                            rec_diameter = float(dcm.ReconstructionDiameter)
                            rec_radius_pix = 0.5 * (
                                rec_diameter / float(pixel_spacing[0])
                            )
                            boundary_checks = self.check_boundary(
                                mask, rec_radius_pix, num_classes
                            )
                            # Update the total boundary_checks
                            for cind, bc in enumerate(boundary_checks):
                                if overall_boundary_checks[cind] is None:
                                    overall_boundary_checks[cind] = bc
                                else:
                                    overall_boundary_checks[cind] += bc
                        else:
                            boundary_checks = [None] * num_classes

                        # Store the per-slice boundary check results
                        for tis, bc in zip(
                            self.slice_params[s]["class_names"], boundary_checks
                        ):
                            slice_results["tissues"][tis]["boundary_check"] = bc

                        # Add slice level DICOM metadata
                        slice_results = {
                            **slice_results,
                            **get_dicom_metadata(dcm, INSTANCE_LEVEL_TAGS),
                        }

                        # Results for each tissue
                        for i, tis in enumerate(self.slice_params[s]["class_names"]):
                            c = i + 1  # offset of one due to background class
                            pixel_count = (mask == c).sum()
                            area = float(pixel_count * pixel_area)

                            slice_results["tissues"][tis]["area_cm2"] = area
                            results["slices"][s]["overall"]["tissues"][tis][
                                "area_cm2"
                            ] += area
                            seg_values = im[mask == c]
                            combined_seg_values[tis].append(seg_values)
                            slice_results["tissues"][tis] = {
                                **slice_results["tissues"][tis],
                                **self.find_pixel_statistics(seg_values),
                            }
                        results["slices"][s]["individual"].append(slice_results)

                # Find aggregate 'overall' results by combining results from different slices
                for tis, bc in zip(
                    self.slice_params[s]["class_names"], overall_boundary_checks
                ):
                    results["slices"][s]["overall"]["tissues"][tis]["area_cm2"] /= len(
                        results["slices"][s]["individual"]
                    )
                    results["slices"][s]["overall"]["tissues"][tis][
                        "boundary_check"
                    ] = (int(bc) if bc is not None else None)
                    if len(combined_seg_values[tis]) > 0:
                        combined_seg_values_arr = np.hstack(combined_seg_values[tis])
                        results["slices"][s]["overall"]["tissues"][tis] = {
                            **self.find_pixel_statistics(combined_seg_values_arr),
                            **results["slices"][s]["overall"]["tissues"][tis],
                        }
                    else:
                        for key in ["mean_hu", "median_hu", "std_hu", "iqr_hu"]:
                            results["slices"][s]["overall"]["tissues"][tis][
                                key
                            ] = np.nan

                if self.dicom_seg:
                    try:
                        image_results[s]["dicom_seg"] = self.make_dicom_seg(
                            image_results[s]["seg_mask"],
                            chosen_datasets,
                            self.slice_params[s]["class_names"],
                        )
                    except Exception as e:
                        results["slices"][s]["dicom_seg_error"] = str(e)
                        image_results[s]["dicom_seg"] = False

            else:
                dcm = dataset_list[chosen_image_index]

                # Perform segmentation
                seg_mask = self.segmentation(windowed_images[chosen_image_index], s)

                # Store results
                image_results[s]["seg_mask"] = seg_mask
                image_results[s]["image"] = images[chosen_image_index]
                if self.dicom_seg:
                    try:
                        image_results[s]["dicom_seg"] = self.make_dicom_seg(
                            [seg_mask], [dcm], self.slice_params[s]["class_names"]
                        )
                    except Exception as e:
                        results["slices"][s]["dicom_seg_error"] = str(e)
                        image_results[s]["dicom_seg"] = False

                pixel_spacing = dcm.PixelSpacing
                pixel_area = float(pixel_spacing[0]) * float(pixel_spacing[1]) / 100.0

                # Find the reconstruction radius in pixel units and use it to
                # perform a boundary check if it's there
                if "ReconstructionDiameter" in dcm:
                    rec_diameter = float(dcm.ReconstructionDiameter)
                    rec_radius_pix = 0.5 * (rec_diameter / float(pixel_spacing[0]))
                    boundary_checks = self.check_boundary(
                        seg_mask, rec_radius_pix, num_classes
                    )
                else:
                    boundary_checks = [None] * num_classes

                # Loop over tissues and calculate metrics
                results["slices"][s]["tissues"] = {}
                for i, (tis, bc) in enumerate(
                    zip(self.slice_params[s]["class_names"], boundary_checks)
                ):
                    c = i + 1  # offset of one due to background class
                    results["slices"][s]["tissues"][tis] = {}
                    pixel_count = (seg_mask == c).sum()
                    results["slices"][s]["tissues"][tis]["area_cm2"] = float(
                        pixel_count * pixel_area
                    )
                    results["slices"][s]["tissues"][tis]["boundary_check"] = bc
                    seg_values = images[chosen_image_index][seg_mask == c]
                    if len(seg_values) == 0:
                        for key in ["mean_hu", "median_hu", "std_hu", "iqr_hu"]:
                            results["slices"][s]["tissues"][tis][key] = np.nan
                    else:
                        # Add pixel statistics to the result
                        results["slices"][s]["tissues"][tis] = {
                            **self.find_pixel_statistics(seg_values),
                            **results["slices"][s]["tissues"][tis],
                        }

                # Add slice level DICOM metadata
                results["slices"][s] = {
                    **results["slices"][s],
                    **get_dicom_metadata(dcm, INSTANCE_LEVEL_TAGS),
                }

        # Save a plot
        if len(save_plot) > 0:
            if self.slice_selection:
                chosen_indices = {
                    s: series_summary["slice selection"]["results"]["slices"][s][
                        "index"
                    ]
                    for s in slices
                }
                self.plot_regression(
                    save_plot,
                    z_locations,
                    np.array(series_summary["slice selection"]["raw predictions"]),
                    np.array(series_summary["slice selection"]["smoothed predictions"]),
                    chosen_indices,
                    gt_location=gt_location,
                )
            else:
                self.plot_regression(save_plot)

        body_composition_results = {"results": results, "image_results": image_results}

        series_summary["body composition"] = body_composition_results

        series_summary["droppable_keys"].append("body composition/image_results")

        return series_summary

    def apply(self, study_summary):

        slices = list(self.slice_params.keys())

        with TemporaryDirectory() as intermediate_output_dir:

            output_plot = os.path.join(
                intermediate_output_dir,
                study_summary["study_name"] + "_{}_regression.png",
            )

            for series_uid, series_summary in study_summary["UIDs"].items():

                if self.keep_existing and "body composition" in series_summary:
                    continue

                plot_name = output_plot.format(series_uid)

                try:
                    study_summary["UIDs"][series_uid] = self.F(
                        series_summary, slices, save_plot=plot_name
                    )
                except Exception as e:
                    study_summary["UIDs"][series_uid]["body composition"] = str(e)
                    continue

                if self.check_valid(series_summary):
                    if self.dicom_seg:
                        for slice_name, slice_results in series_summary[
                            "body composition"
                        ]["image_results"].items():
                            if slice_results["dicom_seg"]:
                                seg_file = os.path.join(
                                    self.seg_output_dir,
                                    "{}_{}_{}.dcm".format(
                                        study_summary["study_name"],
                                        series_uid,
                                        slice_name,
                                    ),
                                )
                                slice_results["dicom_seg"].save_as(seg_file)

                # save image results
                self.save_image_results(
                    study_name=study_summary["study_name"],
                    series_summary=series_summary,
                    output_plot=plot_name,
                )

        return study_summary
