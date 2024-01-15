from highdicom.sr.coding import CodedConcept

from pydicom.sr.codedict import codes


DEFAULT_SLICE_PARAMS = {
    "L3": {
        "slice_selection_model_output_index": 0,
        "class_names": ["muscle", "subcutaneous_fat", "visceral_fat"],
        "model_weights": None,
        "regression_plot_colour": "red",
    }
}


# Dictionaries of tags at instances, series and study to put into the results
INSTANCE_LEVEL_TAGS = {
    "tube_current_mA": {"keyword": "XRayTubeCurrent", "type": float},
    "exposure_mAs": {"keyword": "Exposure", "type": float},
    "exposure_time_ms": {"keyword": "ExposureTime", "type": float},
    "kvp": {"keyword": "KVP", "type": float},
}

SERIES_LEVEL_TAGS = {
    "modality": {"keyword": "Modality", "type": str},
    "slice_thickness_mm": {"keyword": "SliceThickness", "type": float},
    "reconstruction_kernel": {"keyword": "ConvolutionKernel", "type": str},
    "contrast_bolus_agent": {"keyword": "ContrastBolusAgent", "type": str},
    "contrast_bolus_ingredient": {"keyword": "ContrastBolusIngredient", "type": str},
    "contrast_bolus_route": {"keyword": "ContrastBolusRoute", "type": str},
    "contrast_bolus_volume": {"keyword": "ContrastBolusVolume", "type": float},
    "manufacturer": {"keyword": "Manufacturer", "type": str},
    "manufacturer_model_name": {"keyword": "ManufacturerModelName", "type": str},
    "station_name": {"keyword": "StationName", "type": str},
    "series_description": {"keyword": "SeriesDescription", "type": str},
    "acquisition_date": {"keyword": "AcquisitionDate", "type": str},
    "acquisition_time": {"keyword": "AcquisitionTime", "type": str},
    "series_uid": {"keyword": "SeriesInstanceUID", "type": str},
}

STUDY_LEVEL_TAGS = {
    "patient_id": {"keyword": "PatientID", "type": str},
    "study_date": {"keyword": "StudyDate", "type": str},
    "accession_number": {"keyword": "AccessionNumber", "type": str},
    "study_description": {"keyword": "StudyDescription", "type": str},
}


MODEL_NAME = "CCDS Body Composition Estimation"
MANUFACTURER = "MGH & BWH Center for Clinical Data Science"
SERIAL_NUMBER = "1"


KNOWN_SEGMENT_DESCRIPTIONS = {
    "muscle": {
        "segment_label": "Muscle",
        "segmented_property_category": codes.SCT.Muscle,
        "segmented_property_type": codes.SCT.SkeletalMuscle,
    },
    "subcutaneous_fat": {
        "segment_label": "Subcutaneous Fat",
        "segmented_property_category": codes.SCT.BodyFat,
        "segmented_property_type": CodedConcept(
            "727176007", "SCT", "Entire subcutaneous fatty tissue"
        ),
    },
    "visceral_fat": {
        "segment_label": "Visceral Fat",
        "segmented_property_category": codes.SCT.BodyFat,
        "segmented_property_type": CodedConcept(
            "725274000", "SCT", "Entire adipose tissue of abdomen"
        ),
    },
}


MASK_COLOURS = (
    "black",  # background
    "red",  # muscle
    "green",  # visceral fat
    "yellow",  # subcutaneous_fat
)
