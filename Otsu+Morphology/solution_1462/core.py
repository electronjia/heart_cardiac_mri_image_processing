from typing import Any

from numpy import ndarray, zeros_like
from skimage import filters, morphology, measure


def distance(p0: tuple[float, float], p1: tuple[float, float]) -> float:
    return ((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2) ** .5


def select_center(labels: ndarray, props: list[Any], center: tuple[int, int]) -> Any:
    x, y = center
    if (target_label := labels[y][x]) > 0:
        for prop in props:
            if prop.label == target_label:
                return prop
    return min(props, key=lambda e: distance(e.centroid, center))


def segment(image: ndarray, sigma: float | None = None,
            threshold_offset: float | None = None,
            eccentricity_threshold: float | None = None) -> tuple[ndarray, ndarray]:
    sigma = .515 if sigma is None else sigma
    threshold_offset = .026 if threshold_offset is None else threshold_offset
    eccentricity_threshold = .85 if eccentricity_threshold is None else eccentricity_threshold
    smoothed = filters.gaussian(image, sigma)
    thresh_value = filters.threshold_otsu(smoothed)
    binary = smoothed > (thresh_value + threshold_offset)
    cleaned = morphology.remove_small_objects(binary, min_size=50)
    cleaned = morphology.remove_small_holes(cleaned, area_threshold=50)
    labels = measure.label(cleaned)
    props = measure.regionprops(labels)
    if len(props) < 1:
        return cleaned, zeros_like(image).astype(bool)
    w, h = image.shape
    final_mask = select_center(labels, props, (int(w * .5), int(h * .5)))
    if final_mask.eccentricity > eccentricity_threshold:
        return cleaned, zeros_like(image).astype(bool)
    return cleaned, morphology.binary_opening(labels == final_mask.label, morphology.disk(3))
