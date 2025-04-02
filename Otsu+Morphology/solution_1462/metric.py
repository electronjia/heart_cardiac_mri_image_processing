from numpy import ndarray


def compute_dice_coefficient(mask: ndarray, label: ndarray) -> float:
    volume_sum = label.sum() + mask.sum()
    if volume_sum == 0:
        return 0
    return float(2 * (mask & label).sum() / volume_sum)


def compute_intersection_over_union(mask: ndarray, label: ndarray) -> float:
    denominator = (mask | label).sum()
    if denominator == 0:
        return 0
    return float((mask & label).sum() / denominator)
