from dataclasses import dataclass
from math import ceil

from matplotlib import pyplot as plt
from numpy import ndarray, array
from pandas import DataFrame
from skimage import color, io
from tqdm import tqdm

from solution_1462.core import segment
from solution_1462.data import Dataset
from solution_1462.metric import compute_dice_coefficient, compute_intersection_over_union


def generate(dataset: Dataset, output_dir: str, label_suffix: str = "_label", sigma: float | None = None,
             threshold_offset: float | None = None) -> None:
    for case_id, image, _ in dataset:
        _, mask = segment(image, sigma, threshold_offset)
        io.imsave(f"{output_dir}/{case_id}{label_suffix}.png", mask)


def show_case(image: ndarray, cleaned: ndarray, mask: ndarray, label: ndarray, case_id: str = "") -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    if image.max() > 1:
        image = image / 255
    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title(f"Case {case_id}" if case_id else "Input Image")
    axes[0, 1].imshow(cleaned, cmap="gray")
    axes[0, 1].set_title("Otsu's Method")
    axes[1, 0].imshow(color.label2rgb(mask, image=image, bg_label=0))
    axes[1, 0].set_title("Predicted")
    axes[1, 1].imshow(color.label2rgb(label, image=image, bg_label=0))
    axes[1, 1].set_title("Ground Truth")
    for i in axes:
        for j in i:
            j.axis('off')
    plt.tight_layout()
    plt.show()


def evaluate(dataset: Dataset, output_csv: str = "", show_cases: int = 5, sigma: float | None = None,
             threshold_offset: float | None = None) -> dict[str, list[str] | list[float]]:
    if not dataset.labelled:
        raise ValueError("Dataset must be labelled")
    results = {"case_id": [], "dcs": [], "iou": []}
    logging_interval = int(len(dataset) / show_cases) if show_cases > 0 else -1
    i = 0
    for case_id, image, label in dataset:
        cleaned, mask = segment(image, sigma, threshold_offset)
        if logging_interval > 0 and i % logging_interval == 0:
            show_case(image, cleaned, mask, label, case_id)
        results["case_id"].append(case_id)
        label = label.astype(bool)
        results["dcs"].append(compute_dice_coefficient(mask, label))
        results["iou"].append(compute_intersection_over_union(mask, label))
        i += 1
    if output_csv:
        DataFrame(results).to_csv(output_csv, index=False)
    return results


@dataclass
class OptimizationResult(object):
    dcs: float
    params: tuple[float, float]


def optimize(dataset: Dataset, min_sigma: float = .6, max_sigma: float = 3, step_sigma: float = 1e-4,
             min_offset: float = 0, max_offset: float = 0.1, step_offset: float = 1e-5) -> OptimizationResult:
    if not dataset.labelled:
        raise ValueError("Dataset must be labelled")
    best_param = (min_sigma, min_offset)
    best_dcs = 0
    sigma, offset = min_sigma, min_offset
    with tqdm(total=ceil((max_sigma - min_sigma) / step_sigma) * ceil((max_offset - min_offset) / step_offset)) as pbar:
        while True:
            if sigma >= max_sigma:
                return OptimizationResult(best_dcs, best_param)
            if offset >= max_offset:
                offset = min_offset
                sigma += step_sigma
            results = evaluate(dataset, show_cases=0, sigma=sigma, threshold_offset=offset)
            dcs = array(results["dcs"])
            nonzero_dcs = dcs[dcs > 0]
            avg_dcs = nonzero_dcs.mean() if nonzero_dcs.size > 1 else 0
            if avg_dcs > best_dcs:
                best_dcs = avg_dcs
                best_param = (sigma, offset)
            offset += step_offset
            pbar.update()
