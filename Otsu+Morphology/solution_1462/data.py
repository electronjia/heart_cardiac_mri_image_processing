from os import listdir
from typing import Sequence, override

from numpy import ndarray, ogrid, sqrt, clip
from skimage import io


def enhance(image: ndarray) -> ndarray:
    height, width = image.shape
    cx, cy = int(width * .5), int(height * .5)
    y, x = ogrid[:height, :width]
    dist_from_center = sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_dist = dist_from_center.max()
    mask = 1 - (dist_from_center / max_dist) ** 2
    mask = clip(mask, 0, 1)
    return clip(image * mask, 0, 255)


class Dataset(Sequence):
    def __init__(self, root_dir: str, labelled: bool = False, cache: bool = False) -> None:
        self.root_dir: str = root_dir
        self.labelled: bool = labelled
        self.cache: bool = cache
        self._cases: list[str] = [case[:case.rfind(".")] for case in listdir(f"{root_dir}/cases")
                                  if case.endswith(".png")]
        self._case_cache: dict[str, ndarray] = {}
        self._label_cache: dict[str, ndarray] = {}

    def load_case(self, case_id: str) -> ndarray:
        return io.imread(f"{self.root_dir}/cases/{case_id}.png", as_gray=True)

    def get_case(self, case_id: str) -> ndarray:
        if case_id in self._case_cache:
            return self._case_cache[case_id]
        r = self.load_case(case_id)
        if self.cache:
            self._case_cache[case_id] = r
        return r

    def load_label(self, case_id: str) -> ndarray:
        return io.imread(f"{self.root_dir}/labels/{case_id}_label.png", as_gray=True)

    def get_label(self, case_id: str) -> ndarray:
        if case_id in self._label_cache:
            return self._label_cache[case_id]
        r = self.load_case(case_id)
        if self.cache:
            self._label_cache[case_id] = r
        return r

    @override
    def __getitem__(self, index: int) -> [str, ndarray, ndarray | None]:
        case_id = self._cases[index]
        return case_id, self.load_case(case_id), self.load_label(case_id) if self.labelled else None

    @override
    def __len__(self) -> int:
        return len(self._cases)
