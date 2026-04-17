from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

import constants as c


LandmarkArray = np.ndarray


@dataclass
class ExtractedFaceParts:
  """一帧中提取出的“局部 landmark 点集”。

  约定：所有数组形状均为 (N, 3)，三列依次为 (x, y, z)。
  - x/y 为归一化坐标（相对图像宽高，通常范围 0~1）
  - z 为相对深度（MediaPipe 的定义：单位与尺度不是毫米；更多是“相对深浅”）
  """

  left_eye: LandmarkArray
  right_eye: LandmarkArray
  left_iris: LandmarkArray
  right_iris: LandmarkArray


class FeatureExtractor:
  """从 MediaPipe Face Mesh 输出中提取你关心的点位集合。

  使用方式（典型在视频循环中）：
  - 每帧调用 `update(results.multi_face_landmarks)` 更新内部缓存
  - 然后按需调用 `get_eyes()` / `get_iris()` / `get_subset_numpy(...)`

  为什么需要这个类：
  - MediaPipe 返回的是一堆对象（NormalizedLandmark），用起来方便但不利于数值计算
  - 转成 NumPy 后更适合做几何计算、滤波、拟合、归一化、特征工程等
  """

  def __init__(self) -> None:
    # `raw_landmarks` 缓存的是“单张脸”的 landmark 列表：
    # 通常长度为 468（refine_landmarks=False）或 478（True，包含虹膜点）。
    self.raw_landmarks: Optional[Sequence[object]] = None

  def update(self, multi_face_landmarks) -> bool:
    """每帧更新原始 landmark 缓存。

    参数：
    - multi_face_landmarks：来自 `results.multi_face_landmarks`（可能为 None/空列表）

    返回：
    - True：本帧成功拿到第一张脸的 landmark
    - False：本帧无人脸或结果为空（内部缓存会清空）
    """

    if not multi_face_landmarks:
      self.raw_landmarks = None
      return False

    # 约定：只取第一张脸（与 demo 里 max_num_faces=1 一致）
    self.raw_landmarks = multi_face_landmarks[0].landmark
    return True

  def get_subset_numpy(self, indices: Iterable[int], *, dtype=np.float32) -> LandmarkArray:
    """按索引列表提取 (x, y, z) 并返回 NumPy 数组。

    输入：
    - indices：一组 landmark 索引（例如 `constants.LEFT_EYE`）

    输出：
    - shape 为 (N, 3) 的 np.ndarray，其中 N=len(indices)
    - 如果当前没有缓存到 landmark（还没 update 成功），返回 shape 为 (0, 3) 的空数组

    设计取舍：
    - 返回 (0, 3) 而不是 (0,)：这样下游代码在拼接/矩阵运算时更稳定
    """

    if self.raw_landmarks is None:
      return np.empty((0, 3), dtype=dtype)

    # 将对象列表转成数值矩阵：(x, y, z)
    # 这里用 list append 更直观；后续如需极致性能可改成预分配数组。
    subset = []
    for idx in indices:
      lm = self.raw_landmarks[idx]
      # lm 是 MediaPipe 的 NormalizedLandmark：含 x/y/z/visibility/presence 等字段
      subset.append([lm.x, lm.y, lm.z])

    return np.asarray(subset, dtype=dtype)

  def get_eyes(self) -> Tuple[LandmarkArray, LandmarkArray]:
    """返回左右眼轮廓点集（各自 shape (N, 3)）。"""

    left = self.get_subset_numpy(c.LEFT_EYE)
    right = self.get_subset_numpy(c.RIGHT_EYE)
    return left, right

  def get_iris(self) -> Tuple[LandmarkArray, LandmarkArray]:
    """返回左右虹膜点集（各自 shape (N, 3)）。"""

    left = self.get_subset_numpy(c.LEFT_IRIS)
    right = self.get_subset_numpy(c.RIGHT_IRIS)
    return left, right

  def get_all_common_parts(self) -> ExtractedFaceParts:
    """一次性提取 eyes + iris，便于下游统一处理。"""

    left_eye, right_eye = self.get_eyes()
    left_iris, right_iris = self.get_iris()
    return ExtractedFaceParts(
      left_eye=left_eye,
      right_eye=right_eye,
      left_iris=left_iris,
      right_iris=right_iris,
    )

