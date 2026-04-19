import numpy as np
import constants as c

class FeatureExtractor:
    def __init__(self):
        self.raw_landmarks = None

    def update(self, multi_face_landmarks):
        """每帧更新原始地标点数据"""
        if not multi_face_landmarks:
            self.raw_landmarks = None
            return False
        self.raw_landmarks = multi_face_landmarks[0].landmark
        return True

    def get_subset_numpy(self, indices):
        """
        核心方法：根据索引列表，提取对应的 (x, y, z) 并转为 NumPy 数组
        """
        if not self.raw_landmarks:
            return np.array([])
        
        # 提取指定的点位
        subset = []
        for idx in indices:
            lm = self.raw_landmarks[idx]
            subset.append([lm.x, lm.y, lm.z])
            
        return np.array(subset)

    def get_eyes(self):
        left = self.get_subset_numpy(c.LEFT_EYE)
        right = self.get_subset_numpy(c.RIGHT_EYE)
        return left, right

    def get_iris(self):
        left = self.get_subset_numpy(c.LEFT_IRIS)
        right = self.get_subset_numpy(c.RIGHT_IRIS)
        return left, right

    def get_gaze_vector(self, eye_points, iris_center):
        """
        计算虹膜中心相对于眼眶中心的偏移量
        eye_points: (6, 3) numpy 数组
        iris_center: (3,) numpy 数组 (468号点)
        """
        # 1. 计算眼眶几何中心 (使用 6 个点的平均值)
        eye_center = np.mean(eye_points, axis=0)

        # 2. 计算相对位移
        offset = iris_center - eye_center

        # 3. 归一化 (除以眼宽，消除距离影响)
        eye_width = np.linalg.norm(eye_points[0] - eye_points[3])
        normalized_offset = offset / eye_width

        return normalized_offset
