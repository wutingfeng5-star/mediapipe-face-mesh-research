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
