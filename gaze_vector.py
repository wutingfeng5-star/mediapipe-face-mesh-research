import numpy as np
import constants as c
import feature_extractor as fe
import cv2
import mediapipe as mp


def main() -> None:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    feature_extractor = fe.FeatureExtractor()

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, image= cap.read()
        if not ret:
            break

        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        if feature_extractor.update(results.multi_face_landmarks):
    
            # --- 提取左眼数据 ---
            # 获取左眼眶的 6 个关键点 (numpy array)
            left_eye_pts, _ = feature_extractor.get_eyes() 
            # 获取左虹膜中心 (468号点)
            left_iris_pts, _ = feature_extractor.get_iris() 
            iris_center_raw = left_iris_pts[0] # 取 468 号点

            # --- 1. 计算眼眶几何中心 (像素坐标) ---
            # 先在归一化空间算均值，再映射到像素
            eye_center_norm = np.mean(left_eye_pts, axis=0)
            eye_center_px = (int(eye_center_norm[0] * w), int(eye_center_norm[1] * h))

            # --- 2. 计算虹膜中心 (像素坐标) ---
            iris_center_px = (int(iris_center_raw[0] * w), int(iris_center_raw[1] * h))

            # --- 3. 绘制 Ground Truth 2.0 可视化 ---
            # A. 画眼眶中心 (黄色小圆点)
            cv2.circle(image, eye_center_px, 2, (0, 255, 255), -1)
            
            # B. 画虹膜中心 (红色小圆点)
            cv2.circle(image, iris_center_px, 2, (0, 0, 255), -1)

            # C. 画指向线 (绿色直线)
            # 增加一个放大系数，让微小的偏移更明显（可选）
            cv2.line(image, eye_center_px, iris_center_px, (0, 255, 0), 2)

            # D. 分别显示 x、y 方向分量长度 (归一化坐标，用于定量分析)
            vec_x = abs(float(left_iris_pts[0][0] - eye_center_norm[0]))
            vec_y = abs(float(left_iris_pts[0][1] - eye_center_norm[1]))
            cv2.putText(image, f"Gaze X: {vec_x:.4f}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Gaze Y: {vec_y:.4f}", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Gaze Vector", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
