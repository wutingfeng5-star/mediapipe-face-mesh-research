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
    try:
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break

            h, w, _ = frame_bgr.shape
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if feature_extractor.update(results.multi_face_landmarks):
                # 1) 眼睛 (绿色)
                left_eye, right_eye = feature_extractor.get_eyes()
                for pt in np.vstack((left_eye, right_eye)):
                    pos = (int(pt[0] * w), int(pt[1] * h))
                    cv2.circle(frame_bgr, pos, 2, (0, 255, 0), -1)

                # 2) 虹膜 (红色)
                left_iris, right_iris = feature_extractor.get_iris()
                for pt in np.vstack((left_iris, right_iris)):
                    pos = (int(pt[0] * w), int(pt[1] * h))
                    cv2.circle(frame_bgr, pos, 2, (0, 0, 255), -1)

                # 3) 脸部轮廓 (蓝色)
                face_oval = feature_extractor.get_subset_numpy(c.FACE_OVAL)
                for pt in face_oval:
                    pos = (int(pt[0] * w), int(pt[1] * h))
                    cv2.circle(frame_bgr, pos, 1, (255, 0, 0), -1)

            cv2.imshow("Feature Extraction Verification", frame_bgr)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
