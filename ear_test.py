import time

import matplotlib.pyplot as plt
import numpy as np
import constants as c
import feature_extractor as fe
import cv2
import mediapipe as mp


def calculate_ear(eye_points):
    # eye_points 是 (6, 3) 的 numpy 数组
    # P2-P6, P3-P5 (纵向)
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    # P1-P4 (横向)
    C = np.linalg.norm(eye_points[0] - eye_points[3])

    ear = (A + B) / (2.0 * C)
    return ear


def main() -> None:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    feature_extractor = fe.FeatureExtractor()

    cap = cv2.VideoCapture(0)
    sample_dt = 0.1
    t0 = None
    next_sample_t = None
    ears_l: list[float] = []
    ears_r: list[float] = []
    ear_times_s: list[float] = []
    completed_measurement = False
    try:
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break

            h, w, _ = frame_bgr.shape
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            now = time.perf_counter()

            if feature_extractor.update(results.multi_face_landmarks):
                # 1) 眼睛 (绿色)
                left_eye, right_eye = feature_extractor.get_eyes()
                if t0 is None:
                    t0 = now
                    next_sample_t = t0 + sample_dt
                while (
                    next_sample_t is not None
                    and now >= next_sample_t
                    and next_sample_t <= t0 + 5.0
                ):
                    ear_l = calculate_ear(left_eye)
                    ear_r = calculate_ear(right_eye)
                    print(f"EAR: L={ear_l:.4f} R={ear_r:.4f}")
                    ears_l.append(ear_l)
                    ears_r.append(ear_r)
                    ear_times_s.append(next_sample_t - t0)
                    next_sample_t += sample_dt
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

            if t0 is not None and now - t0 >= 5.0:
                if ears_l:
                    print(
                        "5s 统计 — 均值 L={:.4f} R={:.4f} | 方差 L={:.6f} R={:.6f}".format(
                            float(np.mean(ears_l)),
                            float(np.mean(ears_r)),
                            float(np.var(ears_l)),
                            float(np.var(ears_r)),
                        )
                    )
                completed_measurement = True
                break

            cv2.imshow("EAR Test", frame_bgr)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()

    if completed_measurement and ears_l:
        _, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ear_times_s, ears_l, "o-", label="left eye EAR", color="#2ca02c")
        ax.plot(ear_times_s, ears_r, "o-", label="right eye EAR", color="#ff7f0e")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("EAR")
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title("5s EAR changes over time")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
