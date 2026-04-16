import cv2
import mediapipe as mp
import numpy as np


def _resolve_mp_solutions():
  """解析并返回可用的 Solutions 入口（强制优先 pip 安装版）。"""

  import importlib
  import sys
  from pathlib import Path

  repo_root = Path(__file__).resolve().parent

  def is_repo_checkout_module(module) -> bool:
    module_file = getattr(module, "__file__", None)
    if not module_file:
      return False
    try:
      return Path(module_file).resolve().is_relative_to(repo_root)
    except Exception:
      return str(repo_root).lower() in str(module_file).lower()

  def import_pip_mediapipe():
    """尽量从 site-packages 导入 mediapipe（避免导入仓库源码树）。"""
    original_sys_path = list(sys.path)
    try:
      filtered = []
      for p in original_sys_path:
        if not p:
          # 空字符串代表“当前工作目录”，在仓库根目录运行时会把源码树排到最高优先级
          continue
        try:
          if Path(p).resolve().is_relative_to(repo_root):
            continue
        except Exception:
          if str(repo_root).lower() in str(p).lower():
            continue
        filtered.append(p)
      sys.path[:] = filtered
      importlib.invalidate_caches()
      sys.modules.pop("mediapipe", None)
      return importlib.import_module("mediapipe")
    finally:
      sys.path[:] = original_sys_path

  # 1) 当前导入的 mediapipe 如果不是源码树且有 solutions，就直接用
  if hasattr(mp, "solutions") and not is_repo_checkout_module(mp):
    return mp.solutions  # type: ignore[attr-defined]

  # 2) 尝试强制导入 pip 版 mediapipe
  pip_mp = None
  try:
    pip_mp = import_pip_mediapipe()
  except ModuleNotFoundError:
    pip_mp = None

  if pip_mp is not None and hasattr(pip_mp, "solutions"):
    return pip_mp.solutions  # type: ignore[attr-defined]

  # 3) 仍失败：说明当前环境没有安装 pip 版 mediapipe（或被路径遮蔽）
  raise ModuleNotFoundError(
      "未能导入到带 mp.solutions 的 mediapipe（pip 安装版）。\n"
      "请在已激活的环境里安装：\n"
      "  python -m pip install mediapipe opencv-python numpy\n"
      "并重新打开终端后再运行本脚本。"
  )


mp_solutions = _resolve_mp_solutions()


def main() -> None:
  # 初始化
  mp_face_mesh = mp_solutions.face_mesh
  face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

  cap = cv2.VideoCapture(0)

  print("按下 's' 键记录当前距离的数据，按下 'q' 退出")

  try:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        break

      # 获取图像尺寸（动态获取更准确）
      h, w, _ = image.shape
      results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

      last_metrics = None
      if results.multi_face_landmarks:
        lms = results.multi_face_landmarks[0].landmark

        # 1. 提取关键点 (归一化坐标)
        p_nose = np.array([lms[0].x, lms[0].y, lms[0].z])
        p_left_side = np.array([lms[234].x, lms[234].y, lms[234].z])
        p_right_side = np.array([lms[454].x, lms[454].y, lms[454].z])

        # 2. 计算像素级指标
        # 宽度：左右鬓角的 x 像素差
        width_px = abs(p_right_side[0] - p_left_side[0]) * w
        # 深度：鼻尖相对于参考平面的 z 像素深度
        depth_px = abs(p_nose[2]) * w

        # 计算比值
        ratio = depth_px / width_px if width_px != 0 else 0.0
        last_metrics = (width_px, depth_px, ratio)

        # 显示数据
        cv2.putText(image, f"Width_px: {int(width_px)}", (30, 50), 1, 1, (0, 255, 0), 1)
        cv2.putText(image, f"Depth_px: {int(depth_px)}", (30, 80), 1, 1, (0, 255, 0), 1)
        cv2.putText(image, f"Ratio (D/W): {ratio:.4f}", (30, 110), 1, 1, (255, 255, 0), 2)

      key = cv2.waitKey(1) & 0xFF
      if key == ord("s"):
        if last_metrics is None:
          print("\n--- 未检测到人脸，无法采集数据 ---")
        else:
          width_px, depth_px, ratio = last_metrics
          print("\n--- 数据采集成功 ---")
          print(f"像素宽度: {width_px:.2f}, 像素深度: {depth_px:.2f}, 比值: {ratio:.4f}")
      elif key == ord("q"):
        break

      cv2.imshow("Z-Scale Experiment", image)
  finally:
    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
  main()

