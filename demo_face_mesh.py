import cv2  # OpenCV：用于摄像头读取、颜色空间转换、图像显示与键盘事件监听
import mediapipe as mp  # MediaPipe：在仓库里运行时可能会“误导入”源码目录下的 mediapipe，而不是 pip 安装版
import time  # 用于按时间间隔（例如每 3 秒）做一次输出/采样

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
      "  python -m pip install mediapipe opencv-python\n"
      "并重新打开终端后再运行本脚本。"
  )


mp_solutions = _resolve_mp_solutions()


def main() -> None:  # 程序主入口：封装成函数便于管理资源与异常处理 -None 表示没有返回值（告诉Python解释器这个函数没有返回值）
  # 初始化 MediaPipe Solutions（传统 Solutions API）下的子模块引用，写成局部变量便于后续调用与阅读
  mp_face_mesh = mp_solutions.face_mesh  # 人脸网格（Face Mesh）解决方案：提供 FaceMesh 类与连接拓扑常量等
  mp_drawing = mp_solutions.drawing_utils  # 绘制工具：把 landmarks/连接线画到 OpenCV 图像上
  mp_drawing_styles = mp_solutions.drawing_styles  # 绘制风格：提供默认的线条颜色/粗细等 DrawingSpec

  # 核心配置参数（研究重点）
  # static_image_mode: False 代表处理视频流，会利用上帧信息进行追踪（Tracking）
  # max_num_faces: 探测的最大人脸数
  # refine_landmarks: 是否开启虹膜（Iris）追踪，设为 True 会从 468 点增加到 478 点
  #
  # 创建 FaceMesh 实例（底层会构建/加载一张 MediaPipe graph）：
  # - 对视频流（static_image_mode=False）更稳定：会进行 tracking，减少每帧都从头检测带来的抖动/开销
  # - refine_landmarks=True 会额外输出虹膜相关点（更适合研究眼部与注视方向等）
  face_mesh = mp_face_mesh.FaceMesh(  # FaceMesh “算子对象”：接收 RGB 图像，输出人脸关键点列表
      static_image_mode=False,  # 设为 False：视频流模式，利用上帧信息进行跟踪
      max_num_faces=1,  # 最大检测人脸数：1 表示只关注最显著/最先检测到的一张脸
      refine_landmarks=True,  # 是否细化关键点：True 会启用虹膜细化，landmarks 数量从 468 增加到 478
      min_detection_confidence=0.5,  # 人脸检测最低置信度阈值：越高越严格（漏检可能增加）
      min_tracking_confidence=0.5,  # 追踪最低置信度阈值：越高越严格（丢追踪概率可能增加）
  )  # FaceMesh 初始化完成

  cap = cv2.VideoCapture(0)  # 打开默认摄像头（索引 0）；如有多摄像头可尝试 1/2/...
  last_print_ts = 0.0  # 上次打印时间戳（秒）；用于控制“每 3 秒打印一次”

  try:
    while cap.isOpened():  # 循环读取视频流：cap 打开成功才继续
      success, image = cap.read()  # 读取一帧：success 表示是否成功，image 是 BGR 格式的 numpy 数组
      if not success:  # 如果读取失败（如摄像头被占用/断开），退出循环
        break  # 跳出 while 循环，进入 finally 释放资源

      # 性能优化：将图像标记为不可写以提升处理速度
      image.flags.writeable = False  # 告诉 numpy/后续流程“此数组不需要被原地修改”，可减少不必要的拷贝
      # MediaPipe 要求输入为 RGB，而 OpenCV 默认是 BGR
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 颜色空间转换：BGR -> RGB（FaceMesh 期望 RGB）
      results = face_mesh.process(image)  # 核心推理：对当前帧做检测+关键点回归（视频模式下包含追踪）

      # 还原图像进行绘制
      image.flags.writeable = True  # 恢复可写：后面要在图像上画线/点（OpenCV 会原地修改像素）
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 转回 BGR：OpenCV 的绘制/显示默认按 BGR 解释

      if results.multi_face_landmarks:  # 若检测到人脸：这里会返回一个列表，每个元素是一张脸的 landmark 列表
        for face_landmarks in results.multi_face_landmarks:  # 遍历每张检测到的人脸（本例 max_num_faces=1 通常只有一个）
          # --- 研究切入点：观察 face_landmarks 对象 ---
          # 这里的 landmark[0] 是鼻尖位置
          now_ts = time.time()  # 当前时间戳（秒）
          if now_ts - last_print_ts >= 3.0:  # 距离上次打印超过 3 秒才输出一次
            # print(face_landmarks.landmark[0])  # 打印鼻尖 landmark（NormalizedLandmark：含 x/y/z 等）
            # print(face_landmarks.landmark[234])  # 打印鬓角 landmark（NormalizedLandmark：含 x/y/z 等）
            print(face_landmarks.landmark[468])  # 打印左眼球 虹膜landmark（NormalizedLandmark：含 x/y/z 等）
            # print(face_landmarks.landmark[159])  # 打印左眼球上边缘landmark（NormalizedLandmark：含 x/y/z 等）
            # print(face_landmarks.landmark[145])  # 打印左眼球下边缘landmark（NormalizedLandmark：含 x/y/z 等）
            print(len(face_landmarks.landmark))  # 打印 landmark 列表长度（通常为 468 或 478）
            last_print_ts = now_ts  # 更新“上次打印时间”
          # face_landmarks 的类型通常是 NormalizedLandmarkList：
          # - face_landmarks.landmark 是长度为 468/478 的列表
          # - 每个 landmark 含 x/y/z（一般为归一化坐标，z 是相对深度）

          # 绘制网格
          mp_drawing.draw_landmarks(  # 使用 MediaPipe 自带绘制工具把关键点与连接线画到 image 上（原地修改）
              image=image,  # OpenCV 的 BGR 图像（numpy.ndarray），画完后直接用于显示
              landmark_list=face_landmarks,  # 单张脸的关键点列表（NormalizedLandmarkList）
              connections=mp_face_mesh.FACEMESH_TESSELATION,  # 连接拓扑：面部三角网格（tesselation）
              landmark_drawing_spec=None,  # 不画关键点“点”，只画连线（网格）——视觉上更清晰
              connection_drawing_spec=(  # 连线绘制风格：颜色/粗细等
                  mp_drawing_styles.get_default_face_mesh_tesselation_style()  # 默认网格风格（通常是细灰线）
              ),
          )  # 一张脸的网格绘制完成

      cv2.imshow("MediaPipe Face Mesh Study", cv2.flip(image, 1))  # 镜像显示：更符合“自拍/镜子”直觉（左右翻转）
      if cv2.waitKey(5) & 0xFF == 27:  # 等待 5ms 并读取按键；27 是 ESC 键，按下则退出
        break  # 跳出 while 循环，进入 finally 做清理
  finally:
    cap.release()  # 释放摄像头句柄：避免摄像头被一直占用导致其他程序无法打开
    face_mesh.close()  # 关闭 FaceMesh：释放内部 graph/资源（重要，防止进程退出前资源泄露）
    cv2.destroyAllWindows()  # 关闭 OpenCV 创建的窗口：防止窗口残留/卡住


if __name__ == "__main__":  # 仅当脚本被直接运行（而非被 import）时执行下面逻辑
  main()  # 调用主函数启动摄像头 Face Mesh Demo

