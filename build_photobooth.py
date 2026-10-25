"""Đóng gói worker Photobooth cùng runtime DirectML và phương án CPU."""

import importlib.metadata
from pathlib import Path
import shutil
import sys

from PyInstaller.__main__ import run as pyinstaller_run


ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
# Model FP32 làm nguồn chuẩn; model FP16 được chuyển đổi và lưu trong project.
MODEL = Path.home() / ".u2net" / "birefnet-general.onnx"
MODEL_FP16 = ROOT / ".build_models" / "birefnet-general-fp16.onnx"
DIST_ROOT = ROOT / "dist"


def check_environment():
    """Dừng sớm nếu Python, thư viện build hoặc model không đúng yêu cầu."""
    if sys.version_info[:2] != (3, 12) or sys.maxsize <= 2**32:
        raise RuntimeError("Build can Python 3.12 x64.")
    required = (
        "onnxruntime-directml", "pillow", "numpy", "onnx", "pyinstaller"
    )
    missing = []
    for package in required:
        try:
            importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
    if missing:
        raise RuntimeError("Thieu thu vien Python 3.12 build: " + ", ".join(missing))
    if importlib.metadata.version("onnxruntime-directml") != "1.24.4":
        raise RuntimeError("Build can onnxruntime-directml 1.24.4.")
    if not MODEL.is_file():
        raise FileNotFoundError(f"Khong tim thay model: {MODEL}")


def ensure_fp16_model():
    """Tạo bản FP16 một lần để giảm dung lượng model nạp trên GPU DirectML."""
    if MODEL_FP16.is_file():
        return
    MODEL_FP16.parent.mkdir(parents=True, exist_ok=True)
    print("Chuyen model sang FP16 de giam bo nho GPU...", flush=True)
    from onnxruntime.transformers.float16 import convert_float_to_float16
    import onnx

    fp16_model = convert_float_to_float16(
        str(MODEL), keep_io_types=True, disable_shape_infer=True
    )
    onnx.save(fp16_model, str(MODEL_FP16))


def build_package(name, entry_script, readme):
    """Dùng PyInstaller tạo thư mục portable và đặt model/tài liệu đi kèm."""
    output_dir = DIST_ROOT / name
    work_dir = ROOT / "build" / name
    work_dir.mkdir(parents=True, exist_ok=True)
    args = [
        "--noconfirm",
        "--clean",
        "--onedir",
        "--console",
        f"--specpath={work_dir}",
        f"--name={name}",
        f"--distpath={DIST_ROOT}",
        f"--workpath={work_dir}",
        f"--add-data={MODEL};models",
        f"--add-data={MODEL_FP16};models",
    ]
    # Model được nhúng trong models để máy đích không phải tải model lần đầu.
    args.append(str(SRC_ROOT / entry_script))
    print(f"\nDang dong goi {name}...", flush=True)
    pyinstaller_run(args)

    # Hai thư mục rỗng này là mặc định khi chỉ mang riêng gói dist sang máy khác.
    (output_dir / "Input").mkdir(parents=True, exist_ok=True)
    (output_dir / "Output").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / readme, output_dir / Path(readme).name)
    print(f"Da tao: {output_dir}", flush=True)


def build_root_launcher():
    """Tạo EXE nhỏ ở thư mục gốc để chuyển lệnh tới gói portable trong dist."""
    launcher = ROOT / "photobooth_launcher.py"
    work_dir = ROOT / "build" / "PhotoboothLauncher"
    work_dir.mkdir(parents=True, exist_ok=True)
    args = [
        "--noconfirm",
        "--clean",
        "--onefile",
        "--console",
        f"--specpath={work_dir}",
        "--name=Photobooth",
        f"--distpath={ROOT}",
        f"--workpath={work_dir}",
        str(launcher),
    ]
    print("\nDang tao Photobooth.exe tai thu muc goc...", flush=True)
    pyinstaller_run(args)


def main():
    """Kiểm tra đầu vào, đóng gói worker rồi tạo launcher ở thư mục gốc."""
    check_environment()
    ensure_fp16_model()
    print(f"Python build: {sys.executable}")
    print(f"Model birefnet-general FP32: {MODEL} ({MODEL.stat().st_size / (1024**3):.2f} GiB)")
    print(f"Model FP16 for DirectML GPUs: {MODEL_FP16} ({MODEL_FP16.stat().st_size / (1024**3):.2f} GiB)")
    print("Backend: ONNX Runtime DirectML; CPU fallback.")

    build_package(
        "Photobooth", "photobooth_worker.py", "README.md",
    )
    build_root_launcher()
    print("\nBUILD HOAN TAT: Photobooth.exe va dist\\Photobooth")


if __name__ == "__main__":
    # Giữ thông báo lỗi build dễ đọc và trả mã lỗi khác 0 cho script gọi.
    try:
        main()
    except Exception as error:
        print("BUILD THAT BAI:", error)
        raise SystemExit(1)
