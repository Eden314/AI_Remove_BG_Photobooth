"""Tách nền ảnh bằng BiRefNet và xuất kết quả PNG có nền trong suốt.

File này chỉ chạy worker Photobooth: bỏ qua ảnh đã có khi khởi động, giữ model
đang mở và xử lý tuần tự các lô ảnh mới được thêm vào thư mục Input.
"""

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

# Cấu hình đường dẫn/model và các giới hạn của vòng chờ file đầu vào.
SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_MODEL = "birefnet-general"
FP16_MODEL = "birefnet-general-fp16.onnx"
MODEL_URL = (
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
    "BiRefNet-general-epoch_244.onnx"
)
MODEL_MD5 = "7a35a0141cbbc80de11d9c9a28f52697"
INPUT_QUIET_SECONDS = 1.0
MAX_INPUT_READY_RETRIES = 20


class InputNotReadyError(Exception):
    """Ảnh chưa ghi xong hoặc chưa thể giải mã; chế độ watch có thể thử lại."""


def file_identity(path: Path):
    """Tạo dấu nhận dạng file để phát hiện file bị sửa trong khi chờ/đọc."""
    stat = path.stat()
    return (str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def load_stable_snapshot(path: Path, expected_identity, Image, ImageOps):
    """Đọc bản sao ổn định của ảnh, tránh xử lý file nguồn đang được ghi."""
    snapshot_path = None
    try:
        # So sánh trước và sau khi sao chép để không lấy ảnh ở trạng thái dở dang.
        if file_identity(path) != expected_identity:
            raise InputNotReadyError("tep anh van dang thay doi")
        with tempfile.NamedTemporaryFile(
            prefix="photobooth_", suffix=path.suffix, delete=False
        ) as snapshot:
            snapshot_path = Path(snapshot.name)
            with path.open("rb") as source:
                shutil.copyfileobj(source, snapshot, length=4 * 1024 * 1024)
        if file_identity(path) != expected_identity:
            raise InputNotReadyError("tep anh thay doi trong luc tao ban doc")
        # verify() kiểm tra cấu trúc ảnh; lần mở tiếp theo giải mã pixel thực tế.
        with Image.open(snapshot_path) as check:
            check.verify()
        with Image.open(snapshot_path) as opened:
            opened.load()
            # Tôn trọng hướng chụp trong EXIF, rồi chuẩn hóa ảnh cho model.
            image = ImageOps.exif_transpose(opened).convert("RGB")
        return image
    except InputNotReadyError:
        raise
    except Exception as error:
        raise InputNotReadyError(f"chua doc duoc anh day du: {error}") from error
    finally:
        if snapshot_path is not None:
            snapshot_path.unlink(missing_ok=True)


def parse_args():
    """Đọc thư mục cần theo dõi và các tùy chọn xử lý Photobooth."""
    parser = argparse.ArgumentParser(
        description="Photobooth: cho anh moi va tach nen lien tuc bang BiRefNet."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="Thu muc can theo doi (mac dinh: Input trong thu muc Photobooth).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Thu muc luu PNG (mac dinh: Output trong thu muc Photobooth).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Ep buoc xu ly bang CPU, khong thu dung GPU DirectML.",
    )
    parser.add_argument(
        "--keep-mask",
        type=Path,
        help=(
            "Anh mask den-trang cung kich thuoc: vung trang se duoc giu lai "
            "(dung de sua vat bi AI xoa nham)."
        ),
    )
    return parser.parse_args()


def application_root() -> Path:
    """Tìm thư mục gốc phù hợp khi chạy dạng EXE hoặc chạy từ mã nguồn."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def model_file(model_home: Path) -> Path:
    """Tìm model ONNX; nếu thiếu thì tải, xác minh checksum rồi lưu."""
    model_path = model_home / f"{DEFAULT_MODEL}.onnx"
    if model_path.is_file():
        return model_path

    model_home.mkdir(parents=True, exist_ok=True)
    partial_path = model_path.with_suffix(".onnx.part")
    print("Dang tai model lan dau (gan 1 GB)...", flush=True)

    def report_progress(block_count, block_size, total_size):
        """Hiển thị phần trăm tải model trong terminal."""
        if total_size > 0:
            percent = min(100, int(block_count * block_size * 100 / total_size))
            print(f"\rTai model: {percent}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(MODEL_URL, partial_path, report_progress)
        # Không dùng model tải dở hoặc sai nội dung.
        digest = hashlib.md5()
        with partial_path.open("rb") as downloaded:
            for chunk in iter(lambda: downloaded.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != MODEL_MD5:
            raise ValueError("Checksum cua model khong dung; tep tai xuong bi loi.")
        partial_path.replace(model_path)
        print("\nTai model xong.")
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise
    return model_path


def predict_mask(session, image, np, Image):
    """Chuẩn hóa ảnh, chạy BiRefNet và trả mask xám cùng kích thước ảnh gốc."""
    # Model nhận ảnh 1024x1024; chuẩn hóa pixel theo mean/std dùng khi huấn luyện.
    resized = image.convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.float64)
    pixels /= max(float(pixels.max()), 1e-6)
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float64)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float64)
    pixels = (pixels - mean) / std
    tensor = np.expand_dims(pixels.transpose((2, 0, 1)), axis=0).astype(np.float32)

    # Đầu ra model là logits; sigmoid đổi sang xác suất rồi chuẩn hóa thành mask 0-255.
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: tensor})[0][:, 0, :, :]
    prediction = 1.0 / (1.0 + np.exp(-output))
    low = float(prediction.min())
    high = float(prediction.max())
    if high > low:
        prediction = (prediction - low) / (high - low)
    else:
        prediction.fill(0)
    mask = Image.fromarray(
        (np.squeeze(prediction) * 255).astype("uint8"), mode="L"
    )
    return mask.resize(image.size, Image.Resampling.LANCZOS)


def create_session(ort, model_path, providers, use_directml=False):
    """Tạo phiên ONNX Runtime với provider GPU/CPU và tùy chọn DirectML."""
    options = None
    if use_directml:
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # Tắt graph fusion để tránh một số GPU/driver DirectML lỗi với graph lớn.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        str(model_path), sess_options=options, providers=providers
    )


def is_gpu_memory_error(error: Exception) -> bool:
    """Nhận diện lỗi DirectML báo thiếu VRAM/bộ nhớ tài nguyên GPU."""
    message = str(error).casefold()
    return any(
        marker in message
        for marker in (
            "8007000e",
            "not enough memory",
            "out of memory",
            "memory resources are not available",
        )
    )


def wait_for_new_images(
    folder: Path,
    processed: set,
    stable: dict,
    excluded: Path = None,
    ignored_paths: set = None,
):
    """Chờ ảnh mới ổn định rồi trả về một snapshot để khóa thành lô xử lý.

    File được nhận diện bằng đường dẫn, kích thước và thời điểm sửa cuối. Chỉ
    khi bộ nhận diện giữ nguyên trong INPUT_QUIET_SECONDS mới được đưa vào lô.
    Hàm này không chạy trong lúc model xử lý lô, vì vậy ảnh mới đến sau snapshot
    sẽ chỉ được quét khi vòng lặp ngoài bắt đầu lô kế tiếp.
    """
    while True:
        ready = []
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED:
                continue
            if excluded and path.resolve() == excluded:
                continue
            path_key = str(path.resolve())
            if ignored_paths and path_key in ignored_paths:
                continue
            try:
                identity = file_identity(path)
            except OSError:
                continue
            if identity in processed:
                continue
            now = time.monotonic()
            previous = stable.get(path_key)
            # Bắt đầu tính thời gian ổn định lại nếu kích thước/ngày sửa đã đổi.
            if previous is None or previous[0] != identity:
                stable[path_key] = (identity, now)
                continue
            if now - previous[1] >= INPUT_QUIET_SECONDS:
                ready.append((path, identity))
        if ready:
            return ready
        time.sleep(0.5)


def main():
    """Chạy Photobooth liên tục, bỏ qua ảnh cũ và xử lý các lô ảnh mới."""
    if not getattr(sys, "frozen", False):
        print(f"Source code: {Path(__file__).resolve()}")
        print(f"Python runtime: {sys.executable}")
    args = parse_args()
    app_root = application_root()
    input_path = (args.path or (app_root / "Input")).expanduser().resolve()
    output_dir = (args.output or (app_root / "Output")).expanduser().resolve()
    # Tạo Input mặc định, nhưng không âm thầm tạo đường dẫn người dùng chỉ định nếu nó thiếu.
    if args.path is None:
        input_path.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        print("Khong tim thay tep/thu muc:", input_path)
        return 1
    if not input_path.is_dir():
        print("Photobooth chi theo doi thu muc; hay truyen duong dan folder Input.")
        return 1
    # Chụp danh sách file ngay lúc bật worker; ảnh cũ không được đưa vào hàng đợi.
    ignored_paths = {
        str(path.resolve())
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED
    }

    # Nạp thư viện nặng sau khi đã kiểm tra đường dẫn đầu vào.
    print("Dang nap thu vien AI...", flush=True)
    bundled_model_dir = None
    if getattr(sys, "frozen", False):
        # PyInstaller giải nén tài nguyên vào _MEIPASS; model portable nằm trong models.
        bundled_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        bundled_model_dir = bundled_root / "models"
        os.environ["U2NET_HOME"] = str(bundled_model_dir)
    try:
        from PIL import Image, ImageChops, ImageOps
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("Thieu thu vien. Hay chay INSTALL.bat de cai moi truong source.")
        return 1
    print("Thu vien da san sang.", flush=True)
    if bundled_model_dir and not (
        bundled_model_dir / f"{DEFAULT_MODEL}.onnx"
    ).is_file():
        print("Goi portable bi thieu model birefnet-general.onnx:", bundled_model_dir)
        return 1

    # Mask tùy chọn được loại khỏi hàng đợi để không bị coi là ảnh chụp mới.
    keep_mask_path = args.keep_mask.expanduser().resolve() if args.keep_mask else None
    if keep_mask_path and not keep_mask_path.is_file():
        print("Khong tim thay mask:", keep_mask_path)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Thu muc dau vao: {input_path}")
    print(f"Thu muc ket qua: {output_dir}")
    print(f"Mo hinh: {DEFAULT_MODEL}")
    print("Tinh chinh vien alpha theo kich thuoc anh goc")

    # Xác định các lựa chọn một lần theo thứ tự: DirectML FP32, DirectML FP16,
    # rồi CPU FP32. CPU FP32 tương thích rộng và nhanh hơn trên máy đã đo.
    # Lựa chọn đầu tiên khởi tạo và chạy ảnh thành công sẽ được giữ cho cả phiên.
    available = ort.get_available_providers()
    gpu_provider = "DmlExecutionProvider"
    if not args.cpu and gpu_provider not in available:
        print(
            "CANH BAO: Python hien tai khong co DirectML provider; "
            "hay chay INSTALL.bat run de dung moi truong .photobooth-env."
        )
    has_directml = gpu_provider in available and not args.cpu
    model_home = bundled_model_dir or Path(
        os.environ.get("U2NET_HOME", Path.home() / ".u2net")
    )
    model_path = None
    cpu_model_path = None
    fp16_path = None
    runtime_tiers = []
    active_tier = -1
    selected_runtime_confirmed = False
    # Giá trị này cũng tồn tại nếu lỗi xảy ra trước khi tầng đầu tiên khởi tạo.
    providers = (
        [gpu_provider, "CPUExecutionProvider"]
        if has_directml
        else ["CPUExecutionProvider"]
    )
    session_started = time.perf_counter()
    print("Dang khoi tao mo hinh AI...\n")
    try:
        # FP32 là bước thử đầu tiên; FP16 chỉ được dùng nếu FP32 lỗi.
        cpu_model_path = model_file(model_home)
        fp16_candidates = (
            model_home / FP16_MODEL,
            app_root / ".build_models" / FP16_MODEL,
            app_root / "dist" / "Photobooth" / "_internal" / "models" / FP16_MODEL,
        )
        fp16_path = next(
            (candidate for candidate in fp16_candidates if candidate.is_file()),
            fp16_candidates[0],
        )
        # Thử lần lượt FP32 GPU, FP16 GPU, rồi FP32 CPU.
        if has_directml:
            runtime_tiers.append(("DirectML FP32", cpu_model_path, True))
            if fp16_path.is_file() and fp16_path != cpu_model_path:
                runtime_tiers.append(("DirectML FP16", fp16_path, True))
        runtime_tiers.append(("CPU FP32", cpu_model_path, False))

        # Nếu session không khởi tạo được, thử lần lượt tầng tiếp theo.
        # Không coi khởi tạo thành công là đủ: lỗi suy luận cũng được xử lý ở dưới.
        for tier_index, (tier_name, tier_model, use_gpu) in enumerate(runtime_tiers):
            tier_providers = (
                [gpu_provider, "CPUExecutionProvider"]
                if use_gpu
                else ["CPUExecutionProvider"]
            )
            try:
                session = create_session(
                    ort, tier_model, tier_providers, use_directml=use_gpu
                )
                active_tier = tier_index
                model_path = tier_model
                providers = tier_providers
                print(f"Dang dung thu: {tier_name}.")
                break
            except Exception as tier_error:
                print(f"Khong khoi tao duoc {tier_name}: {tier_error}")
        else:
            raise RuntimeError("Khong khoi tao duoc model tren GPU hoac CPU.")
    except Exception as error:
        if gpu_provider in providers:
            print(f"DirectML khong khoi tao duoc ({error}); thu lai bang CPU.")
            try:
                cpu_model_path = cpu_model_path or model_file(model_home)
                model_path = cpu_model_path
                providers = ["CPUExecutionProvider"]
                runtime_tiers = [("CPU FP32", model_path, False)]
                active_tier = 0
                session = create_session(ort, model_path, providers)
            except Exception as cpu_error:
                print(f"Khong the khoi tao ONNX Runtime tren CPU: {cpu_error}")
                return 1
        else:
            print(f"Khong the khoi tao model {DEFAULT_MODEL}: {error}")
            return 1
    print(f"Model ONNX dang dung: {model_path}")
    active = session.get_providers()
    if gpu_provider in active:
        print("DirectML da khoi tao; dang su dung GPU DirectX 12 neu cac toan tu duoc ho tro.")
    else:
        print("Xu ly bang: CPU (DirectML/GPU khong san sang hoac da tat)")
    print(f"Thoi gian nap model/session: {time.perf_counter() - session_started:.2f} giay.")

    # Photobooth luôn chạy liên tục; ảnh có sẵn đã được loại khỏi hàng đợi.
    if ignored_paths:
        print(
            f"Bo qua {len(ignored_paths)} anh da co luc khoi dong; "
            "chi xu ly anh duoc them sau do."
        )
    print("Che do photobooth: mo hinh se duoc giu san; nhan Ctrl+C de dung.")
    print(
        f"Moi anh phai dung kich thuoc/thoi gian ghi it nhat "
        f"{INPUT_QUIET_SECONDS:.1f} giay va doc duoc truoc khi xu ly."
    )
    print(
        "Worker khoa danh sach tung lo truoc khi xu ly; anh moi them "
        "trong luc dang chay se duoc xep vao lo ke tiep."
    )
    processed = set()
    stable = {}
    input_retries = {}
    batch_number = 0
    totals = {"ok": 0, "fail": 0}

    # Mỗi lượt lấy một lô đã ổn định rồi xử lý xong trước khi quét lô kế tiếp.
    while True:
        try:
            ready = wait_for_new_images(
                input_path,
                processed,
                stable,
                keep_mask_path,
                ignored_paths,
            )
        except KeyboardInterrupt:
            print("\nDa dung worker Photobooth.", flush=True)
            break
        # Chốt snapshot; ảnh mới xuất hiện sau thời điểm này vào lô tiếp theo.
        batch_number += 1
        batch = tuple(ready)
        files = [item[0] for item in batch]
        identities = {item[0]: item[1] for item in batch}
        print(
            f"\n[LO {batch_number}] Da chot {len(files)} anh; "
            "worker xu ly tuan tu. Anh den luc nay se cho lo tiep theo.",
            flush=True,
        )
        ok = fail = 0
        # Xử lý lần lượt từng ảnh để giới hạn bộ nhớ và bảo đảm thứ tự công việc.
        for i, src in enumerate(files, 1):
            started = time.perf_counter()
            dst = output_dir / (src.stem + ".png")
            identity = identities[src]
            retry_input = False
            temporary_output = None
            try:
                # Chỉ suy luận trên snapshot đã xác minh, không đọc trực tiếp file đang đổi.
                image = load_stable_snapshot(
                    src, identity, Image, ImageOps
                )
                # Khi một tầng đã chạy thành công, giữ tầng đó cho ảnh hiện tại và
                # mọi ảnh tiếp theo trong phiên; không tự chuyển ngược lên GPU.
                while True:
                    try:
                        mask = predict_mask(session, image, np, Image)
                        # Ghi nhận tier sau lần suy luận đầu tiên thành công. Từ đây
                        # worker giữ nguyên tier này cho toàn bộ phần còn lại phiên.
                        if not selected_runtime_confirmed:
                            print(
                                f"Da xac nhan {runtime_tiers[active_tier][0]} "
                                "chay duoc; co dinh lua chon nay cho cac anh tiep theo."
                            )
                            selected_runtime_confirmed = True
                        break
                    except Exception as inference_error:
                        next_tier = active_tier + 1
                        if next_tier >= len(runtime_tiers):
                            raise
                        print(
                            f"\n{runtime_tiers[active_tier][0]} gap loi khi suy luan "
                            f"({inference_error}); dang thu {runtime_tiers[next_tier][0]}."
                        )
                        # Có thể lỗi xảy ra ngay khi tạo session tầng kế tiếp; khi
                        # đó tiếp tục lần lượt xuống dưới cho đến tầng dùng được.
                        while next_tier < len(runtime_tiers):
                            tier_name, tier_model, use_gpu = runtime_tiers[next_tier]
                            tier_providers = (
                                [gpu_provider, "CPUExecutionProvider"]
                                if use_gpu
                                else ["CPUExecutionProvider"]
                            )
                            try:
                                session = create_session(
                                    ort, tier_model, tier_providers,
                                    use_directml=use_gpu,
                                )
                                active_tier = next_tier
                                model_path = tier_model
                                providers = tier_providers
                                print(f"Da chuyen sang {tier_name}; thu lai anh nay.")
                                break
                            except Exception as session_error:
                                print(f"Khong khoi tao duoc {tier_name}: {session_error}")
                                next_tier += 1
                        else:
                            raise RuntimeError("Da het cac phuong an GPU va CPU.")
                # Mask AI trở thành kênh alpha: nền trong suốt, chủ thể được giữ lại.
                result = image.convert("RGBA")
                result.putalpha(mask)
                if keep_mask_path:
                    # Mask trắng bổ sung nâng alpha lên, giúp phục hồi vùng AI xóa nhầm.
                    with Image.open(keep_mask_path) as opened_mask:
                        keep_mask = opened_mask.convert("L")
                        if keep_mask.size != image.size:
                            keep_mask = keep_mask.resize(
                                image.size, Image.Resampling.LANCZOS
                            )
                        alpha = ImageChops.lighter(
                            result.getchannel("A"), keep_mask
                        )
                        result.putalpha(alpha)
                # Dùng ngưỡng nhỏ để tìm hộp bao vùng còn nhìn thấy và cắt viền rỗng.
                trim_alpha = result.getchannel("A").point(
                    lambda value: 255 if value > 8 else 0
                )
                trim_box = trim_alpha.getbbox()
                if trim_box:
                    result = result.crop(trim_box)
                # Ghi vào file tạm rồi thay thế đích để tránh lộ PNG ghi chưa xong.
                with tempfile.NamedTemporaryFile(
                    prefix=f".{src.stem}_",
                    suffix=".tmp",
                    dir=output_dir,
                    delete=False,
                ) as temp_file:
                    temporary_output = Path(temp_file.name)
                result.save(temporary_output, "PNG")
                os.replace(temporary_output, dst)
                temporary_output = None
                if i == 1 and gpu_provider in providers:
                    active = session.get_providers()
                    if gpu_provider in active:
                        print("Xac nhan: DirectML provider da duoc chon cho session.")
                    else:
                        print(
                            "CANH BAO: DirectML khong kha dung; ONNX Runtime da chuyen sang CPU. "
                            "Hay cap nhat driver GPU va kiem tra ho tro DirectX 12."
                        )
                input_retries.pop(identity, None)
                ok += 1
                totals["ok"] += 1
                print(
                    f"[{i}/{len(files)}] OK  {src.name} "
                    f"({time.perf_counter() - started:.2f} giay)"
                )
            except InputNotReadyError as error:
                # Worker luôn chờ lại ảnh nếu nguồn vẫn chưa ghi/đọc ổn định.
                retries = input_retries.get(identity, 0) + 1
                input_retries[identity] = retries
                if retries >= MAX_INPUT_READY_RETRIES:
                    fail += 1
                    totals["fail"] += 1
                    input_retries.pop(identity, None)
                    print(
                        f"[{i}/{len(files)}] LOI {src.name}: tep khong hoan tat "
                        f"sau {MAX_INPUT_READY_RETRIES} lan thu ({error})"
                    )
                else:
                    retry_input = True
                    print(
                        f"Anh {src.name} chua ghi xong/doc duoc; se doi on dinh "
                        f"roi thu lai ({retries}/{MAX_INPUT_READY_RETRIES})."
                    )
            except Exception as error:
                fail += 1
                totals["fail"] += 1
                print(f"[{i}/{len(files)}] LOI {src.name}: {error}")
            finally:
                # Dọn file tạm; đánh dấu file đã xong hoặc cho phép chờ ổn định để thử lại.
                if temporary_output is not None:
                    temporary_output.unlink(missing_ok=True)
                if retry_input:
                    stable.pop(str(src.resolve()), None)
                else:
                    processed.add(identity)

        print(
            f"[LO {batch_number}] HOAN TAT: thanh cong {ok}, loi {fail}. "
            f"Da xu ly xong lo nay; worker dang cho lo tiep theo trong: {input_path}",
            flush=True,
        )

    # Báo cáo tổng kết khi người dùng nhấn Ctrl+C để dừng worker.
    print("\n==============================")
    print("HOAN TAT")
    print("Thanh cong:", totals["ok"])
    print("Loi:", totals["fail"])
    print("Ket qua:", output_dir)
    print("Mask AI duoc noi suy theo kich thuoc anh goc; vien trong suot da duoc trim.")
    if keep_mask_path:
        print("Da ap dung mask giu vat:", keep_mask_path)
    print("==============================")
    return 0 if totals["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
