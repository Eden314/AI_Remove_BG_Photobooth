"""Small root-level launcher for the portable Photobooth package."""

from pathlib import Path
import subprocess
import sys


def main() -> int:
    """Mở worker portable nằm trong dist và chuyển tiếp tham số dòng lệnh."""
    # Launcher nằm ở thư mục gốc; toàn bộ runtime/model được giữ trong dist.
    root = Path(sys.executable).resolve().parent
    app = root / "dist" / "Photobooth" / "Photobooth.exe"
    if not app.is_file():
        print("Khong tim thay goi Photobooth:", app)
        print("Hay giu nguyen thu muc dist\\Photobooth ben canh file nay.")
        input("Nhan Enter de dong...")
        return 1
    try:
        arguments = sys.argv[1:]
        # Không truyền đường dẫn thì chọn Input/Output cạnh launcher ở thư mục gốc.
        has_input = bool(arguments) and not arguments[0].startswith("-")
        has_output = "--output" in arguments
        if not has_input:
            arguments = [str(root / "Input"), *arguments]
        if not has_output:
            arguments.extend(("--output", str(root / "Output")))
        # Chờ worker kết thúc và trả nguyên mã thoát về cho terminal gọi launcher.
        return subprocess.run([str(app), *arguments], cwd=app.parent).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
