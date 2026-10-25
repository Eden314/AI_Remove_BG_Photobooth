"""Điểm vào riêng của Photobooth.

File mỏng này là điểm vào được PyInstaller dùng khi đóng gói. Engine
remove_background.py hiện luôn chạy Photobooth và giữ model qua nhiều lô ảnh.
"""

from remove_background import main


if __name__ == "__main__":
    # Gọi cùng worker Photobooth như khi chạy trực tiếp remove_background.py.
    raise SystemExit(main())
