# Photobooth BG Remover

Photobooth BG Remover dùng BiRefNet General để tách nền ảnh. Repository chỉ chứa source và công cụ setup/build; môi trường Python, model, cache, ảnh đầu vào/đầu ra và gói `dist` được tạo riêng trên từng máy.

## Chạy source sau khi clone

1. Clone repository và giữ nguyên cấu trúc thư mục.
2. Trên Windows x64, mở CMD hoặc PowerShell tại thư mục vừa clone và chạy:

   ```powershell
   .\INSTALL.bat run
   ```

3. Lần đầu, `INSTALL.bat` tạo môi trường Python 3.12 riêng và cài thư viện. Nếu cần, script cài Python qua `winget`. Nếu không có `winget` cài Python thủ công bằng link:
   https://www.python.org/ftp/python/3.12.9/python-3.12.9-amd64.exe
Chương trình cần Internet để cài đặt và tải model BiRefNet (khoảng 1 GB) ở lần chạy đầu. 
4. Giữ cửa sổ terminal mở. Thêm ảnh mới vào `Input`; chương trình chỉ xử lý ảnh được thêm sau khi khởi động. Ảnh PNG đã tách nền và trim viền trong suốt sẽ được lưu vào `Output`.
5. Worker chờ ảnh ổn định **1 giây**, xử lý ảnh tuần tự theo lô và giữ model mở trong phiên. Nhấn `Ctrl+C` để dừng.

Có thể chọn thư mục khác:

```powershell
.\INSTALL.bat run "D:\Anh\Input" --output "D:\Anh\Output"
```

## Tạo gói portable

Sau khi cài môi trường và tải model bằng lần chạy source đầu tiên, tạo gói bằng:

```powershell
.\INSTALL.bat build
```

Gói hoàn chỉnh được tạo trong `dist\Photobooth`. Để mang gói sang máy khác, sao chép toàn bộ thư mục đó; Python không cần cài trên máy đích. `Photobooth.exe` ở thư mục gốc được tạo như launcher tiện ích khi build.

## Những gì được lưu trong repository

- `src\`: worker Photobooth và xử lý tách nền.
- `INSTALL.bat`: setup môi trường, chạy source hoặc build gói.
- `build_photobooth.py`, `photobooth_launcher.py`: tạo gói portable và launcher.
- `.gitignore`: loại môi trường, model cache, ảnh, build và `dist` khỏi Git.
