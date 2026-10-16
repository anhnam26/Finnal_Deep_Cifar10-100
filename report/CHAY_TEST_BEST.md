# Nạp trọng số và chạy test CIFAR-10/CIFAR-100

Chạy từ thư mục gốc dự án bằng đúng Python đang dùng:

```powershell
& 'C:\Users\anhna\AppData\Local\Microsoft\WindowsApps\python3.13.exe' -X utf8 report/evaluate_best.py
```

Mặc định chạy lần lượt cả hai checkpoint đã lưu:

| Dataset | File trọng số |
|---|---|
| CIFAR-10 | `report/checkpoints/cifar10/best.pt` |
| CIFAR-100 | `report/checkpoints/cifar100/best.pt` |

Đã chạy kiểm tra lại bằng script này:

| Dataset | Epoch / weights | Test accuracy | Đúng / tổng |
|---|---|---:|---:|
| CIFAR-10 | 567 / EMA | 97,18% | 9.718 / 10.000 |
| CIFAR-100 | 399 / EMA | 82,98% | 8.298 / 10.000 |

Code dựng model theo config checkpoint, nạp `checkpoint["model"]` bằng `strict=True`, dùng `model.eval()` và không tính gradient. Mỗi dataset đánh giá đủ **10.000 ảnh test**, chỉ ToTensor/Normalize, không augmentation ngẫu nhiên. Tự dùng CUDA nếu có; thêm `--device cpu` để chạy CPU.

Console in dataset, epoch, loại weights, accuracy, đúng/tổng, top-5, loss, macro-F1 và đường dẫn checkpoint. File kết quả ở `experiments/test_best/`:

- `cifar10_test.json`, `cifar100_test.json`: metrics, confusion matrix, SHA256 checkpoint và thông tin lần đánh giá.
- `test_summary.csv`: bảng tóm tắt hai dataset.

Chỉ chạy CIFAR-10 hoặc thay checkpoint:

```powershell
& 'C:\Users\anhna\AppData\Local\Microsoft\WindowsApps\python3.13.exe' -X utf8 report/evaluate_best.py --dataset cifar10
& 'C:\Users\anhna\AppData\Local\Microsoft\WindowsApps\python3.13.exe' -X utf8 report/evaluate_best.py --dataset cifar100 --checkpoint 'duong_dan/best.pt'
```

Đổi cả hai checkpoint trong một lệnh: dùng `--checkpoint-cifar10 'path10/best.pt' --checkpoint-cifar100 'path100/best.pt'`. Checkpoint từ trainer gốc và `compactmix_manual_train.py` đều được hỗ trợ; code model phải khớp weights, nếu không sẽ báo lỗi thay vì bỏ qua tensor sai.

Hai checkpoint mặc định là best EMA của selection train45k/val5k đã được test trước đây; không gọi chúng là retrain50k. Script không train hay chọn model từ test, không thay checkpoint/báo cáo cũ. Checkpoint smoke/tập con của lượt so sánh nhanh bị từ chối; dùng validation để so sánh các recipe mới. Mỗi lần chạy lệnh trên sẽ đánh giá test lại, không chỉ đọc điểm lưu sẵn.
