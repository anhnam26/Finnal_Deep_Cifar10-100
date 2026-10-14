# Gói báo cáo CIFAR-10 và CIFAR-100

Mở **Bao_cao_CIFAR10_CIFAR100.pdf** để đọc/nộp. Bản **DOCX** dùng để điền tên, lớp, giảng viên và chỉnh sửa.
Bản **Markdown** cho phép xem trực tiếp trong IDE.
Snapshot số liệu: 14/10/2026 13:56:14 (UTC+7).

| Dataset | Best validation | Test best train 45k | Best epoch |
|---|---:|---:|---:|
| CIFAR-10 | 97,56% | 97,18% | 567 (EMA) |
| CIFAR-100 | 83,36% | 82,98% | 399 (EMA) |

## Thành phần

- figures/: sơ đồ tổng thể, MixBlock, SE, giao thức dữ liệu; learning curves, confusion matrix, recall theo lớp. Có PNG và SVG.
- evidence/cifar10/ và evidence/cifar100/: config, metadata, 600 epoch CSV/log, kết quả selection, split đầy đủ, test JSON, layer shapes/parameters, confusion matrix và per-class CSV.
- evidence/Final exam.docx: đề bài gốc.
- code/: bản sao source train, Conv/forward, data, optimizer, loss, EMA, resume, final evaluation, search và config.
- checkpoints/cifar10/best.pt và checkpoints/cifar100/best.pt: hai trọng số best 45k đã test. Không sao chép last.pt; bản resume vẫn nằm ở checkpoints/ của repository.
- evaluate_best.py: chạy lại đánh giá archived best trên dữ liệu CIFAR local; không tải dataset và không train.
- results_summary.csv/json: bảng chỉ số gọn.
- manifest.json: SHA256 toàn bộ file trong gói và các kiểm tra tính đúng.

## Đọc kết quả đúng phạm vi

Hai số test trên là kết quả best train trên 45k. Retrain 50k chưa có kết quả final test trong snapshot này.
Source selection result.json giữ test_accuracy=null theo giao thức gốc; test đo riêng nằm ở test_evaluation.json.
Đề còn yêu cầu so sánh SGD/Adam và nhiều initial LR: hiện mới có full-data SGD, LR=0,1. Báo cáo nêu rõ phần chưa hoàn thành.
Không coi hỗ trợ --matrix hay Optuna là kết quả đã train; không tuyên bố ablation/multi-seed đã hoàn tất.

## Lệnh dùng môi trường Python duy nhất của người dùng

Từ thư mục gốc D:\DeepLearning\Finnal:

    python scripts/build_submission_report.py
    python report/evaluate_best.py --dataset cifar10 --data-root datasets
    python report/evaluate_best.py --dataset cifar100 --data-root datasets

Alias python chỉ dùng khi đã xác minh là cùng Microsoft Store Python 3.13.
Executable chính thức:
C:\Users\anhna\AppData\Local\Microsoft\WindowsApps\python3.13.exe

Không cần tạo hoặc cài thêm môi trường. Việc build report không train, không thay đổi checkpoint nguồn.
Weights *.pt có trong thư mục cục bộ nhưng bị Git ignore theo chính sách repository; khi gửi folder phải gửi kèm hai file này.
