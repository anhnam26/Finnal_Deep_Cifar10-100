# Chạy notebook và báo cáo ngày 16/10/2026

Mở [CompactMix_CIFAR10_CIFAR100_BAO_CAO.ipynb](CompactMix_CIFAR10_CIFAR100_BAO_CAO.ipynb) trong VS Code. Notebook có đầy đủ code Conv, 13 block, forward, dữ liệu, loss, optimizer, train, validation, checkpoint, resume và final evaluation trong các cell; không import code từ `src/`.

## Bắt đầu

1. Chọn đúng kernel Python đã dùng: `C:\Users\anhna\AppData\Local\Microsoft\WindowsApps\python3.13.exe`. Không tạo môi trường mới. Cell import kiểm tra executable, base environment, CUDA và vị trí thư viện.
2. Chạy từ cell đầu theo thứ tự. Có thể dùng **Run All** với cấu hình mặc định: chỉ train demo ngắn; các cờ train nhiều LR, final 50k và đo lại test đều mặc định tắt.
3. Ở mục **3. Cấu hình chạy và recipe**, đổi `RUN_NAME` thành tên mới, ví dụ `bao_cao_truc_tiep`, để bắt đầu minh họa từ epoch 1. Run mặc định `bao_cao_20261016` đã có checkpoint của lượt kiểm tra; giữ tên đó thì dùng lại trạng thái đã lưu.
4. Giữ `MODE="demo"` để trình bày nhanh. Notebook dùng 512 ảnh train và 256 validation cho mỗi dataset. `DEMO_TOTAL_EPOCHS=2`, mỗi lần chạy cell train thêm tối đa một epoch. Đây là điểm demo subset, không phải accuracy full-data trong PDF.
5. Muốn bỏ qua train mới và chỉ trình bày model/kết quả cũ, đặt `MODE="report_only"`. Muốn train selection đầy đủ thì đặt `MODE="full"`; lịch mặc định 600 epoch, cần thời gian dài.

Không cần truy cập Colab để chạy. [Link PyTorch bạn gửi](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/cifar10_tutorial.ipynb) được dùng làm mẫu bố cục cell. Notebook hiện chạy bằng Python Windows đang có theo chính sách môi trường của bạn.

## Thứ tự thuyết trình

**Bám sát Final exam.docx:** ưu tiên thiết kế L → kiểm tra ≤6M/<1G → learning settings → kết quả CIFAR-10/100 → bảng SGD/Adam và nhiều LR → kết luận. Phần thực nghiệm bắt buộc đang thiếu là Adam và các LR khác. 600 epoch, retrain 50k, multi-seed, ablation không phải yêu cầu bắt buộc của văn bản đề; chúng là lựa chọn/giao thức mở rộng của dự án.

| Mục trong notebook | Thao tác và nội dung nói |
|---|---|
| 1–3 | Đề bài ≤6M/<1G; kết quả cũ và giới hạn; Python/GPU; recipe |
| 4 | Mở train dataset, tách 45k/5k, normalization, xem ảnh |
| 5 | Mở code Block01, giải thích chunk → depthwise hai kernel → concat → grouped Conv → SE → projection → ADD; các block còn lại khai báo riêng |
| 6 | Chạy forward/backward; xem 261 leaf layers, shape từng lớp, tham số từng block, MACs/FLOPs và hai sơ đồ |
| 7 | Xem nhãn mềm, loss và lịch LR; giải thích SGD/Adam, EMA, AMP, clip gradient |
| 8 | Chạy cell **Chạy train mới / chạy tiếp**; đợi một dòng log/epoch; xem best/last và đường học demo |
| 8, chạy lại cell train | Minh họa resume: tiếp tục epoch kế tiếp bằng model/EMA/optimizer/scaler/RNG đã lưu |
| 9 | Giao thức mở rộng của dự án: freeze và retrain 50k; không trình bày đây là yêu cầu bắt buộc trong đề; để RUN_FINAL=False khi báo cáo ngắn |
| 10 | Đọc evidence của 600 epoch, kiểm tra hash/split/confusion, hiển thị kết quả và cấu hình đã dùng |
| 11 | Nạp best cũ theo ánh xạ tên layer chính xác, dự đoán 8 ảnh validation; tùy chọn đo lại test mặc định tắt |
| 12 | Nêu rõ mới có full-data SGD LR=0,1; những LR/Adam còn thiếu; không gọi demo là kết quả so sánh full-data |
| 13–14 | Đo tốc độ forward tại máy hiện tại; câu hỏi vấn đáp; xuất receipt và đường dẫn hình/CSV |
| 15 | Đọc lại Final exam.docx, xuất PDF/Markdown/CSV đối chiếu từng yêu cầu; ghi rõ các ô SGD/Adam/LR chưa có kết quả |

PDF bám sát đề: [BAO_CAO_THEO_DE_FINAL_EXAM.pdf](BAO_CAO_THEO_DE_FINAL_EXAM.pdf). Bản [Markdown](BAO_CAO_THEO_DE_FINAL_EXAM.md) để đọc/sửa nội dung, [bảng SGD/Adam/LR](BANG_SO_SANH_SGD_ADAM_LR.csv) để kiểm tra số liệu. PDF snapshot cũ được giữ nguyên.

## Các phép đo đã có

- Tham số học được, kích thước weight FP32, MACs, FLOPs và GFLOPs cho cả hai head.
- Shape input/output và tham số của từng layer; số tham số từng block; kiểm tra forward/backward.
- Train loss, mixed accuracy, raw/EMA validation loss/accuracy, LR, epoch best, thời gian và AMP skips.
- Test top-1/top-5, số đúng/sai, test loss, macro-F1, precision/recall/F1/support từng lớp, confusion matrix 10×10 và 100×100.
- Lịch sử 600 epoch, cấu hình/seed/split/runtime/source/checkpoint SHA của thí nghiệm cũ.
- Latency median/p95 và tốc độ forward batch 1, FP32 đo mới. Dùng tensor giả chỉ để đo thời gian, không đo accuracy; không suy ra tốc độ từ FLOPs.

Số liệu đã được đối chiếu với PDF:

| Dataset | Parameters | FLOPs/ảnh | Best val 45k/5k | Test best 45k |
|---|---:|---:|---:|---:|
| CIFAR-10 | 5.248.172 | 842.406.944 | 97,56% | 97,18% |
| CIFAR-100 | 5.271.302 | 842.453.114 | 83,36% | 82,98% |

Các điểm test này là bằng chứng đã lưu của checkpoint selection train 45k. Chưa có final-test 50k trong snapshot PDF. Notebook mặc định kiểm chứng và hiển thị điểm cũ, không tự đo lại test.

Đề yêu cầu train/test cả hai dataset với SGD, Adam và các LR khác nhau, nhưng không ấn định số epoch hay số LR cụ thể. Bảng kế hoạch hiện có 3 LR mỗi optimizer trên cả hai dataset. Những hàng chưa chạy phải ghi **Chưa có kết quả**. Chỉ có code hỗ trợ SGD/Adam hoặc demo ngắn chưa đủ để nói đã hoàn tất thực nghiệm này. Kiến trúc cũng cần được đối chiếu với mạng đã công bố và các nhóm khác; không khẳng định độc nhất khi chưa có bằng chứng.

## Lưu và chạy tiếp

Output mới: `experiments/notebook_presentation/` ở thư mục gốc dự án.

- Mỗi run giữ đúng `last.pt` và `best.pt`; log/metrics chỉ cập nhật sau epoch hoàn chỉnh.
- Muốn dừng, dùng **Interrupt Kernel**. Nếu đang giữa epoch, epoch dở không được lưu; lần tiếp theo phát lại epoch đó từ last.
- Muốn resume, giữ nguyên RUN_NAME, tổng epochs, recipe và các cell định nghĩa; Restart Kernel rồi chạy lại từ đầu vẫn được.
- Nếu sửa Conv/forward hoặc recipe, dùng RUN_NAME mới. Source fingerprint lấy nội dung cell định nghĩa thực sự đã chạy, không lấy execution_count hay output; sửa chưa Save cũng được nhận biết.
- Trong notebook dùng `workers=0` để DataLoader chạy ổn định trên Windows/Jupyter. Recipe báo cáo cũ dùng workers=2 và được hiển thị nguyên trạng từ evidence.
- Không đổi `DEMO_TOTAL_EPOCHS` từ 2 xuống 1 chỉ để chạy ngắn: `DEMO_STOP_AFTER=1` đã đảm nhiệm dừng sau một epoch mới mà giữ lịch LR/resume nhất quán.

Hình, CSV layer/profile/per-class/confusion, latency và `presentation_receipt.json` cũng nằm trong output này. Notebook đã lưu output kiểm tra để bạn mở xem trước; có thể chạy lại từng cell khi báo cáo.

## Những file cần giữ khi chuyển máy/thư mục

Model và trainer đều nằm trong notebook. Phần trình bày kết quả cũ đọc dữ liệu ngoài:

- `report/evidence/cifar10/`, `report/evidence/cifar100/`.
- `report/checkpoints/cifar10/best.pt`, `report/checkpoints/cifar100/best.pt`.
- `datasets/` chứa CIFAR-10/CIFAR-100 đã tải.

Giữ cấu trúc `report/` và `datasets/` như trong dự án. Hai weights `.pt` bị Git ignore nhưng đã có trên máy; sao chép folder để nộp phải mang kèm chúng. Notebook không thay đổi báo cáo PDF, source gốc hay checkpoint train cũ.
