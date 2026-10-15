# Bám sát Final exam.docx

Nguồn yêu cầu là `Final exam.docx` ở thư mục gốc. PDF mới [BAO_CAO_THEO_DE_FINAL_EXAM.pdf](BAO_CAO_THEO_DE_FINAL_EXAM.pdf) trình bày theo bảng dưới, giữ nguyên kết quả đã đo và ghi rõ phần thực nghiệm còn thiếu.

| Yêu cầu đúng theo đề | Báo cáo cần có | Tình trạng hiện tại |
|---|---|---|
| Thiết kế model nhẹ L | Sơ đồ, Conv/forward, mô tả cấu trúc và lý do lựa chọn | CompactMix-L, 3 stage, 13 block tường minh |
| Learnable parameters không quá 6M | Công thức và kết quả đếm trực tiếp model | C10 5.248.172; C100 5.271.302 |
| FLOPs dưới 1G | Input size, quy ước MAC/FLOP, kết quả | C10 842.406.944; C100 842.453.114 FLOPs/ảnh |
| Train/test cả CIFAR-10 và CIFAR-100 | Cấu hình, số ảnh, lịch sử train, kết quả test đúng checkpoint | Đã có hai selection 45k/5k, SGD LR=0,1, test đủ 10k ảnh mỗi tập |
| Các initial learning rate khác nhau cùng SGD và Adam | Bảng dataset × optimizer × LR, số epoch, validation/test và phân tích | **CHƯA HOÀN THÀNH**; mới có full-data SGD LR=0,1 |
| Learning settings chi tiết | Momentum, LR, epochs; thêm batch, scheduler, decay, loss, augment, seed/runtime để tái lập | Đã ghi settings của run đã chạy; settings kế hoạch tách riêng |
| Tất cả đưa vào một PDF | Nội dung đủ các mục trên, số liệu và kết luận | Đã xuất PDF mới theo đề; chưa vì vậy mà đủ phần thực nghiệm còn thiếu |
| Kiến trúc không trùng nhóm khác, không hoàn toàn giống mạng trước (trừ model giữa kỳ của nhóm) | Đối chiếu sơ đồ/thành phần với nguồn tham khảo và nhóm khác | Có mô tả thiết kế; **chưa có bằng chứng xác nhận độc nhất** |
| Vấn đáp, nộp đúng hạn theo link giảng viên | Chuẩn bị giải thích và nộp file | Sinh viên thực hiện |

## Thứ tự nên dùng trong báo cáo

1. Mục tiêu và ràng buộc ≤6M/<1G từ đề.
2. Thiết kế model L, kích thước tensor và tính khác biệt kiến trúc.
3. Kiểm chứng tham số/MACs/FLOPs với input `[1,3,32,32]`.
4. Dữ liệu và learning settings chi tiết.
5. Kết quả CIFAR-10 và CIFAR-100; chỉ rõ train 45k, validation 5k, test 10k.
6. Bảng so sánh SGD/Adam và nhiều LR; ô thiếu ghi “Chưa có kết quả”.
7. Phân tích và kết luận đúng mức bằng chứng; các việc còn thiếu.
8. Mã nguồn, hash/config/log và tài liệu tham khảo; nội dung chuẩn bị vấn đáp.

## Những điều đề không bắt buộc

- **600 epoch:** là lựa chọn của hai run đã chạy. Đề không ấn định số epoch.
- **Split 45k/5k:** là cách bài làm chọn validation, không phải tỷ lệ ghi trong đề.
- **Retrain 50k:** là giao thức nghiên cứu mở rộng của repository, không phải một mục bắt buộc trong Final exam.docx.
- **Multi-seed, ablation:** hữu ích cho độ tin cậy/phân tích nhưng đề không yêu cầu rõ.
- **Accuracy 98% CIFAR-10 hoặc 90% CIFAR-100:** không phải ngưỡng bắt buộc trong đề. Hiệu năng được xếp hạng giữa các nhóm.
- **LR 0,1 và 0,15:** được đề nêu làm ví dụ cho nhiều initial LR, không phải một danh sách LR duy nhất bắt buộc cho cả SGD và Adam.

Các lựa chọn mở rộng vẫn phải tuân thủ quy tắc dự án khi chạy thí nghiệm mới: dùng validation để lựa chọn, giữ test kín trong quá trình chọn kiến trúc/recipe, giữ checkpoint và không dùng test để tối ưu.

## Kết quả đã có và việc cần làm để đủ thực nghiệm

Đã đo: CIFAR-10 best validation **97,56%**, test best train 45k **97,18%**; CIFAR-100 best validation **83,36%**, test best train 45k **82,98%**. Cả hai là SGD initial LR=0,1, 600 epoch selection, best EMA. Không gán hai điểm này cho lượt demo hoặc retrain 50k.

Phần cần bổ sung: chạy thực nghiệm full-data Adam và các LR khác trên hai dataset, lưu settings/lịch sử/checkpoint, thực hiện đánh giá cuối theo protocol, rồi cập nhật bảng so sánh bằng số đo thật. Hiện chưa có dữ liệu để kết luận SGD hơn Adam hay LR nào tốt nhất. Notebook có cell matrix để chạy các run selection này; không tự bật train dài chỉ vì cập nhật báo cáo.

Kế hoạch matrix hiện tại: SGD LR 0,05 / 0,1 / 0,15; Adam LR 0,0003 / 0,001 / 0,003. Đây là **kế hoạch 12 run trên hai dataset**, không phải 12 kết quả đã hoàn tất. Bảng CSV ghi rõ nguồn và trạng thái. Giữ cùng kiến trúc/split/seed/ngân sách khi so sánh; công khai nếu weight decay khác giữa optimizer.

## Thang điểm và vấn đáp

Theo đề: vấn đáp **30%**, hiệu năng CIFAR-10 **30%**, hiệu năng CIFAR-100 **30%**, chất lượng PDF **10%**. Đề lưu ý không qua vấn đáp thì trượt môn. Cần tự giải thích được công thức shape/parameters, kernel/stride/padding/groups, concat/ADD/multiply, SE, BN train/eval, loss, optimizer và ý nghĩa các kết quả.

Notebook: [CompactMix_CIFAR10_CIFAR100_BAO_CAO.ipynb](CompactMix_CIFAR10_CIFAR100_BAO_CAO.ipynb). Chạy cell cuối **Xuất PDF bám sát Final exam.docx** sau khi đọc evidence và cập nhật bảng matrix để tạo lại PDF/Markdown/CSV; cell này không train và không mở test loader.
