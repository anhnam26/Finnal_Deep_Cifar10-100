# Lý do chọn CompactMix-L và lịch sử các kiến trúc

## 1. Lý do chọn kiến trúc hiện tại

CompactMix-L được thiết kế để cân bằng khả năng học đặc trưng và giới hạn **≤6 triệu tham số, <1 GFLOP/ảnh** của `Final exam.docx`.

| Lựa chọn | Cơ chế và lý do dùng |
|---|---|
| Stem Conv 3×3, stride 1 | Giữ ảnh CIFAR ở 32×32 để hạn chế mất chi tiết ngay đầu mạng |
| Ba stage 40/112/256 kênh, độ sâu 2/3/8 | Tăng khả năng biểu diễn khi kích thước không gian giảm 32→16→8; bố trí nhiều block ở 8×8 để kiểm soát tính toán |
| Hai nhánh depthwise 3×3 và 5×5 | Xử lý đặc trưng ở hai phạm vi lân cận với ít weight hơn Conv thường cùng số kênh |
| Grouped Conv, groups 8 | Kết hợp đặc trưng trong từng nhóm và giảm chi phí so với Conv thường cùng kênh/kernel |
| Conv 1×1 expand/project | Trộn thông tin giữa các kênh; project đưa về số kênh phù hợp để cộng residual |
| SE | Dùng mô tả toàn ảnh để học hệ số điều chỉnh mức đóng góp của từng kênh |
| Residual | Giữ đường truyền tín hiệu và gradient qua skip connection |
| DropPath | Regularize nhánh residual trong train, giảm sự phụ thuộc vào các nhánh cụ thể |
| GAP và Linear | Tổng hợp mỗi kênh thành một feature rồi phân loại 10/100 lớp, giảm kích thước đầu vào classifier |

Các cơ chế trên giải thích động cơ thiết kế; chưa có ablation đủ để quy riêng mức tăng accuracy cho từng thành phần hoặc khẳng định CompactMix tốt hơn mọi kiến trúc cũ trong điều kiện tương đương.

## 2. Những kiến trúc trước đây và kết quả tìm được

**Các accuracy lịch sử bên dưới là best validation, không phải test.** Những mục không còn đủ dữ liệu được ghi rõ, không suy ra điểm từ việc có code hoặc config.

| Kiến trúc | Đặc điểm chính | Kết quả được ghi lại |
|---|---|---|
| LightNet baseline | Dense residual CNN,4 stage | CIFAR-10: val **95,50%** ở lượt 150 epoch; **95,46%** ở lượt 300 epoch |
| LightNet multiscale | Thêm nhánh depthwise 5×5 và channel gate ở các stage sau | Có lịch sử cấu hình/run; không còn đủ số liệu gốc để xác nhận điểm cụ thể |
| LightNet hybrid | Dense residual kết hợp các block pointwise–depthwise–pointwise và SE tại 8×8 | Có thiết kế và kiểm tra chạy thử; chưa tìm thấy kết quả train dài hoàn chỉnh |
| Preactivation backbone (`preact`) | BN–ReLU trước Conv; kênh 40/104/248, độ sâu 2/3/4 | CIFAR-10: val **95,22%**,150 epoch |
| StageFusionNet — no_fusion | Backbone không dùng phần fusion |150 epoch: CIFAR-10 val **95,18%**; CIFAR-100 val **75,86%** theo log |
| StageFusionNet — fusion | Kết hợp thêm thông tin từ các stage |150 epoch: CIFAR-10 val **95,30%**; CIFAR-100 val **76,54%** |
| ScaleGateNet | Chia kênh thành các nhánh liên kết phân cấp, gate tùy chọn; kênh 40/120/320 | Xác nhận được các lượt chạy thử 2 epoch; chưa tìm thấy kết quả hoàn chỉnh của kế hoạch 300 epoch |
| CompactMix-L hiện tại | Hai nhánh DW 3×3/5×5, grouped Conv, SE, residual | Đã hoàn thành 600 epoch trên cả CIFAR-10 và CIFAR-100; kết quả chi tiết ở mục 3 |

Các điểm cần giữ đúng khi đọc lịch sử:

- Baseline 150 và baseline 300 có lịch LR khác nhau; không xem việc tăng số epoch là một cải tiến kiến trúc.
- Riêng StageFusionNet CIFAR-100 no_fusion, log ghi 75,86% nhưng export cũ ghi 75,84%, chênh một ảnh validation. Nguyên nhân chưa được xác nhận.
- StageFusionNet fusion chỉ hơn no_fusion 0,12 điểm phần trăm trên CIFAR-10 và 0,68 điểm trên CIFAR-100 ở một seed; chưa chứng minh lợi ích ổn định qua nhiều seed.
- Phần lớn checkpoint/log cũ đã được dọn. Các số lịch sử được đối chiếu từ tài liệu lưu trong Git, không phải kết quả đánh giá lại model cũ ở lần tổng hợp này.

## 3. Kết quả CompactMix-L hiện tại

Cùng backbone 40/112/256 kênh,2/3/8 block; classifier ra 10 hoặc 100 lớp. Hai run dùng SGD initial LR 0,1,600 epoch, train 45.000 ảnh và validation 5.000 ảnh; chọn best EMA theo validation.

| Dataset | Learnable Params | GFLOPs/ảnh | Best validation | Test accuracy | Epoch best / weights |
|---|---:|---:|---:|---:|---|
| CIFAR-10 |5.248.172 |0,842406944 |**97,56%** |**97,18%** |567 / EMA |
| CIFAR-100 |5.271.302 |0,842453114 |**83,36%** |**82,98%** |399 / EMA |

FLOPs tính cho input `[1,3,32,32]` theo quy ước profiler của bài. Cả hai head đều đạt giới hạn tham số và FLOPs của đề. Các điểm test là của **best train 45k**, không phải retrain 50k; lần kiểm tra bằng `evaluate_best.py` dùng đủ 10.000 ảnh test mỗi dataset và khớp điểm đã lưu.

Không so các số này với những model cũ như một thí nghiệm chỉ thay kiến trúc: số epoch, augmentation, EMA và các thành phần recipe khác nhau. Mức tăng quan sát được thuộc toàn bộ cấu hình train, chưa tách được đóng góp riêng của backbone.

## 4. Phân biệt kiến trúc và các bản code

- `src/models.py`, `report/compactmix_manual_train.py` và notebook hiện tại triển khai cùng cấu hình CompactMix-L gốc; bản manual viết tường minh từng block để thuận tiện học và sửa. Đây không phải ba kiến trúc độc lập để so accuracy.
- Các family như WRN, ConvNeXt, ViT có code hỗ trợ không đồng nghĩa đã train và có kết quả hoàn chỉnh. Không tìm thấy kết quả train đầy đủ của chúng trong các run hiện còn lưu.
- Tổ hợp Conv/SE/residual dựa trên những ý tưởng đã biết. Chưa có bằng chứng xác nhận kiến trúc độc nhất hoặc không trùng nhóm khác.

## 5. Nguồn đối chiếu

Kết quả hiện tại:

- [Báo cáo theo đề](BAO_CAO_THEO_DE_FINAL_EXAM.md).
- [Evidence CIFAR-10](evidence/cifar10/result.json) và [CIFAR-100](evidence/cifar100/result.json).
- [Thông tin best CIFAR-10](checkpoints/cifar10/checkpoint_info.json) và [CIFAR-100](checkpoints/cifar100/checkpoint_info.json).
- [Hướng dẫn và kết quả chạy test](CHAY_TEST_BEST.md).

Tài liệu lịch sử còn trong Git, xem bằng `git show <commit>:<path>`:

| Nội dung | Commit và đường dẫn |
|---|---|
| LightNet baseline/multiscale | `d8e8af8:models/lightnet.py` |
| Hybrid | `e62164e:docs/CIFAR_IMPROVEMENT.md` |
| Preactivation và baseline 300 | `f21aa81:docs/CIFAR_REDESIGN.md` |
| Tổng hợp baseline 150/300 và preactivation 150 | `ab14b8f:README.md` |
| Kết quả StageFusionNet | `ed80f90:docs/RETIRED_STAGEFUSION.md` |
| Thiết kế và kiểm tra ScaleGateNet | `ab14b8f:docs/SCALEGATE_RESEARCH.md`, `ab14b8f:docs/SCALEGATE_VALIDATION.md` |
