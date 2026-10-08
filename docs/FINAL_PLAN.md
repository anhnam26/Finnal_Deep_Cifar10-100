# Kế hoạch final Deep Learning

## 1. Yêu cầu và điểm số
Nguồn: `D:\DeepLearning\Finnal\Final exam.docx`.

- Một mô hình L tự thiết kế, learnable parameters <= 6M và FLOPs < 1G.
- Train/test trên CIFAR-10 và CIFAR-100, nhiều initial learning rates và cả SGD, Adam.
- Báo cáo learning settings đầy đủ trong PDF, nộp theo link giảng viên cung cấp.
- Không dùng nguyên kiến trúc mạng trước đây, trừ mô hình giữa kỳ của nhóm. Hai nhóm trùng kiến trúc không được đánh giá.
- Điểm: vấn đáp 30%, CIFAR-10 30%, CIFAR-100 30%, PDF 10%. Không đạt vấn đáp thì trượt môn.

## 2. Điều kiện và giả định
2–3 tuần; RTX 4060 Laptop 8 GB; Python/PyTorch hiện có, không tạo environment mới. Chưa có mô hình giữa kỳ được cung cấp. Chưa có accuracy hoặc tốc độ train thực đo khi lập kế hoạch.

Tạm dùng single-model, train from scratch, official CIFAR data, single-view inference. Không dùng pretrained, dữ liệu ngoài, teacher, ensemble hoặc TTA khi chưa được phép.

Cần hỏi giảng viên: quy ước FLOPs; metric xếp hạng; quy định kỹ thuật phụ trợ; cơ chế đăng ký kiến trúc. Cần xác nhận thiết kế khác nhóm khác và được chấp nhận trước run dài.

## 3. Chiến lược mô hình
CNN residual chuyên cho 32x32; stem 3x3 stride 1, không max pooling đầu; tăng kênh khi downsample; global average pooling; một backbone, head 10/100 lớp.

Ứng viên hiện tại: StageFusionNet; các kiến trúc thử nghiệm cũ đã được retire.

| Thành phần | Resolution | Channels | Blocks |
|---|---:|---:|---:|
| Stem | 32 | 40 | 1 conv |
| Stage 1 | 32 | 40 | 2 |
| Stage 2 | 16 | 104 | 3 |
| Stage 3 | 8 | 240 | 4 |
| Stage descriptors + classifier | 1 | 240 | 10/100 outputs |

Block: BN/ReLU -> conv3x3 -> BN/ReLU -> conv3x3 -> shortcut add, không post-add ReLU. Downsample ở conv thứ nhất và projection shortcut. Descriptor mỗi stage: BN/ReLU/GAP -> projection 240 chiều -> LayerNorm. Fusion: descriptor cuối cộng hai descriptor sớm được gate theo ảnh. no_fusion bỏ riêng early descriptors/gates.

Giả thuyết: các descriptor sớm giữ bằng chứng chi tiết bổ sung cho descriptor cuối. Phải kiểm chứng validation/ablation, không mặc định tăng accuracy. Residual/GAP/gating là thành phần đã biết; không tuyên bố phát minh mới. Architecture cần giảng viên xác nhận không trùng.

Tối đa ba ứng viên: baseline, multi-scale, điều chỉnh nhẹ ứng viên tốt hơn. Sàng lọc CIFAR-100, xác nhận CIFAR-10. Mục tiêu khoảng 4.5–5.7M params, 0.65–0.90 GFLOPs nhưng số thực đo mới quyết định. Không tối giản quá mức nếu mất accuracy, không vượt giới hạn.

## 4. Resource protocol
Input [1,3,32,32], forward inference. Bao gồm classifier, BN affine, projection, gate. Report cả MACs/FLOPs; conv/linear 1 MAC = 2 FLOPs. Kiểm tra custom ops không bị bỏ sót, ghi rõ estimates/fusion convention, chừa margin. Latency/VRAM chỉ bổ sung, không thay giới hạn.

## 5. Dữ liệu và chống leakage
Official train/test: mỗi dataset 50k/10k ảnh RGB 32x32. CIFAR-10 10 lớp; CIFAR-100 100 fine labels.

Development: stratified 45k train/5k validation, split_seed 2026. Mỗi lớp giữ 500 (C10) hoặc 50 (C100) ảnh validation. Lưu indices và fingerprint. Train augmentation và eval transforms dùng dataset objects riêng. Fixed normalization constants được công khai, không tính statistics từ test.

Validation chọn architecture, LR/optimizer, recipe, epochs và checkpoint. Không đọc test trong train loop. Sau khóa quyết định, train lại từ đầu với đủ 50k và frozen epoch budget; đánh giá cuối ở final epoch. Nếu báo cáo test cho grid, khóa configs và checkpoint selection bằng validation trước khi evaluate test. Không quay lại tuning sau test.

## 6. Ma trận bắt buộc
Mỗi dataset:
- SGD: LR 0.05, 0.10, 0.15; momentum 0.9; Nesterov true; WD 5e-4.
- Adam (không thay bằng AdamW): LR 0.0003, 0.001, 0.003; betas 0.9/0.999; eps 1e-8; WD 1e-4.

Tổng 12 configurations. Giữ architecture, split/seed, batch size, epochs và augmentation cố định. WD khác giữa optimizer phải công khai; đây là so sánh recipes được khảo sát, không kết luận optimizer nào luôn tốt hơn.

Khởi đầu batch 128, AMP nếu ổn định, 150 epochs, warmup 5 epochs và per-step cosine. Report LR bước đầu, peak LR và final LR. Baseline augmentation random crop padding4 + horizontal flip. CE; chưa CutMix/smoothing ở grid đầu.

## 7. Cải thiện recipe
Sau baseline: CutMix -> smoothing nhẹ nếu cần -> WD -> EMA nếu còn ngân sách -> longer schedule. Không bật mọi thứ đồng thời. CutMix lambda dùng diện tích patch thực sau clip; training accuracy phải chú thích weighted. EMA cần xử lý BN buffers đúng nếu triển khai.

Ablation ít nhất trên CIFAR-100: bỏ stage fusion; sau đó recipe controls (CutMix, augmentation, EMA nếu triển khai). Xác nhận trên C10. Ghi params/FLOPs và thời gian.

## 8. Budget và timeline
Đo epoch time, images/s, peak VRAM sau warmup. Tổng giờ = sum(epochs x epoch_time) + validation/checkpoint; giữ 20–30% dự phòng.

| Ngày | Công việc | Deliverable |
|---|---|---|
| 1 | Quy định/data/environment | Protocol |
| 2–3 | Model/tests/profiling | Baseline hợp lệ |
| 4–5 | <=3 kiến trúc | Chốt backbone |
| 6–9 | 12 optimizer/LR runs | Bảng bắt buộc |
| 10–12 | Recipe/ablations | Config cuối |
| 13–16 | Full training/repeat seeds | Checkpoints/kết quả |
| 17–19 | Report/figures | PDF |
| 20–21 | Reproduction/oral rehearsal | Submission |

Sàng lọc 60–100 epochs; grid 120–150 common epochs tùy benchmark; final 250–350 dựa validation curves. Hướng tới 3 seeds mỗi dataset cho config cuối, không 3 seeds toàn grid. Nếu chỉ 14 ngày, cắt ý tưởng phụ trước, giữ SGD/Adam/nhiều LR/cả datasets; chừa >=2 ngày report/oral. Không so sánh như tương đương runs có budgets khác nhau. Không chọn seed bằng test accuracy.

## 9. Engineering và kiểm tra
Configs JSON, metrics CSV, summaries/environment/splits JSON; checkpoint model/optimizer/scaler/RNG/epoch và resume epoch-boundary. Không framework quản lý thí nghiệm mới. Tests: shapes/gradients/constraints, balanced disjoint split, deterministic eval transform, schedule endpoints, CutMix area, tiny overfit, checkpoint identity và resume. Synthetic benchmark không thay kết quả CIFAR.

## 10. PDF và vấn đáp
PDF: problem/constraints; theory; architecture diagram/stage table; resource method; datasets/protocol; full training settings; optimizer/LR tables; final accuracy mean/std nếu có; learning curves; ablations; confusion/per-class errors; limitations; references; reproduction commands.

Oral: CNN/residual/gradient; stride/channels/receptive fields; depthwise vs dense; gate; parameters vs FLOPs; SGD momentum vs Adam; LR/warmup/cosine; BN train/eval; CutMix/WD/generalization; train/val/test; C100 difficulty. Mọi quyết định giải thích theo: thay gì, vì sao, bằng chứng nào.

## 11. Completion gates
- Giảng viên chấp nhận kiến trúc và không trùng nhóm khác.
- Params <=6M và FLOPs <1G trên cả hai heads.
- SGD/Adam và nhiều LR trên cả datasets, settings/logs đầy đủ.
- Không test leakage; checkpoint/config/lệnh tái lập.
- Ablations; PDF; vấn đáp đạt yêu cầu.
- Không cam kết accuracy/thứ hạng trước thực nghiệm.

## 12. Tài liệu nền
- CIFAR official: https://www.cs.toronto.edu/~kriz/cifar.html ; Krizhevsky, Learning Multiple Layers of Features from Tiny Images, 2009.
- Wide Residual Networks: https://arxiv.org/abs/1605.07146 . Tham khảo trade-off width/depth, không sao chép nguyên kiến trúc.
- CutMix: https://arxiv.org/abs/1905.04899 . Tham khảo regularization, không coi accuracy công bố là bảo đảm.