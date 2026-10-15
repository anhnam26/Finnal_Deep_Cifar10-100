# BÁO CÁO FINAL EXAM — COMPACTMIX-L

Sinh viên/nhóm: ........................................ | Lớp: ........................ | Giảng viên: ........................

Nguồn yêu cầu: Final exam.docx. Báo cáo tổ chức theo yêu cầu bắt buộc trong đề; kết quả cũ được kiểm chứng từ evidence, không lấy accuracy demo để thay điểm full-data.

Trạng thái: còn thiếu thực nghiệm so sánh full-data Adam và các LR khác; cần đối chiếu tính khác biệt kiến trúc. Không tuyên bố đã hoàn thành toàn bộ đề.

## 1. Đối chiếu yêu cầu của đề

| Yêu cầu | Trạng thái | Bằng chứng / việc còn lại |
|---|---|---|
| Thiết kế model L, mô tả kiến trúc | Đã mô tả | CompactMix-L: widths 40/112/256, depths 2/3/8; code Conv/forward tường minh. |
| Tham số học được không quá 6M | Đạt | CIFAR-10: 5.248.172; CIFAR-100: 5.271.302; kiểm tra bằng forward/profile. |
| FLOPs dưới 1G | Đạt theo convention bài làm | 842.406.944 và 842.453.114 FLOPs/ảnh; 1 MAC=2 FLOPs cộng các phép toán bổ sung. |
| Train/test CIFAR-10 và CIFAR-100 | Đã có ở SGD LR=0,1 | Hai selection 45k/5k, 600 epoch; test best trên đủ 10k ảnh mỗi dataset. |
| SGD, Adam và các initial LR khác nhau | CHƯA HOÀN THÀNH | Xem bảng optimizer × LR × dataset; ô chưa đo giữ trống. Code hỗ trợ chạy không thay thế kết quả thực nghiệm. |
| Mô tả learning settings chi tiết | Đã ghi | LR, momentum/betas, decay, epochs, batch, scheduler, loss, augmentation, AMP/EMA, seed và môi trường. |
| Nộp báo cáo PDF | Đã xuất PDF | File BAO_CAO_THEO_DE_FINAL_EXAM.pdf; cần điền thông tin sinh viên và kiểm tra trước khi nộp. |
| Kiến trúc không trùng nhóm/mạng trước | CẦN ĐỐI CHIẾU | Mô tả cấu trúc riêng; chưa có bằng chứng xác nhận không trùng tất cả nhóm hay mạng đã công bố. |
| Vấn đáp và nộp đúng hạn | Sinh viên thực hiện | Vấn đáp 30%; hiệu năng C10 30%, C100 30%; chất lượng PDF 10%. Không qua vấn đáp thì trượt môn. |

Đề không quy định số epoch cố định, không bắt buộc 600 epoch, split 45k/5k, retrain 50k, multi-seed, ablation hoặc ngưỡng accuracy 98%/90%. Những mục này là lựa chọn thực nghiệm hoặc mở rộng của dự án. Hiệu năng được chấm theo thứ hạng giữa các nhóm.

## 2. Thiết kế model L và kiến trúc

CompactMix-L dùng stem Conv3×3 3→40; ba stage 40/112/256 kênh với 2/3/8 MixBlock; transition BN→SiLU→AvgPool2×2→Conv1×1; head BN→SiLU→GlobalAvgPool→Flatten→Linear. Hai dataset dùng chung backbone, head xuất 10 hoặc 100 logits.

| Khối | Output | Chi tiết |
|---|---|---|
| Stem | [N,40,32,32] | Conv3×3, s=1,p=1, không bias |
| Stage 1 | [N,40,32,32] | 2 MixBlock |
| Transition 1 / Stage 2 | [N,112,16,16] | Giảm không gian; 3 MixBlock |
| Transition 2 / Stage 3 | [N,256,8,8] | Giảm không gian; 8 MixBlock |
| Head | [N,10] / [N,100] | GAP→flatten→Dropout(p=0)→Linear |

Mỗi MixBlock: BN/SiLU→expand 1×1 C→2C→chia đôi kênh→depthwise 3×3 và 5×5→concat→BN/SiLU→grouped Conv3×3 groups=8→BN/SiLU→SE→projection 1×1 2C→C→DropPath→ADD với skip. SE pool về 1×1, Conv giảm/mở kênh và Sigmoid tạo gate; D=8/14/32 theo stage.

![Luồng model CompactMix-L](../experiments/notebook_presentation/architecture.png)

Tính khác biệt: tổ hợp hai kernel trên hai phần kênh, grouped spatial mixing, SE và cấu hình stage được tự triển khai trong bài. Các thành phần Conv/SE/residual đã có trong tài liệu. Cần trình bày nguồn tham khảo và đối chiếu kiến trúc với nhóm khác/mạng liên quan; chưa có cơ sở khẳng định tính mới hay độc nhất.

## 3. Kiểm chứng tham số và FLOPs

| Dataset | Parameters | MACs/ảnh | FLOPs/ảnh | GFLOPs |
|---|---|---|---|---|
| cifar10 | 5,248,172 | 409,492,864 | 842,406,944 | 0.842407 |
| cifar100 | 5,271,302 | 409,515,904 | 842,453,114 | 0.842453 |

Conv: P=Cout×(Cin/groups)×kH×kW, thêm Cout nếu có bias. Linear: P=Fin×Fout+Fout nếu có bias. BN affine: P=2C. MAC Conv=N×Hout×Wout×Cout×(Cin/groups)×kH×kW. Profiler dùng N=1, 1 MAC=2 FLOPs, cộng bias/BN/SiLU/Sigmoid/pooling/SE MUL/residual ADD; bỏ qua memory operations split/cat/reshape. Đây là ước lượng inference, không phải latency hoặc training FLOPs.

Đếm trực tiếp model.parameters(), thực chạy forward để lấy shape và kiểm tra backward. Cả hai head đạt ≤6M tham số và <1G FLOPs theo convention trên. Code và bảng layer đầy đủ nằm trong notebook/CSV của output.

## 3.1. Bảng tổng hợp cộng dồn từng khối

Input [1,3,32,32]; hai cột shape bỏ chiều batch. B(C,E,D;H,W) là MixBlock với E=2C, hai DW3×3/5×5, grouped Conv3×3 g=8, SE E→D→E ở 1×1, hai PW và 4 BN. Stem thực tế không có BN. Σ là tổng đến khối đó; không cộng lại các giá trị Σ để tính tổng toàn mạng.

P_B=2CE+34C+9E²/8+2ED+(D+E)+2(C+3E). MAC_B=HW×(2CE+34C+9E²/8)+2ED. BN đóng góp 2(C+3E) tham số nhưng 0 MAC Conv/Linear; hai Conv SE tính trên 1×1, không nhân lại với H×W.

## CIFAR10 — bảng cộng dồn toàn mô hình

CIFAR10: 5,248,172 Params; 0.409492864 GMac; 0.842406944 GFLOPs.

| Khối | Input C×H×W | Output C×H×W | Chi tiết công thức / cấu trúc | Params / Σ Params | MACs / Σ MACs |
|---|---|---|---|---|---|
| stem | 3×32×32 | 40×32×32 | Conv 3×3 3→40; s=1,p=1; không BN | 1,080; Σ 1,080 | 1.105920 M; Σ 1.105920 M |
| block01 | 40×32×32 | 40×32×32 | B(C=40,E=80,D=8;H=32,W=32); 4 BN + SiLU + SE + skip | 16,888; Σ 17,968 | 15.320320 M; Σ 16.426240 M |
| block02 | 40×32×32 | 40×32×32 | B(C=40,E=80,D=8;H=32,W=32); 4 BN + SiLU + SE + skip | 16,888; Σ 34,856 | 15.320320 M; Σ 31.746560 M |
| transition1 | 40×32×32 | 112×16×16 | BN(40) + SiLU + AvgPool2×2,s2 + Conv1×1 40→112 | 4,560; Σ 39,416 | 1.146880 M; Σ 32.893440 M |
| block03 | 112×16×16 | 112×16×16 | B(C=112,E=224,D=14;H=16,W=16); 4 BN + SiLU + SE + skip | 118,510; Σ 157,926 | 28.276864 M; Σ 61.170304 M |
| block04 | 112×16×16 | 112×16×16 | B(C=112,E=224,D=14;H=16,W=16); 4 BN + SiLU + SE + skip | 118,510; Σ 276,436 | 28.276864 M; Σ 89.447168 M |
| block05 | 112×16×16 | 112×16×16 | B(C=112,E=224,D=14;H=16,W=16); 4 BN + SiLU + SE + skip | 118,510; Σ 394,946 | 28.276864 M; Σ 117.724032 M |
| transition2 | 112×16×16 | 256×8×8 | BN(112) + SiLU + AvgPool2×2,s2 + Conv1×1 112→256 | 28,896; Σ 423,842 | 1.835008 M; Σ 119.559040 M |
| block06 | 256×8×8 | 256×8×8 | B(C=256,E=512,D=32;H=8,W=8); 4 BN + SiLU + SE + skip | 602,656; Σ 1,026,498 | 36.241408 M; Σ 155.800448 M |
| block07 | 256×8×8 | 256×8×8 | B(C=256,E=512,D=32;H=8,W=8); 4 BN + SiLU + SE + skip | 602,656; Σ 1,629,154 | 36.241408 M; Σ 192.041856 M |
| block08 | 256×8×8 | 256×8×8 | B(C=256,E=512,D=32;H=8,W=8); 4 BN + SiLU + SE + skip | 602,656; Σ 2,231,810 | 36.241408 M; Σ 228.283264 M |
| block09 | 256×8×8 | 256×8×8 | B(C=256,E=512,D=32;H=8,W=8); 4 BN + SiLU + SE + skip | 602,656; Σ 2,834,466 | 36.241408 M; Σ 264.524672 M |
| block10 | 256×8×8 | 256×8×8 | B(C=256,E=512,D=32;H=8,W=8); 4 BN + SiLU + SE + skip | 602,656; Σ 3,437,122 | 36.241408 M; Σ 300.766080 M |
| block11 | 256×8×8 | 256×8×8 | B(C=256,E=512,D=32;H=8,W=8); 4 BN + SiLU + SE + skip | 602,656; Σ 4,039,778 | 36.241408 M; Σ 337.007488 M |
| block12 | 256×8×8 | 256×8×8 | B(C=256,E=512,D=32;H=8,W=8); 4 BN + SiLU + SE + skip | 602,656; Σ 4,642,434 | 36.241408 M; Σ 373.248896 M |
| block13 | 256×8×8 | 256×8×8 | B(C=256,E=512,D=32;H=8,W=8); 4 BN + SiLU + SE + skip | 602,656; Σ 5,245,090 | 36.241408 M; Σ 409.490304 M |
| head | 256×8×8 | 256 | BN(256) + SiLU + GAP1×1 + flatten + Dropout(p=0); MACs Conv/Linear=0 | 512; Σ 5,245,602 | 0.000000 M; Σ 409.490304 M |
| classifier | 256 | 10 | Linear 256→10; P=256×10+10; MAC=256×10 | 2,570; Σ 5,248,172 | 0.002560 M; Σ 409.492864 M |
| TỔNG | 3×32×32 | 10 | Cộng từng khối một lần | 5,248,172 | 409.492864 M |

## CIFAR100 — bảng cộng dồn toàn mô hình

CIFAR100: 5,271,302 Params; 0.409515904 GMac; 0.842453114 GFLOPs.

| Khối | Input C×H×W | Output C×H×W | Chi tiết công thức / cấu trúc | Params / Σ Params | MACs / Σ MACs |
|---|---|---|---|---|---|
| stem | 3×32×32 | 40×32×32 | Conv 3×3 3→40; s=1,p=1; không BN | 1,080; Σ 1,080 | 1.105920 M; Σ 1.105920 M |
| block01 | 40×32×32 | 40×32×32 | B(C=40,E=80,D=8;H=32,W=32); 4 BN + SiLU + SE + skip | 16,888; Σ 17,968 | 15.320320 M; Σ 16.426240 M |
| block02 | 40×32×32 | 40×32×32 | B(C=40,E=80,D=8;H=32,W=32); 4 BN + SiLU + SE + skip | 16,888; Σ 34,856 | 15.320320 M; Σ 31.746560 M |
| transition1 | 40×32×32 | 112×16×16 | BN(40) + SiLU + AvgPool2×2,s2 + Conv1×1 40→112 | 4,560; Σ 39,416 | 1.146880 M; Σ 32.893440 M |
| block03 | 112×16×16 | 112×16×16 | B(C=112,E=224,D=14;H=16,W=16); 4 BN + SiLU + SE + skip | 118,510; Σ 157,926 | 28.276864 M; Σ 61.170304 M |
| block04 | 112×16×16 | 112×16×16 | B(C=112,E=224,D=14;H=16,W=16); 4 BN + SiLU + SE + skip | 118,510; Σ 276,436 | 28.276864 M; Σ 89.447168 M |
| block05 | 112×16×16 | 112×16×16 | B(C=112,E=224,D=14;H=16,W=16); 4 BN + SiLU + SE + skip | 118,510; Σ 394,946 | 28.276864 M; Σ 117.724032 M |
| transition2 | 112×16×16 | 256×8×8 | BN(112) + SiLU + AvgPool2×2,s2 + Conv1×1 112→256 | 28,896; Σ 423,842 | 1.835008 M; Σ 119.559040 M |
| block06 | 256×8×8 | 256×8×8 | B(C=256,E=512,D=32;H=8,W=8); 4 BN + SiLU + SE + skip | 602,656; Σ 1,026,498 | 36.241408 M; Σ 155.800448 M |
| block07 | 256×8×8 | 256×8×8 | B(C=256,E=512,D=32;H=8,W=8); 4 BN + SiLU + SE + skip | 602,656; Σ 1,629,154 | 36.241408 M; Σ 192.041856 M |
| block08 | 256×8×8 | 256×8×8 | B(C=256,E=512,D=32;H=8,W=8); 4 BN + SiLU + SE + skip | 602,656; Σ 2,231,810 | 36.241408 M; Σ 228.283264 M |
| block09 | 256×8×8 | 256×8×8 | B(C=256,E=512,D=32;H=8,W=8); 4 BN + SiLU + SE + skip | 602,656; Σ 2,834,466 | 36.241408 M; Σ 264.524672 M |
| block10 | 256×8×8 | 256×8×8 | B(C=256,E=512,D=32;H=8,W=8); 4 BN + SiLU + SE + skip | 602,656; Σ 3,437,122 | 36.241408 M; Σ 300.766080 M |
| block11 | 256×8×8 | 256×8×8 | B(C=256,E=512,D=32;H=8,W=8); 4 BN + SiLU + SE + skip | 602,656; Σ 4,039,778 | 36.241408 M; Σ 337.007488 M |
| block12 | 256×8×8 | 256×8×8 | B(C=256,E=512,D=32;H=8,W=8); 4 BN + SiLU + SE + skip | 602,656; Σ 4,642,434 | 36.241408 M; Σ 373.248896 M |
| block13 | 256×8×8 | 256×8×8 | B(C=256,E=512,D=32;H=8,W=8); 4 BN + SiLU + SE + skip | 602,656; Σ 5,245,090 | 36.241408 M; Σ 409.490304 M |
| head | 256×8×8 | 256 | BN(256) + SiLU + GAP1×1 + flatten + Dropout(p=0); MACs Conv/Linear=0 | 512; Σ 5,245,602 | 0.000000 M; Σ 409.490304 M |
| classifier | 256 | 100 | Linear 256→100; P=256×100+100; MAC=256×100 | 25,700; Σ 5,271,302 | 0.025600 M; Σ 409.515904 M |
| TỔNG | 3×32×32 | 100 | Cộng từng khối một lần | 5,271,302 | 409.515904 M |

1 MMac=10^6 MAC; 1 GMac=10^9 MAC. MACs trong bảng chỉ đếm Conv/Linear. Tổng FLOPs lớn hơn 2×MACs vì còn các phép BN, activation, pool, SE multiply và residual add. Bảng CSV giữ số nguyên chính xác; PNG và Markdown riêng có trong report/figures và report/BANG_CONG_DON_THAM_SO_MACS.md.

## 4. Dữ liệu và learning settings

CIFAR-10/100 có 50.000 ảnh train và 10.000 test, RGB 32×32. Bài chọn stratified split 45.000 train / 5.000 validation với split seed 2026; seed khởi tạo 42. Không giao nhau, phủ đủ 50k indices. Validation dùng chọn best; hai checkpoint best 45k được đánh giá trên đủ 10k test theo yêu cầu người dùng trước đó.

Train: RandomCrop(32,padding=4,reflect), HorizontalFlip, AutoAugment CIFAR10 policy cho cả hai dataset, ToTensor, Normalize theo mean/std riêng. Validation/test chỉ ToTensor/Normalize; không TTA, ensemble, pretrained hoặc dữ liệu ngoài trong hai phép đo đã có.

| Learning setting | Giá trị của thí nghiệm/bản kế hoạch |
|---|---|
| Optimizer / initial LR | SGD / 0,1 |
| Momentum / Nesterov | 0,9 / True |
| Weight decay | 0,0005 cho tham số ndim>1; 0 cho nhóm còn lại |
| Epochs / batch size | 600 / 128 |
| Scheduler / warmup | Cosine / 5 epoch |
| Loss | Soft-label cross entropy; label smoothing=0,05 |
| Mixup / CutMix / xác suất | 0,2 / 1,0 / 0,75; tắt mixing ở đoạn cuối lịch |
| EMA decay / clipping | Tối đa 0,999 / grad norm 5 |
| AMP / deterministic | CUDA AMP / True |
| DropPath / head dropout | Tăng dần 0→0,1 / 0 |
| Workers đã train / notebook | 2 / 0 (Jupyter Windows) |
| Seed model / split | 42 / 2026 |
| Adam được hỗ trợ | betas=(0,9;0,999), eps=1e-8; LR kế hoạch 0,0003/0,001/0,003; chưa có full-data result |

Train loss dùng soft targets; mixed accuracy là tỷ lệ đúng có trọng số theo nhãn trộn. Validation/test dùng nhãn cứng và cross entropy chuẩn. Không coi train mixed accuracy là accuracy sạch, không diễn giải hai loss là hoàn toàn cùng bài toán.

Môi trường thí nghiệm đã lưu: {"python": "3.13.14", "pytorch": "2.7.1+cu118", "numpy": "2.2.6", "cuda": "11.8", "device": "cuda", "platform": "Windows", "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU"}. Notebook dùng chính Microsoft Store Python 3.13 hiện tại; không tạo môi trường khác. Config/runtime/source/split/seed được lưu cùng checkpoint.

## 5. Kết quả CIFAR-10 và CIFAR-100 đã đo

| Dataset | Best epoch | Val % | Test % | Top-5 % | Macro-F1 | Đúng/tổng |
|---|---|---|---|---|---|---|
| cifar10 | 567 / ema | 97.56 | 97.18 | 99.87 | 0.971811 | 9718/10000 |
| cifar100 | 399 / ema | 83.36 | 82.98 | 96.75 | 0.829492 | 8298/10000 |

Phạm vi: hai điểm test là của best selection train 45k, không phải demo subset và không phải retrain 50k. Điểm được kiểm chứng bằng SHA256 checkpoint, số ảnh và đường chéo confusion matrix. Không tự đo thêm test trong bước xuất báo cáo.

cifar10: train/validation đã hoàn thành 600 epoch; tổng thời gian engine 50979.17 giây, trung bình 84.97 giây/epoch; tổng AMP skipped updates 79. Test loss=0.185050.

![cifar10 — lịch sử đầy đủ 600 epoch](../experiments/notebook_presentation/cifar10_archived_curves.png)

![cifar10 — confusion matrix và recall thấp nhất](../experiments/notebook_presentation/cifar10_archived_confusion_recall.png)

cifar100: train/validation đã hoàn thành 600 epoch; tổng thời gian engine 49986.76 giây, trung bình 83.31 giây/epoch; tổng AMP skipped updates 82. Test loss=0.651891.

![cifar100 — lịch sử đầy đủ 600 epoch](../experiments/notebook_presentation/cifar100_archived_curves.png)

![cifar100 — confusion matrix và recall thấp nhất](../experiments/notebook_presentation/cifar100_archived_confusion_recall.png)

Accuracy=100×correct/total; precision=TP/(TP+FP); recall=TP/(TP+FN); F1=2PR/(P+R); macro-F1 là trung bình các lớp. Confusion dùng hàng=true, cột=predicted. Precision/recall/F1/support toàn bộ lớp có trong CSV; top-5 là nhãn thật thuộc 5 logits cao nhất.

## 6. So sánh SGD / Adam và các initial learning rates

Đây là yêu cầu thực nghiệm trong đề, hiện chưa đủ bằng chứng hoàn tất. Các LR cụ thể là lựa chọn của bài; đề nêu 0,1 và 0,15 làm ví dụ, không ấn định số run hoặc số epoch. Bảng dưới có kế hoạch 3 LR mỗi optimizer trên 2 dataset; không xem những ô trống là accuracy bằng 0.

| Dataset | Opt | LR | Epochs | Best val % | Test % | Trạng thái |
|---|---|---|---|---|---|---|
| cifar10 | sgd | 0.05 | — | — | — | Chưa có kết quả |
| cifar100 | sgd | 0.05 | — | — | — | Chưa có kết quả |
| cifar10 | sgd | 0.1 | 600 | 97.56 | 97.18 | Đã train và test |
| cifar100 | sgd | 0.1 | 600 | 83.36 | 82.98 | Đã train và test |
| cifar10 | sgd | 0.15 | — | — | — | Chưa có kết quả |
| cifar100 | sgd | 0.15 | — | — | — | Chưa có kết quả |
| cifar10 | adam | 0.0003 | — | — | — | Chưa có kết quả |
| cifar100 | adam | 0.0003 | — | — | — | Chưa có kết quả |
| cifar10 | adam | 0.001 | — | — | — | Chưa có kết quả |
| cifar100 | adam | 0.001 | — | — | — | Chưa có kết quả |
| cifar10 | adam | 0.003 | — | — | — | Chưa có kết quả |
| cifar100 | adam | 0.003 | — | — | — | Chưa có kết quả |

Giữ cùng kiến trúc, split, seed và ngân sách epoch khi so sánh; ghi đầy đủ settings mỗi run. Kế hoạch hiện dùng weight decay SGD=0,0005 và Adam=0,0001; nếu giữ khác nhau thì kết luận là so sánh recipe optimizer, không cô lập riêng tác động optimizer. Chọn candidate bằng validation, không dùng test để tối ưu LR/kiến trúc. Không có đủ kết quả để kết luận SGD tốt hơn Adam hoặc LR nào tốt nhất.

## 7. Kết luận theo đúng mức bằng chứng

CompactMix-L đạt ngân sách ≤6M/<1G theo profiler và đã train/test cả CIFAR-10, CIFAR-100 với SGD LR=0,1. Test best train 45k đạt 97,18% và 82,98%. Phần thực nghiệm so sánh Adam/nhiều LR chưa hoàn thành; cần đối chiếu tính khác biệt kiến trúc. Vì vậy báo cáo chưa tuyên bố hoàn thành toàn bộ yêu cầu final exam.

Retrain 50k, multi-seed và ablation là mở rộng, không phải thiếu sót bắt buộc theo văn bản đề. Quy tắc giữ kín test và freeze cho thí nghiệm mới vẫn tuân theo protocol repository. Các phép đo latency, macro-F1 hoặc sơ đồ chi tiết hỗ trợ giải thích nhưng không thay thế so sánh optimizer/LR.

## 8. Chuẩn bị vấn đáp và tài liệu kiểm chứng

Cần giải thích được: công thức shape Conv và tham số; groups/depthwise; concat khác ADD; SE multiply/broadcast; BN train/eval; SiLU; loss/optimizer; momentum; LR/warmup/cosine; EMA; best/last và resume; validation khác test; phạm vi của điểm số đã báo cáo. Theo đề, không qua vấn đáp thì trượt môn dù hiệu năng tốt.

| Dataset | SHA256 best đã đo test |
|---|---|
| cifar10 | 10df71eacd01d1b511d9e2ff66f65e9c3bb6fb9a017f1a59083fa399f8bb904d |
| cifar100 | fc8346b6493c70eea2804f1493c1e2fab7dfb5aaeb175730438106ea53118cae |

Code đầy đủ: CompactMix_CIFAR10_CIFAR100_BAO_CAO.ipynb. Evidence: report/evidence/cifar10 và cifar100; checkpoint best: report/checkpoints; bảng optimizer/LR: BANG_SO_SANH_SGD_ADAM_LR.csv. PDF snapshot trước: Bao_cao_CIFAR10_CIFAR100.pdf được giữ nguyên.

Tham khảo: PyTorch Training a Classifier (mẫu bố cục notebook); PyTorch nn API; residual learning (He et al., 2015); squeeze-and-excitation (Hu et al., 2017); grouped/depthwise convolution trong tài liệu liên quan. Các khối chuẩn không tự chứng minh kiến trúc độc nhất; cần đối chiếu thiết kế cuối với mạng tham khảo và các nhóm.

## Phụ lục A. Nguyên văn yêu cầu từ Final exam.docx

Requirements for the final examination:

Design a lightweight model (so-called L) that is in accordance with several conditions as follows:

Its number of learnable parameters is not over 6M.

Its FLOP is under 1G.

Train and test L on CIFAR-10 and CIFAR-100 in consideration of different initial learning rates: e.g., 0.1, 0.15,… along with optimizers: SGD and Adam. Report the learning settings in detail (e.g., momentum, learning rate, epochs,….).

All above should be figured out a pdf file for a final report.

Submit on time all subject to a link (provide later).

Evaluation criteria:

Answer several oral exam questions (30%).

The performance of L on CIFAR-10 (30%), the score is evaluated subject to the performance of L ranked among the student groups.

The performance of L on CIFAR-100 (30%), the score is evaluated subject to the performance of L ranked among the student groups.

The quality of the final report (10%)

Note:

If you cannot reply well (i.e., not pass) the oral exam questions, you will fall this course.

Two groups with the same model (architecture) will not be assessed.

The proposed model L is not the quite same as previous networks, except your proposed model in the midterm examination.

SHA256 đề bài: 7d371893f16263b91cc9b39d4a75824132ed2ac336faa110ad562715be3af479
