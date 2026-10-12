# CIFAR: train một file, log rõ ràng, tự resume

Chạy tại D:/DeepLearning/Finnal. Chỉ có một file Python vận hành: train.py.
Dữ liệu CIFAR đã có được dùng lại, không tải lại mặc định. Môi trường hiện tại
Python 3.13 / PyTorch 2.7.1+cu118 / torchvision 0.22.1 chạy được trên RTX 4060 8GB;
pipeline mới không cần timm hay pretrained weights, không tự cài package.

## Chạy ngay

    python train.py

Lệnh này chạy dualpath trên CIFAR-10 rồi CIFAR-100, mỗi dataset 600 epochs,
SGD LR 0.1, batch 128, workers 2, AMP trên CUDA. Đây là queue train dài;
không khởi chạy thêm cửa sổ train song song. Terminal và file log chỉ ghi một
dòng tổng kết khi mỗi epoch đã train và validation xong, cùng kết quả test cuối run.

Nếu phiên gần nhất chưa hoàn thành, chính lệnh trên tiếp tục từ checkpoint.
Nếu phiên trước đã hoàn thành, lệnh tạo phiên mới, không ghi đè kết quả cũ.
Khi resume, recipe của phiên được giữ nguyên; muốn đổi epochs/model/LR/batch
thì thêm --new. --download vẫn có thể bật khi resume.

**98% CIFAR-10 / 95% CIFAR-100 là mục tiêu, chưa phải kết quả đo được và không
được cam kết.** Đề Final exam.docx yêu cầu <=6M tham số và <1 GFLOP. Hai biến thể
mới đều đáp ứng giới hạn này, nhưng CIFAR-100 95% từ scratch với model nhỏ là
mục tiêu rất khó. Không thay số đo, không lấy validation thay test để báo đạt.

## Dừng và chạy tiếp

- Checkpoint chỉ ghi sau khi hoàn thành cả train và validation của mỗi epoch.
  Không ghi lúc khởi động, giữa epoch, khi dừng hoặc sau test.
- Nhấn Ctrl+C một lần: trainer hoàn tất batch đang xử lý rồi dừng, giữ nguyên
  checkpoint của epoch hoàn tất gần nhất.
- Chạy lại python train.py: bắt đầu từ epoch kế tiếp của checkpoint. Epoch đang
  train hoặc validation dở sẽ được chạy lại từ đầu.
- Nếu chưa hoàn tất epoch đầu tiên, chưa có checkpoint; chạy lại từ epoch 1.
- Mất điện/End Task/kill cũng khôi phục theo epoch hoàn tất gần nhất.
- Test bị ngắt sẽ được đánh giá lại từ đầu, không train lại epoch đã hoàn tất.
  last.pt lỗi/mất sẽ thử previous.pt của epoch trước.
- Không xóa lock bằng tay: lock do hệ điều hành quản lý và tự nhả khi process chết.
- Không sửa train.py trong lúc một phiên đang chạy; hash code chặn resume sai
  kiến trúc/logic. Sau khi đổi code, bắt đầu --new.

## Các lệnh cần dùng

Train lại độc lập, vẫn giữ phiên cũ:

    python train.py --new

Chỉ CIFAR-100:

    python train.py --new --dataset cifar100

Biến thể localpath để đối chứng:

    python train.py --new --model localpath

Adam hoặc LR/batch khác; nếu hết VRAM, bắt đầu phiên mới batch nhỏ hơn:

    python train.py --new --optimizer adam --lr 0.001
    python train.py --new --batch-size 64 --workers 2

Ma trận theo đề: cả hai dataset, SGD 0.05/0.1/0.15, Adam 0.0003/0.001/0.003,
12 lần train độc lập (rất lâu):

    python train.py --new --matrix

Resume một phiên cụ thể; truyền thư mục phiên, không phải file .pt:

    python train.py --resume outputs/rebuilt/TEN_THU_MUC_PHIEN

Kiểm tra kiến trúc, không train, không tải dữ liệu:

    python train.py --profile

Smoke test 2 epochs trên dữ liệu giả, kết quả KHÔNG phải benchmark CIFAR:

    python train.py --smoke --new

Smoke lưu ở outputs/rebuilt_smoke, tách khỏi phiên train thật.
Nếu có vấn đề workers trên Windows, dùng --new --workers 0.
--download chỉ dùng khi dữ liệu bị thiếu. --fast cho phép thuật toán không
deterministic để ưu tiên tốc độ; mặc định ưu tiên reproducibility.

    python train.py --help

## Kết quả lưu ở đâu?

    outputs/rebuilt/<timestamp_id>/
        session.json                 recipe/environment và trạng thái toàn phiên
        session.log                  log toàn queue, append khi resume
        results.json                 tóm tắt các run đã hoàn tất
        cifar10_dualpath_sgd_lr0.1/
            config.json              hyperparameters và profile model
            split.json               chỉ số train/validation, SHA256
            terminal.log             log riêng run, append khi resume
            metrics.csv              mỗi epoch một dòng, không trùng khi resume
            last.pt                  trạng thái sau epoch hoàn tất gần nhất
            previous.pt              checkpoint của epoch hoàn tất trước đó
            best.pt                  weights chọn bằng validation cuối epoch
            result.json              test metrics, target_met, profile và provenance
            confusion_matrix.json    ma trận nhầm lẫn test
        cifar100_dualpath_sgd_lr0.1/
            ...

Mỗi epoch chỉ có một dòng: dataset/model/optimizer, epoch, loss, mixed train
accuracy, validation raw/EMA, best accuracy, LR, thời gian, peak VRAM, AMP skips
và xác nhận checkpoint=last.pt. Không có log tiến độ batch hoặc dòng save riêng.
Khi dừng có thông báo trạng thái; lỗi vẫn được hiển thị để xử lý.
Train accuracy có MixUp/CutMix nên không so trực tiếp với accuracy ảnh sạch.
Train/validation chia 45k/5k từ training set; test 10k chỉ chạy sau khi train xong.
Best checkpoint được chọn bằng validation, không dùng test để chọn model.
Kết quả có top-1, top-5, macro F1 và recall từng class.

## Kiến trúc và kiểm thử

- CompactMixNet dualpath: 5.248M/5.271M params (10/100 classes), ~0.8424 GFLOP.
- CompactMixNet localpath: 5.209M/5.232M params, ~0.8329 GFLOP.
- Cả hai là tổ hợp residual + spatial branches + grouped mixing + channel gates;
  không tuyên bố novelty. Cần xác nhận điều kiện không trùng kiến trúc với giảng viên.
- docs/ARCHITECTURE.md giải thích cấu trúc, recipe, FLOP convention và resume.
- tests/test_training.py kiểm tra budgets, splits, sampler, RNG, atomic save,
  corrupt-checkpoint fallback, locks, SGD/Adam resume theo epoch so với chạy
  liền mạch, và không tạo checkpoint cho epoch đang dở.

    python -B -m pytest -q

Kiểm thử resume dùng mạng nhỏ để test state machine nhanh; smoke test CLI dùng
chính kiến trúc đầy đủ. Smoke/pytest không chứng minh accuracy 98%/95%.

## Dọn project

Các kiến trúc/model code cũ, scripts train/eval/queue/monitor/profile cũ, configs,
tests và tài liệu cũ đã bỏ. BT1 chứa code model mẫu cũ và plan_action chứa kế
hoạch pipeline cũ cũng được bỏ. Giữ nguyên datasets, kết quả/checkpoints cũ trong
outputs, Final exam.docx, KnowLedge và lịch sử Git. Code mới không resume
checkpoint của kiến trúc cũ; kết quả mới chỉ nằm trong outputs/rebuilt.

Kiểm kê môi trường Python và những phần dọn hệ thống còn bị chặn được ghi rõ
trong docs/ENVIRONMENT.md. VS Code hiện dùng Python 3.13 Microsoft Store;
free-claude-code và Python 3.14 đã được gỡ theo xác nhận của bạn.
