# CIFAR lightweight final project

Kế hoạch chi tiết: [docs/FINAL_PLAN.md](docs/FINAL_PLAN.md). Mô hình hiện là **ứng viên**, chưa được xác nhận tính riêng biệt bởi giảng viên và chưa có kết quả final accuracy.

## Môi trường
Sử dụng Python/PyTorch đang có trên máy; không tạo môi trường mới. Đã xác nhận Python 3.13.14, torch 2.7.1+cu118, torchvision 0.22.1+cu118, NumPy 2.2.6, pytest 9.1.1. Lệnh dưới đây chạy từ `D:\DeepLearning\Finnal`. Đường dẫn data/output trong config tương đối với project root, không phụ thuộc cwd.

## Kiểm tra và đo tài nguyên
```powershell
python -m pytest -q
python profile_model.py --classes 100 --variant multiscale --device cuda --benchmark
python profile_model.py --classes 10 --variant baseline
python benchmark_training.py --device cuda --optimizer sgd --output outputs/synthetic_sgd.json
python benchmark_training.py --device cuda --optimizer adam --output outputs/synthetic_adam.json
```
Profiler report 2 FLOPs/MAC cho conv/linear, BN unfused 4 ops/element, activation/add/mul/pool/gate. Sigmoid cost là convention (exp tính một op), không tương đương latency phần cứng. Không bỏ qua custom add/multiply. Chỉ inference ở batch=1 được so với giới hạn 1G; benchmark batch128 là đo tốc độ riêng.

## Chạy thử dữ liệu thật (không phải kết quả cuối)
Để **chỉ tải/kiểm tra**, không huấn luyện:
```powershell
python download_data.py --dataset cifar10
python download_data.py --dataset cifar100
```
Downloader thử HTTP Range để tiếp tục archive dở, dùng `.part` khi chưa đúng MD5. Nếu server không hỗ trợ Range thì tải lại từ đầu và ghi rõ trong log. Có timeout và tối đa 5 attempts. Chỉ giải nén sau official MD5; kiểm tra checksums bằng torchvision và cấu trúc 50k train/10k test. Không tạo model, không dự đoán hay tính accuracy. Status nằm ở `D:\DeepLearning\Finnal\outputs\downloads\cifar10.json` và `cifar100.json`; `phase: completed` và `verified: true` mới là hoàn tất. Lock ngăn tải trùng cùng dataset; sau kill đột ngột chỉ xóa lock cũ khi đã xác nhận PID dừng.

Sau khi tải xong, lệnh dưới đây **có huấn luyện ngắn**, không tự chạy cùng downloader:
```powershell
python train.py --config configs/smoke_cifar10.json --download
python train.py --config configs/smoke_cifar100.json --download
```
Smoke configs chỉ 10 train batches và 2 validation batches/epoch. Không dùng chúng để so accuracy hoặc evaluate official test. Official data có thể được torchvision tải cả file test trong archive; train loop không truy cập ảnh/nhãn test.

Nếu server CIFAR tải chậm, có thể tải Python archives chính thức từ https://www.cs.toronto.edu/~kriz/cifar.html vào `D:\DeepLearning\Finnal\datasets`, rồi chạy lại với `--download` để torchvision kiểm tra MD5 và giải nén. File chưa tải xong không dùng được. Không thay bằng dataset không rõ nguồn. Synthetic benchmark chỉ kiểm tra training/AMP và throughput, không bao gồm augmentation/I/O thật.

## Huấn luyện development
```powershell
python train.py --config configs/cifar100_sgd.json --download
python train.py --config configs/cifar10_sgd.json --download
```
Hai configs là điểm xuất phát 150 epochs, không phải recipe đã tối ưu. Mỗi output directory phải mới/rỗng. Validation 5k stratified; train 45k. Metrics là %, loss sample-weighted. Train accuracy là lambda-weighted nếu CutMix bật. Validation luôn float32, single-view.

Tạo 12 configs khảo sát **không tự chạy**:
```powershell
python prepare_experiments.py
python train.py --config configs/grid/cifar100_adam_0.001.json
```
Không chạy tất cả trước khi chốt kiến trúc/budget. Grid generator từ chối overwrite configs. Để bật CutMix, chỉnh `cutmix_alpha` (ví dụ 1.0) và `cutmix_probability` (ví dụ 0.5) trong một config/output mới, rồi kiểm chứng validation. `label_smoothing` mặc định 0.

## Resume
```powershell
python train.py --config configs/cifar100_sgd.json --resume outputs/cifar100_sgd_0.1/last.pt
```
Khôi phục model, optimizer, AMP scaler, RNG, loader RNG và per-step LR từ epoch đã hoàn thành. Không thay hyperparameters/epochs giữa run; schedule budget cố định. Resume ở ranh giới epoch, không mid-epoch. `workers` có thể đổi nhưng để tái lập mạnh hơn giữ cùng device/workers/library versions. Checkpoint chỉ load từ nguồn đáng tin vì chứa Python state.

Output gồm `config.json`, `environment.json`, `split.json`, `profile.json`, `metrics.csv`, `summary.json`, `last.pt`, `best.pt`. `best.pt` chọn bằng validation, không test. Với full_train không có best checkpoint; dùng last epoch.

## Train cuối và official test
Sau khóa architecture/recipe/epoch budget, tạo config mới với `full_train: true`, `max_train_batches: null`, `max_val_batches: null`, output mới và epochs đã chọn. Train từ đầu trên đủ 50k. Không dùng resume để chuyển development sang full training.

Chỉ khi mọi lựa chọn đã khóa:
```powershell
python evaluate.py --checkpoint outputs/final_cifar100/last.pt --confirm-final-test --output outputs/final_cifar100/test.json
```
Không có TTA/ensemble. Debug checkpoints bị chặn. Full-data run phải hoàn thành epoch budget. Test results không được overwrite. Công cụ có thể evaluate frozen development checkpoints để lập bảng, nhưng người dùng phải khóa selection protocol trước và không tuning tiếp sau test.

## Điều cần xác nhận
Quy ước FLOPs của giảng viên, architecture originality/non-duplication, metric ranking và kỹ thuật được phép. Không cam kết accuracy/thứ hạng. Báo cáo/vấn đáp là phần bắt buộc, không chỉ code.

`datasets/`, `outputs/`, `plan_action/` không đưa vào Git. Project ban đầu chưa có Git repository; không tự init hoặc tạo remote.

## So sánh hai mô hình và báo cáo đầy đủ
Protocol: `D:\DeepLearning\Finnal\docs\TWO_MODEL_STUDY.md`. Dependencies đã có: sklearn/scipy/matplotlib/pandas/reportlab; không cài mới. Grid cũ 12 configs không bị thay đổi. Generator mới tạo 24 configs (150 epochs) và 6 smoke/worker configs (2 full epochs):
```powershell
python -B D:\DeepLearning\Finnal\prepare_study.py
python -B -u D:\DeepLearning\Finnal\study.py --manifest D:\DeepLearning\Finnal\configs\study\manifest.json
```
Queue sequential một GPU, fail-stop, auto resume checkpoint epoch-boundary, lock ngăn queue trùng. Đo real smoke/worker trước grid; chỉ tiếp tục nếu sample counts/loss/AMP hợp lệ, disk đủ và budget+25% reserve dưới168 giờ cho grid. AMP dynamic scaling có thể skip1–2 steps calibration; ghi số skip/scale/rate, dừng nếu >1% steps trong epoch hoặc loss/weights nonfinite. Giới hạn là vận hành, không thay đổi số epochs giữa run. Workers chọn theo median measured wall time và khóa chung cho cả24 runs. Không chạy lại queue khi process/lock còn sống.

Theo dõi `D:\DeepLearning\Finnal\outputs\study_two_models\status.json`, `budget.json`, `logs\*.log`, `runs\*\progress.json`. Tạo file `D:\DeepLearning\Finnal\outputs\study_two_models\STOP_AFTER_RUN` để dừng ở giữa các runs (không kill giữa checkpoint); xóa stop request sau khi queue đã dừng nếu muốn tiếp tục. Nếu crash để stale lock, kiểm tra PID queue và child đã chết trước khi xóa lock. Không tự bỏ lock của process còn sống.

Full metrics validation/test dùng chung entrypoint mới:
```powershell
python -B D:\DeepLearning\Finnal\evaluate_full.py --checkpoint D:\DeepLearning\Finnal\outputs\study_two_models\runs\cifar10_baseline_sgd_0.1_s42\best.pt --split validation --output-dir D:\DeepLearning\Finnal\outputs\manual_validation
python -B D:\DeepLearning\Finnal\reporting.py --study-root D:\DeepLearning\Finnal\outputs\study_two_models
```
Export `predictions.npz` (logits/probs/labels/indices/class_names), predictions/errors/per-class/confusion/calibration CSV, JSON curves và metrics, model-only weights, PNG/PDF figures và per-run evaluation_report.pdf. `--split train` dùng deterministic hard-label views của train indices để so generalization, không augmentation/mixed metrics. Official test vẫn yêu cầu `--split test --confirm-final-test`, không được queue tự gọi. Thư mục export phải mới/rỗng.

Aggregate provisional: `D:\DeepLearning\Finnal\outputs\study_two_models\reports\comparison_report.pdf` + all_evaluations CSV/JSON và matched-architecture paired-bootstrap JSON. Chỉ các exports có complete marker mới được gộp, không gồm smoke. Seed mean/std chỉ khi có final test seeds, không chọn best-test seed. PDF final `D:\DeepLearning\Finnal\report\comparison_report.pdf` sẽ tạo ở checkpoint cuối sau freeze và test; chưa coi provisional PDF là bài nộp hoàn tất.

Queue chỉ chạy smoke và base-grid. Recipe/300epochs/full-data seeds ở các giai đoạn sau dựa strictly trên validation. Source fingerprint chống resume khác code; giữ nguyên code trong lúc queue chạy. Không cam kết budget tối đa9600epochs khi chưa đo CIFAR thật; latency synthetic forward được ghi tách khỏi real training throughput.