# Train CIFAR-10 và CIFAR-100 cho bài thi cuối kỳ

Chạy tại `D:\DeepLearning\Finnal`, dùng đúng **Python 3.13 đang chọn trong
VS Code**. PyTorch 2.7.1+cu118/torchvision hiện tại đã chạy được CUDA trên RTX
4060 Laptop 8 GB. Chương trình chỉ dùng interpreter này và thư viện cài trực tiếp
trong môi trường của nó; không tạo/tải môi trường Python khác hoặc nạp `.deps`.
Luật áp dụng cho mọi dự án nằm tại `C:\Users\anhna\.codex\AGENTS.md`.

```powershell
python -u train.py
```

Đây là một file Python duy nhất làm entry point. Chương trình chạy lần lượt
CIFAR-10 và CIFAR-100 từ đầu, mỗi bộ 600 epoch chọn model bằng validation. Khi
cả hai hoàn tất, nó đóng băng cấu hình/loại weights/epoch tốt nhất của mỗi bộ,
train lại từ đầu trên toàn bộ 50.000 ảnh theo số epoch đã đóng băng, rồi đánh
giá test chính thức một lần. Đây là quy trình train dài do bạn chủ động bắt đầu;
setup chỉ chạy kiểm tra ngắn. Thời gian thực tế được ước lượng trong log sau
các batch/epoch, không có cam kết số giờ hoặc accuracy trước khi train.

## Model L và giới hạn đề

`Final exam.docx` yêu cầu model tự thiết kế, **không quá 6M tham số**, **dưới
1G FLOPs**, so sánh SGD/Adam và nhiều learning rate, và báo cáo PDF.

| Đầu ra | Tham số trainable | FLOPs / ảnh | MACs / ảnh |
|---|---:|---:|---:|
| CIFAR-10 | 5.248.172 | 0,842407 G | 0,409493 G |
| CIFAR-100 | 5.271.302 | 0,842453 G | 0,409516 G |

`CompactMix-L` có stem 3×3 stride 1; stage 32×32/16×16/8×8, channels
40/112/256, depths 2/3/8. Block residual mở rộng channels, tách hai nhánh
depthwise 3×3/5×5, nối lại, trộn bằng convolution 3×3 có 8 groups, SE channel
attention rồi projection. Mạng tái sử dụng các ý tưởng đã công bố trong một
cấu trúc tùy chỉnh; không khẳng định phát minh mới hoặc bảo đảm khác mọi nhóm.

FLOPs là ước lượng phân tích với **2 FLOPs/MAC**, cộng normalization,
activation, pooling, residual add và channel gate. Nó không phải latency đo
trên GPU. `--profile` kiểm tra và từ chối vượt budget trước khi train.
Các model tham khảo timm có thêm ước lượng attention và biên 10% operations;
không dùng số đó để giả định mọi cấu hình timm đều được profile đầy đủ.

```powershell
python train.py --profile
```

Mục tiêu nghiên cứu là CIFAR-10 ≥98%, CIFAR-100 khoảng ≥90%. **Chưa có kết quả
train đầy đủ chứng minh các mốc này.** Smoke/subset accuracy không phải kết quả
benchmark và bị loại khỏi bảng xếp hạng.

## Theo dõi và dừng/chạy tiếp

Log console và `training.log` ghi dataset, cấu hình, số tham số/FLOPs, batch,
loss, training accuracy có Mixup/CutMix, validation raw/EMA, best và epoch của
best, learning rate, tốc độ, ETA, VRAM, AMP skips và trạng thái lưu checkpoint.

```text
TRAIN epoch=20/600 batch=100/352 loss=... mixed_acc=... lr=... epoch_ETA=...
EPOCH 20/600 | loss=... mixed_acc=... val_raw=... val_ema=... best=... @19
             | time=... total_ETA=... VRAM=... AMP_skips=0 | saved last.pt
RESUME completed_epoch=20 next_epoch=21/600 best_val=... @19
```

Mỗi **thí nghiệm** chỉ giữ hai checkpoint:

- `last.pt`: ghi đè sau mỗi epoch đã hoàn tất train và validation; chứa model,
  EMA, optimizer, AMP scaler, epoch/global step, RNG, history và cấu hình.
- `best.pt`: ghi đè khi validation top-1 cao hơn best cũ; hòa giữ bản trước.

Không có `previous.pt`, checkpoint theo batch hay `epoch_XXX.pt`. Mỗi dataset,
trial và pha retrain là một thí nghiệm riêng, có một cặp best/last riêng. Trong
pha retrain cuối, không chọn bằng validation/test nữa: best là weights ở epoch
cuối đã đóng băng (raw hoặc EMA theo lựa chọn validation), còn last vẫn resume
được. Checkpoint nằm tại `checkpoints/<tên-thí-nghiệm>/`.

Nhấn **Ctrl+C một lần** để hoàn tất epoch đang chạy rồi lưu và dừng. Nhấn lần
hai để dừng ngay. Khi tắt cưỡng bức/mất điện, chỉ phần sau epoch hoàn tất gần
nhất bị chạy lại. Nếu chưa hoàn tất epoch đầu, lần chạy tiếp bắt đầu từ đầu.
Chạy lại **đúng lệnh cũ** để tự tiếp tục; thí nghiệm đã xong được bỏ qua. Nếu đã
xong train selection nhưng dừng ở retrain/test, cùng lệnh vẫn tiếp tục pha đó.

Ghi file tạm, flush/fsync rồi thay thế atomically. Last còn giữ best weights để
khôi phục khi best thiếu/hỏng. Last hỏng sẽ báo lỗi rõ ràng, không tự bắt đầu
lại hoặc âm thầm thay checkpoint. Không nạp checkpoint từ nguồn không tin cậy.

Thay epochs/LR/model/seed/recipe/device hoặc source sẽ bị từ chối resume. Muốn
một thí nghiệm khác, đặt `--run-name <tên-mới>`; không xóa checkpoint cũ để lách
kiểm tra. Muốn giảm VRAM, chọn batch nhỏ **trước** khi bắt đầu thí nghiệm.

## Recipe mặc định

SGD LR 0.1, momentum 0.9, Nesterov, weight decay 0.0005; batch 128; warmup
min(5,epochs/10) rồi cosine về khoảng 0,1% peak LR; label smoothing 0.05;
crop/flip/CIFAR10 AutoAugment; mixing probability 0.75, Mixup alpha 0.2 hoặc
CutMix alpha 1.0 với xác suất bằng nhau; tắt mixing ở min(30,epochs//10) epoch
cuối; gradient clipping 5; EMA capped 0.999; float16 AMP trên CUDA. Bias/norm
không weight decay. Adam dùng betas (0.9,0.999), epsilon 1e-8, mặc định LR
0.001/weight decay 0.0001. AdamW là tùy chọn riêng.

Split có 45.000 train/5.000 validation, stratified seed 2026; initialization seed
42. Validation/test chỉ normalize; augmentation của từng ảnh có seed theo
epoch/index để replay ổn định cả khi workers khởi động lại. Mặc định deterministic;
`--fast` chấp nhận đổi sang thuật toán nhanh có thể không tái lập bitwise.

Mặc định dùng dữ liệu CIFAR đã có ở `datasets/`. Nếu thiếu, torchvision tải ngay
vào đó. Có thể đổi sang thư mục trong project bằng `--data-root data`.

## Các lệnh dùng cùng file train.py

Chỉ train/validate một dataset (chưa mở test):

```powershell
python -u train.py --dataset cifar10 --selection-only
python -u train.py --dataset cifar100 --selection-only
```

Chạy đủ train/retrain/test một dataset: bỏ `--selection-only`. Lệnh không tham
số chạy cả hai. Các baseline có thể chạy trực tiếp hoặc qua YAML:

```powershell
python -u train.py --dataset cifar10 --config configs/baseline_wrn.yaml --run-name wrn --selection-only
python -u train.py --dataset cifar100 --config configs/baseline_convnext.yaml --run-name convnext --selection-only
python -u train.py --dataset cifar10 --model preactresnet --run-name preact --selection-only
```

Family sẵn có: compactmix, wrn, preactresnet, convnext, resnet, resnext,
convnextv2, efficientnet, regnet, mobilenet, vit, deit, swin, hybrid. Tất cả
from-scratch. Không phải mọi cấu hình tham khảo đều đạt giới hạn; chương trình
kiểm tra trước train. L là model CompactMix tùy chỉnh, không phải baseline timm.

So sánh đề bài: SGD LR **0.05/0.1/0.15**, Adam LR **0.0003/0.001/0.003**, tổng
12 lượt độc lập trên hai dataset. Chỉ chọn winner bằng validation. Sau cả queue,
retrain và test một winner mỗi dataset; không tối ưu bằng 12 test scores.

```powershell
python -u train.py --matrix
```

Optuna SQLite nằm tại `experiments/optuna.db`. Screening 30 epoch, có pruning:

```powershell
python -u train.py --search --dataset cifar10 --config configs/cifar10_search.yaml --trials 20 --study-name cifar10-screen30
python -u train.py --search --dataset cifar100 --config configs/cifar100_search.yaml --trials 20 --study-name cifar100-screen30
```

Chạy lại đúng lệnh để tiếp tục trial đang dở và study. `--trials 20` là **tổng
số trial đích**, không thêm 20 trial mỗi lần chạy. Đổi budget thành 100/600 epoch
phải dùng tên study mới. `--search-families compactmix,convnextv2,efficientnet`
đổi family được tìm; `--multi-objective` tối đa validation/min parameters,
không có epoch pruning do giới hạn Optuna multi-objective. Smoke dùng cấu hình
nhỏ cố định cho widths/depths; kết quả đó không phải tìm kiếm kiến trúc thật.

Đóng băng một winner đã chọn bằng validation rồi retrain/test:

```powershell
python -u train.py --finalize checkpoints/default_cifar10_compactmix_sgd_lr0.1_seed42
python -u train.py --finalize checkpoints/default_cifar100_compactmix_sgd_lr0.1_seed42
```

Tên winner từ Optuna nằm trong trial metadata (`checkpoint_path`). Đánh giá
frozen final đã train xong; kết quả test được cache và kiểm tra hash checkpoint:

```powershell
python train.py --evaluate checkpoints/default_cifar10_compactmix_sgd_lr0.1_seed42_final
python train.py --evaluate checkpoints/default_cifar100_compactmix_sgd_lr0.1_seed42_final
```

## Kết quả và báo cáo

`metrics.csv`, `config.json`, `split.json`, `metadata.json`, `training.log`,
TensorBoard và `result.json` nằm trong từng thí nghiệm. Metadata ghi Git commit,
source SHA256, split SHA256, Python/PyTorch/CUDA/GPU/seed.

CSV tổng: `experiments/results/results.csv`; leaderboard:
`reports/generated/leaderboard.md`; PDF:
`reports/generated/final_report.pdf`. Bảng giữ trống test chưa đánh giá.
PDF hiện tại là bản mô tả kiến trúc/protocol; chỉ trở thành báo cáo kết quả đầy
đủ sau khi bạn chạy train và so sánh optimizer/LR. Nó có cấu hình thật, kết quả,
thời gian và metadata của các lượt đã xong; không tự điền target thành accuracy.

```powershell
python train.py --report
python .agents/skills/cifar-sota-architect/scripts/inspect_results.py
python .agents/skills/cifar-sota-architect/scripts/summarize_study.py
```

## Setup nội bộ và kiểm tra

Skill nằm ở `.agents/skills/cifar-sota-architect/`; quy tắc tại `AGENTS.md`;
Optuna MCP chỉ tại `.codex/config.toml`, dùng Python hiện tại và launcher nội bộ.
`codex mcp list` đã nhìn thấy server; STDIO initialize/list tools đã được kiểm tra.
Sau khi đổi vị trí repository/interpreter, chạy `python scripts/configure_mcp.py`
để tạo lại đường dẫn nội bộ. Luật Python global đã được cập nhật theo yêu cầu;
MCP vẫn chỉ được đăng ký trong dự án, không sửa global trust/Codex config.toml.

```powershell
python scripts/setup_environment.py
python -m pytest -q --basetemp experiments/test-temp
python -u train.py --smoke --dataset both --device cpu --no-tensorboard
```

Script setup dùng chính xác
`C:\Users\anhna\AppData\Local\Microsoft\WindowsApps\python3.13.exe -m pip`.
Thư viện thiếu được cài vào user-site thông thường của cùng môi trường này;
script không tạo `.venv`, `.deps`, tải Python khác hoặc dùng pip `--target`.
Nó giữ nguyên phiên bản CUDA torch/torchvision/NumPy hiện tại qua constraints.
`requirements-lock.txt` ghi phiên bản các dependency trực tiếp của dự án.
Entry point sẽ từ chối chạy nếu interpreter hoặc prefix thuộc môi trường khác.

Xem `docs/PYTHON_POLICY.md`, `docs/ENVIRONMENT.md` và
`reports/generated/shared-python-audit.json`. Audit xác nhận thư viện nạp từ
Python 3.13 đã chọn, không có package cũ bị đổi/xóa. `.venv` không còn tồn tại.
Thao tác xóa bản thư viện `.deps` cũ bị kiểm duyệt tự động từ chối; không entry
point nào nạp thư mục đó, nhưng phần xóa trên đĩa chưa hoàn tất. Các báo cáo
`isolation-*.json` trước đây chỉ mô tả setup cũ, không phải chính sách hiện tại.

Thử nghiệm đầu tiên đề xuất: `python -u train.py` với model L và SGD mặc định.
Nếu muốn kiểm tra learning curve rẻ trước full budget, chạy
`python -u train.py --epochs 30 --run-name screen30 --selection-only`; sau đó
chọn kế hoạch 600 epoch ở một run khác, không đổi budget ngay trong checkpoint cũ.
