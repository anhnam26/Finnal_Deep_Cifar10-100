"""Build the self-contained, cell-by-cell CompactMix presentation notebook."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import textwrap

import nbformat


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "report/compactmix_manual_train.py"
OUTPUT = ROOT / "report/CompactMix_CIFAR10_CIFAR100_BAO_CAO.ipynb"


def build_notebook():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = {node.name: node for node in tree.body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    assignments = {target.id: node for node in tree.body if isinstance(node, ast.Assign)
                   for target in node.targets if isinstance(target, ast.Name)}
    lines = source.splitlines()

    def extract(name):
        node = nodes.get(name, assignments.get(name))
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        return "\n".join(lines[start-1:node.end_lineno])

    cells = []

    def md(text):
        cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip()))

    def code(text, tag, *, remember=False):
        content = textwrap.dedent(text).strip()
        if remember:
            content = f'_remember_cell("{tag}")\n\n' + content
        cell = nbformat.v4.new_code_cell(content)
        cell.id = tag
        cell.metadata["tags"] = [tag]
        cells.append(cell)

    def definitions(names, tag):
        code("\n\n".join(extract(name) for name in names), tag, remember=True)

    md(r"""
    # CompactMix-L: chạy code và báo cáo CIFAR-10 / CIFAR-100

    **Sinh viên:** ................................ · **Ngày trình bày:** 16/10/2026

    Bố cục theo notebook [Training a Classifier của PyTorch](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/cifar10_tutorial.ipynb): nạp dữ liệu → định nghĩa mạng/forward → loss/optimizer → train → lưu/nạp → đánh giá từng lớp. Kiến trúc và recipe ở đây là **CompactMix-L của bài làm**, không dùng mạng Net nhỏ của tutorial.

    Notebook chứa đầy đủ code model và train, **không import `src/` hoặc file Python của dự án**. Mở trong VS Code/Jupyter và chọn đúng kernel Microsoft Store Python 3.13 đang dùng. Link Colab là mẫu bố cục; notebook này giữ chính sách môi trường Python hiện tại của bạn.

    ### Cách dùng khi trình bày

    1. Chạy từ trên xuống; xem ảnh, code từng block, bảng shape và kiểm tra ngân sách.
    2. Mặc định `MODE="demo"`: dùng **512 ảnh train + 256 validation mỗi dataset**, chạy **một epoch mới mỗi lần gọi**; tổng lịch đặt trước là 2 epoch. Chạy lại cell train để minh họa resume. Accuracy demo không phải điểm số trong PDF.
    3. Các cell cuối đọc và kiểm chứng **kết quả đã lưu của 600 epoch/45k**, vẽ lại learning curves, confusion matrix và chỉ số từng lớp. Không cần train lại 600 epoch để trình bày các kết quả đã đo.
    4. `MODE="report_only"` bỏ qua train mới. `MODE="full"` dùng đầy đủ 45k/5k, lịch 600 epoch; chỉ chọn khi thực sự muốn train dài.

    Checkpoint lượt mới nằm riêng trong `experiments/notebook_presentation/`. Mỗi run chỉ có `best.pt` và `last.pt`; log train chỉ ghi sau epoch hoàn chỉnh. Không ghi đè checkpoint nguồn của báo cáo.
    """)
    md(r"""
    ## 1. Đề bài và phạm vi kết quả

    Theo `Final exam.docx`: model L có **≤6.000.000 tham số học được**, **<1.000.000.000 FLOPs**, train/test CIFAR-10 và CIFAR-100, so sánh SGD/Adam và nhiều LR, mô tả learning settings trong PDF.

    | Chỉ số trong PDF, snapshot 14/10/2026 | CIFAR-10 | CIFAR-100 |
    |---|---:|---:|
    | Epoch selection đã hoàn thành | 600 | 600 |
    | Best epoch / weights | 567 / EMA | 399 / EMA |
    | Best validation (5k) | 97,56% | 83,36% |
    | Test best train 45k (10k ảnh) | 97,18% | 82,98% |
    | Tham số | 5.248.172 | 5.271.302 |
    | FLOPs/ảnh theo profiler bài làm | 842.406.944 | 842.453.114 |

    Đây là **kết quả cũ đã đo của best train 45k**, không phải kết quả của lượt demo hay retrain 50k. Snapshot PDF chưa có final-test của retrain 50k; so sánh full-data Adam/nhiều LR, multi-seed và ablation còn thiếu. Các cell đọc evidence bên dưới sẽ kiểm chứng bảng này bằng hash checkpoint và số đếm confusion matrix.
    """)
    md("## 2. Import, kiểm tra đúng Python/CUDA và xác định thư mục\n\nCell này chỉ dùng các thư viện đã có trong Python được chọn; không tạo môi trường hoặc cài thư viện vào nơi khác.")
    bootstrap = r'''
import copy
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time

__ENSURE_PYTHON__

ensure_python()
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from IPython import get_ipython
from IPython.display import display, Markdown

get_ipython().run_line_magic("matplotlib", "inline")
torch.set_num_threads(min(4, os.cpu_count() or 1))
pd.set_option("display.max_columns", 18)
pd.set_option("display.max_colwidth", 65)
plt.rcParams.update({"figure.dpi": 115, "axes.spines.top": False, "axes.spines.right": False})
cwd = Path.cwd().resolve()
ROOT = next((p for p in (cwd,*cwd.parents) if (p/"datasets").is_dir()),
            cwd.parent if cwd.name == "report" else cwd)
REPORT_DIR = ROOT/"report"
DATA_ROOT = ROOT/"datasets"
OUTPUT_ROOT = ROOT/"experiments/notebook_presentation"
CHECKPOINT_VERSION = 1
REQUEST_STOP = False
NOTEBOOK_TRAINING_CELLS = {}

def _remember_cell(name):
    # Lưu source cell THỰC SỰ đã chạy, kể cả sửa trong IDE chưa Save.
    # Chạy lại cell định nghĩa sẽ thay source của chính cell đó.
    shell = get_ipython()
    source = shell.history_manager.input_hist_raw[-1]
    if not source.strip():
        raise RuntimeError("Không đọc được source cell đang chạy; chạy notebook bằng IPython/Jupyter.")
    NOTEBOOK_TRAINING_CELLS[name] = source.replace("\r\n", "\n")

_remember_cell("bootstrap")
display(pd.DataFrame([{
    "Python": platform.python_version(), "Executable": sys.executable,
    "Base environment": sys.prefix == sys.base_prefix,
    "PyTorch": torch.__version__, "Torchvision": torchvision.__version__,
    "CUDA build": torch.version.cuda,
    "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "torch location": torch.__file__, "numpy location": np.__file__,
}]))
print("Dữ liệu:", DATA_ROOT, "\nLượt chạy mới:", OUTPUT_ROOT)
'''.replace("__ENSURE_PYTHON__", extract("ensure_python"))
    code(bootstrap, "bootstrap")
    md(r"""
    ## 3. Cấu hình chạy và recipe

    `DEMO_STOP_AFTER=1` chỉ tạm dừng sau một epoch mới; **không đổi lịch tổng `DEMO_TOTAL_EPOCHS=2` giữa hai lần resume**. Chạy lại cell train để tiếp tục. Nếu sửa model/recipe/source, đổi `RUN_NAME` để tạo thí nghiệm mới.

    Notebook dùng `workers=0` để các class Dataset định nghĩa trong cell chạy ổn định với Jupyter trên Windows. PDF đã train bằng `workers=2`; cách đặt worker không làm thay đổi kiến trúc. Các giá trị báo cáo cũ vẫn đọc từ config đã lưu.
    """)
    code(extract("TRAIN_SETTINGS") + "\n\n" + extract("STATS") + "\n\n" + r'''
# Có thể sửa các biến này ngay trong cell, rồi chạy lại.
MODE = globals().get("NB_MODE", "demo")           # demo | report_only | full
DATASETS = ("cifar10", "cifar100")
RUN_NAME = globals().get("NB_RUN_NAME", "bao_cao_20261016")
DEMO_TOTAL_EPOCHS = 2
DEMO_STOP_AFTER = 1
DEMO_TRAIN_IMAGES = 512
DEMO_VAL_IMAGES = 256
FULL_EPOCHS = 600
DOWNLOAD = False                                # dùng datasets/ hiện có
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SYNTHETIC_CHECK = bool(globals().get("NB_SMOKE", False))  # chỉ dành cho smoke kỹ thuật
RUN_MATRIX = False                              # bật riêng khi muốn train nhiều LR
RUN_FINAL = False                               # bật khi selection full đã hoàn tất
RECHECK_ARCHIVED_TEST = False                    # mặc định đọc điểm test đã lưu
TRAIN_SETTINGS.update(workers=0, tensorboard=False)
if MODE not in {"demo", "report_only", "full"}:
    raise ValueError("MODE phải là demo, report_only hoặc full")
if not re.fullmatch(r"[A-Za-z0-9_.-]+", RUN_NAME):
    raise ValueError("RUN_NAME chỉ dùng chữ, số, dấu chấm/gạch dưới/gạch ngang")
display(pd.DataFrame([TRAIN_SETTINGS]).T.rename(columns={0:"Recipe mặc định"}))
print("Chế độ:", MODE, "| datasets:", DATASETS, "| device:", DEVICE)
''', "settings")
    md(r"""
    ## 4. Dữ liệu, stratified split, augmentation và xem ảnh

    Mỗi dataset có 50k train, 10k test, ảnh RGB 32×32. Tách stratified 45k/5k từ official train bằng seed 2026; lưu indices và SHA256. **Phần lựa chọn model chỉ mở `train=True`.** Train dùng RandomCrop/Flip/AutoAugment + Normalize; validation chỉ ToTensor/Normalize.

    Với batch size 128: `[N,3,32,32]`; N có thể nhỏ hơn 128 ở batch cuối. Demo cắt một subset xác định để chạy nhanh và được gắn `subset=True` trong checkpoint/result.
    """)
    definitions(["stratified_split", "prepare_data", "sample_rng", "image_transforms", "SeededImages", "make_loader"], "data-definitions")
    code(r'''
def notebook_configs(mode=None, optimizer=None, lr=None):
    mode = MODE if mode is None else mode
    configs = []
    for name in DATASETS:
        cfg = copy.deepcopy(TRAIN_SETTINGS)
        cfg.update(dataset=name, model={"family":"compactmix_manual"},
                   phase="selection", final_weights="ema", device=DEVICE,
                   data_root=str(DATA_ROOT.resolve()), download=DOWNLOAD,
                   smoke=SYNTHETIC_CHECK, limit_train=0, limit_val=0)
        cfg["epochs"] = FULL_EPOCHS if mode=="full" else DEMO_TOTAL_EPOCHS
        if mode!="full":
            cfg.update(limit_train=DEMO_TRAIN_IMAGES, limit_val=DEMO_VAL_IMAGES)
        if SYNTHETIC_CHECK:
            cfg.update(limit_train=0,limit_val=0,batch_size=4)
        if optimizer:
            cfg["optimizer"] = optimizer
            if optimizer in ("adam","adamw"):
                cfg.update(lr=.001,weight_decay=.0001)
        if lr is not None:
            cfg["lr"] = lr
        label=f"{RUN_NAME}_{mode}_{name}_{cfg['optimizer']}_lr{cfg['lr']:g}_seed{cfg['seed']}"
        cfg["run_dir"] = str((OUTPUT_ROOT/label).resolve())
        configs.append(cfg)
    return configs

CONFIGS = notebook_configs()
DATA_PREVIEW = {}
for cfg in CONFIGS:
    source, split = prepare_data(cfg)
    DATA_PREVIEW[cfg["dataset"]] = (source,split)
    class_names = getattr(source,"classes",[str(i) for i in range(split["classes"])])
    fig, axes = plt.subplots(2,4,figsize=(9,4))
    for ax,index in zip(axes.flat,split["training"][:8]):
        image,label = source[index]
        ax.imshow(image); ax.set_title(class_names[label]); ax.axis("off")
    fig.suptitle(cfg["dataset"]+" — ảnh TRAIN, chưa normalize")
    plt.tight_layout(); plt.show(); plt.close(fig)
    print(cfg["dataset"], "train:",len(split["training"]),"validation:",len(split["validation"]),
          "synthetic:",split["synthetic"],"subset:",split["subset"],"split SHA:",split["sha256"])
''', "data-preview")
    md(r"""
    ## 5. Định nghĩa model: shape checks, DropPath và 13 block tường minh

    Widths **40/112/256**, depths **2/3/8**, kích thước **32/16/8**. Mỗi block khai báo Conv/BN/SE và forward riêng để bạn tự sửa các thông số liên quan. Không tự suy ra hay vá kênh.

    - `chunk(2,dim=1)`: chia kênh làm hai nhánh.
    - Concat: N/H/W bằng nhau; C đầu ra là tổng C hai nhánh.
    - SE multiply: feature `[N,C,H,W]`, gate đúng `[N,C,1,1]`.
    - Residual ADD: hai tensor bằng nhau cả N/C/H/W.

    Conv: `H_out=floor((H_in+2*pH-dH*(kH-1)-1)/sH+1)`, W tương tự. BN/SiLU/Sigmoid/DropPath giữ shape. Xem thêm [công thức shape](CONG_THUC_TINH_SHAPE.md).
    """)
    definitions(["ShapeError", "concat_checked", "multiply_checked", "residual_checked", "DropPath", "ManualBlock"], "shape-helpers")
    for i in range(1,14):
        stage=1 if i<=2 else 2 if i<=5 else 3
        c,res=(40,32) if stage==1 else (112,16) if stage==2 else (256,8)
        md(f"### Block{i:02} — Stage {stage}, input/output `[N,{c},{res},{res}]`\n\nConv expand → hai nhánh depthwise 3×3/5×5 → concat → grouped Conv → SE → projection → DropPath → ADD.")
        definitions([f"Block{i:02}"], f"block{i:02}")
    md("### Ghép stem → stage → transition → head\n\n`self.stem` là Conv 3×3 đầu tiên 3→40. Trong `forward`, stem chạy trước Block01; head trả logits, không thêm softmax trước loss.")
    definitions(["CompactMixManual", "make_model"], "whole-model")
    md(r"""
    ## 6. Đếm tham số, MACs, FLOPs và kiểm tra forward/backward

    - Conv: `P=C_out*(C_in/groups)*kH*kW + bias`; `MACs=H_out*W_out*C_out*(C_in/groups)*kH*kW` cho N=1.
    - Linear: `P=F_in*F_out+F_out` khi có bias; `MACs=F_in*F_out` cho N=1.
    - BN affine: `P=2*C`; running stats là buffer.
    - Convention của PDF: **1 MAC=2 FLOPs**, cộng bias/BN/SiLU/Sigmoid/pooling/SE multiply/residual ADD; không đếm memory movement split/cat/reshape. Đây là ước lượng inference/ảnh, không phải latency hay training FLOPs.

    Cell dưới thực chạy tensor để lấy shape. Nếu sửa layer không khớp, sẽ báo đúng lớp nối sai. Model dùng để kiểm tra là model tạm, không dùng để khởi tạo train sau khi đã cập nhật BN bởi backward thử.
    """)
    definitions(["input_checks", "inspect_model"], "profile-definitions")
    code(PROFILE_CELL, "profile-results")
    md("### Sơ đồ luồng đi của model và MixBlock\n\nHình được vẽ từ các khối của notebook để dùng lúc trình bày. Mỗi Conv/BN chi tiết có trong bảng layer phía trên.")
    code(DIAGRAM_CELL, "architecture-diagram")
    md(r"""
    ## 7. Loss, optimizer, LR, Mixup/CutMix, EMA và reproducibility

    Soft-label cross entropy: `L=-mean(sum(targets*log_softmax(logits),dim=1))`. Mixup/CutMix trộn nhãn; label smoothing=0,05. Train mixed accuracy có trọng số theo nhãn trộn; không phải accuracy sạch và không đồng nhất với validation accuracy.

    SGD: LR=0,1, momentum=0,9, Nesterov; Adam/AdamW cũng có code. Warmup 5 epoch và cosine; EMA decay tối đa 0,999; gradient clipping=5; AMP chỉ bật khi CUDA. Các seed và RNG states đều lưu trong last.pt để resume.
    """)
    definitions(["make_optimizer", "learning_rate", "mix_batch", "update_ema", "seed_everything", "capture_rng", "restore_rng"], "recipe-definitions")
    code(LOSS_CELL, "loss-demo")
    md(r"""
    ## 8. Train, validation, checkpoint và resume

    `train_epoch`: zero_grad → forward → loss → backward → unscale/clip → optimizer step → EMA. Validation chạy `model.eval()` và `torch.no_grad()`; so sánh raw/EMA để chọn best theo validation.

    Mỗi epoch hoàn tất train **và** validation mới ghi best (nếu cải thiện), last, metrics và một dòng log. Last chứa model/EMA/optimizer/scaler/RNG/history; không lưu batch dở hoặc checkpoint đánh số. Ghi bằng file tạm và atomic replace; best ghi trước last. Resume kiểm tra source cell thực thi/config/runtime/split. Kernel interrupt giữa epoch giữ last của epoch hoàn chỉnh trước đó.
    """)
    definitions(["run_lock", "atomic_write", "atomic_json", "atomic_torch", "load_checkpoint", "cpu_weights", "make_logger", "close_logger", "duration", "train_epoch", "evaluate", "write_history", "validate_config"], "engine-helpers")
    code(r'''
_remember_cell("notebook-fingerprint")

def source_fingerprint():
    # Không hash outputs/execution_count; re-run không làm mất khả năng resume.
    # Hash source các cell định nghĩa đã chạy; sửa chưa Save cũng được nhận biết.
    required = {"bootstrap","data-definitions","shape-helpers","whole-model",
                "profile-definitions","recipe-definitions","engine-helpers",
                "notebook-fingerprint","training-engine","final-definitions"}
    required.update(f"block{i:02}" for i in range(1,14))
    if required - NOTEBOOK_TRAINING_CELLS.keys():
        raise RuntimeError("Chưa chạy đủ cell định nghĩa trước train: " + str(sorted(required-NOTEBOOK_TRAINING_CELLS.keys())))
    raw = json.dumps(NOTEBOOK_TRAINING_CELLS, sort_keys=True,ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

''' + extract("environment"), "notebook-fingerprint")
    trainer=extract("run_experiment").replace("compactmix-manual-onefile-v1","compactmix-notebook-v1")
    code(trainer,"training-engine",remember=True)
    definitions(["read_result", "evaluate_final", "finalize"], "final-definitions")
    md(r"""
    ### Chạy train mới / chạy tiếp

    **Cell này là điểm bắt đầu train.** Ở demo, mỗi lần chạy thêm tối đa một epoch cho mỗi dataset. Chạy lại đúng cell để tiếp tục epoch 2; không đổi EPOCHS hoặc source để minh họa resume. Chạy lại khi đủ 2 epoch không train thêm.

    Mặc định dùng ảnh CIFAR thật nhưng là subset nhỏ. Giá trị demo không thay thế kết quả full-data trong PDF. `MODE="full"` sẽ chạy selection 600 epoch; `MODE="report_only"` bỏ qua cell train. Có thể nhấn Interrupt Kernel để dừng; lần sau phát lại epoch chưa hoàn tất từ last.pt.
    """)
    code(TRAIN_CELL, "run-training")
    md("### Kiểm tra last/best và vẽ lịch sử của lượt vừa chạy\n\nBảng này chỉ đọc checkpoint sau epoch hoàn chỉnh. Best của selection dùng validation; log train không in theo batch.")
    code(NEW_CURVES_CELL, "new-run-results")
    md(r"""
    ## 9. Freeze recipe, retrain 50k, official test

    Mặc định **không chạy** phần dài này (`RUN_FINAL=False`). Sau selection full hoàn tất, freeze best epoch và loại weights raw/EMA, khởi tạo lại và train trên toàn bộ 50k. Official test chỉ được mở sau khi final train hoàn tất; SHA checkpoint gắn với test result và điểm số được cache. Final best là checkpoint của epoch cố định, không chọn bằng validation/test.

    Demo/synthetic/subset bị từ chối finalize. Code final có đầy đủ phía trên để kiểm tra; không gán test score của best 45k cho final 50k.
    """)
    code(r'''
FINAL_RESULTS = {}
if RUN_FINAL:
    if MODE!="full" or SYNTHETIC_CHECK:
        raise ValueError("Final cần MODE=full và dữ liệu đầy đủ, không dùng demo/synthetic")
    for cfg in notebook_configs("full"):
        FINAL_RESULTS[cfg["dataset"]] = finalize(cfg["run_dir"])
    display(pd.DataFrame([{"dataset":name,"phase":result["phase"],
                           "epochs":result["epochs"],"test_accuracy":result["test_accuracy"]}
                          for name,result in FINAL_RESULTS.items()]))
''', "run-final")
    md(r"""
    ## 10. Kết quả đã đo trong PDF: kiểm chứng từ evidence

    Các cell tiếp theo đọc `report/evidence/`, `report/checkpoints/` và `results_summary.json`. Chúng kiểm tra SHA256 best, 600 dòng lịch sử liên tục, split 45k/5k disjoint, số ảnh test và số đúng trong confusion matrix. Đây là **đọc bằng chứng cũ**, không mở test loader mới.

    Nếu chỉ sao chép notebook sang thư mục khác, phần model/train vẫn có đủ code; muốn trình bày kết quả cũ thì mang kèm thư mục report/evidence, report/checkpoints và results_summary.json. Dataset đặt trong datasets/.
    """)
    code(ARCHIVE_CELL, "archived-evidence")
    md("### Đường học của 600 epoch và cấu hình train đã dùng\n\nTrain loss có nhãn mềm; validation loss có nhãn cứng. Đường best được chọn bằng validation. Thời gian là tổng engine đã lưu, không cộng khoảng nghỉ giữa các buổi train.")
    code(ARCHIVE_CURVES_CELL, "archived-learning-curves")
    md(r"""
    ### Test top-1/top-5, confusion matrix và precision/recall/F1

    `accuracy=100*correct/total`; `precision_c=TP_c/(TP_c+FP_c)`; `recall_c=TP_c/(TP_c+FN_c)`; `F1_c=2PR/(P+R)`; macro-F1 lấy trung bình các lớp. Hàng confusion là lớp thật, cột là lớp dự đoán. Top-5 kiểm tra nhãn thật nằm trong 5 logits cao nhất.

    C10 vẽ đủ số đếm; C100 vẽ đủ ma trận 100×100, thêm 15 lớp recall thấp nhất để dễ đọc. Bảng đầy đủ tất cả lớp được xuất CSV.
    """)
    code(ARCHIVE_METRICS_CELL, "archived-test-metrics")
    md(r"""
    ## 11. Nạp best 45k và dự đoán ảnh để minh họa

    Trọng số cũ có tên layer từ `src/models.py`, còn notebook dùng tên tường minh. Hàm dưới ánh xạ **tên tương ứng của đúng kiến trúc** và nạp `strict=True`; không nạp bằng thứ tự tùy ý hoặc bỏ qua weight không khớp. Nếu tự đổi kiến trúc, không dùng checkpoint cũ để gán điểm số cho model mới.

    Cell minh họa dùng **ảnh validation từ official training data**, không mở test để lựa chọn kiến trúc. Đây là ví dụ dự đoán, không phải một phép đo accuracy toàn tập mới.
    """)
    code(ADAPTER_CELL,"archive-weight-adapter")
    code(PREDICT_CELL,"archived-predictions")
    md(r"""
    ### Tùy chọn: đo lại best cũ trên đủ 10k test

    `RECHECK_ARCHIVED_TEST=False` mặc định để dùng kết quả test đã có trong PDF. Nếu chủ động bật True, đánh giá lại đúng checkpoint đã mở test trước đây, với hash/kiến trúc được kiểm tra. Không dùng điểm số này để chỉnh kiến trúc. Đây vẫn là test của **best train 45k**, không phải frozen final 50k. Kết quả mới được lưu riêng, không ghi đè evidence cũ.
    """)
    code(RECHECK_CELL,"optional-archived-test")
    md(r"""
    ## 12. So sánh optimizer/LR theo yêu cầu đề

    PDF mới có full-data SGD LR=0,1 trên hai dataset. Các cấu hình còn lại dưới đây là **kế hoạch/chưa chạy**, không phải kết quả đã đạt. `RUN_MATRIX=False` mặc định; bật khi thật sự muốn chạy nhiều thí nghiệm. Mỗi hàng có run_dir riêng, giữ best/last và resume riêng. Matrix chỉ chọn bằng validation, không tự test từng candidate.
    """)
    code(MATRIX_CELL,"optimizer-matrix")
    md(r"""
    ## 13. Đo tốc độ suy luận trên thiết bị hiện tại

    Đo forward bằng tensor giả chỉ để đo thời gian, **không phải accuracy**. Batch=1, FP32, warmup 5 lần, đo 20 lần, đồng bộ CUDA quanh phép đo. Latency này không có trong PDF cũ và có thể thay đổi theo tải máy; không suy ra latency từ FLOPs.
    """)
    code(TIMING_CELL,"inference-timing")
    md(r"""
    ## 14. Các ý cần nói khi vấn đáp

    | Câu hỏi | Ý trả lời |
    |---|---|
    | Vì sao hai kernel 3×3/5×5? | Hai phạm vi không gian trên hai phần kênh, concat lại trước grouped mixing. Đây là ý đồ thiết kế; chưa có ablation đo riêng lợi ích. |
    | Conv groups=8 làm gì? | Chia kết nối theo nhóm, giảm weight/MAC so với dense Conv cùng kích thước. C_in,C_out phải chia hết cho 8. |
    | SE có giảm kênh không? | Pool chỉ giảm H/W về 1; Conv reduce/expand mới đổi số kênh để tạo gate. |
    | ADD khác concat? | ADD yêu cầu cùng shape và giữ C; concat theo kênh cộng C, đòi N/H/W giống nhau. |
    | BN train khác eval? | Train dùng thống kê batch và cập nhật running stats; eval dùng thống kê đã lưu. |
    | SiLU làm gì? | Phi tuyến x*sigmoid(x), không đổi shape hoặc có weight học riêng. |
    | EMA và raw? | EMA là trung bình trọng số theo thời gian; ở selection so cả raw/EMA bằng validation để chọn best. |
    | Last và best? | Last để resume đủ optimizer/scaler/RNG; best selection có validation accuracy cao nhất. Cả hai chỉ cập nhật sau epoch hoàn chỉnh. |
    | Điểm trong PDF từ đâu? | Hai checkpoint best train 45k, test trên đủ 10k và kiểm chứng bằng SHA/confusion; không phải demo hoặc final 50k. |
    | Đã đủ đề chưa? | Còn so sánh full-data Adam/nhiều LR, multi-seed/ablation và final-test 50k trong snapshot báo cáo. Không điền số liệu chưa chạy. |

    Thư mục output của notebook chứa profile/layer CSV, hình kiến trúc, learning curves, confusion/per-class CSV, receipt và checkpoint lượt mới. PDF gốc là `report/Bao_cao_CIFAR10_CIFAR100.pdf`; không tự thay kết quả snapshot bằng accuracy demo.
    """)
    code(EXPORT_CELL,"export-receipt")
    notebook = nbformat.v4.new_notebook(cells=cells,metadata={
        "kernelspec":{"display_name":"Python 3.13 (existing Microsoft Store)","language":"python","name":"python3"},
        "language_info":{"name":"python","version":"3.13"},
        "compactmix":{"source_sha256":hashlib.sha256(source.encode("utf-8")).hexdigest(),
                      "reference":"https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/cifar10_tutorial.ipynb",
                      "runtime":"existing Microsoft Store Python 3.13; no environment creation",
                      "default_mode":"demo"}})
    nbformat.validate(notebook)
    OUTPUT.write_text(nbformat.writes(notebook),encoding="utf-8",newline="\n")
    print(f"Created {OUTPUT}: {len(cells)} cells, complete inline model and trainer")
    return notebook


PROFILE_CELL = r'''
OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
PROFILES = {}
profile_rows=[]
for name in DATASETS:
    classes=10 if name=="cifar10" else 100
    check_model=make_model(classes)
    profile=inspect_model(check_model,classes,backward=True)
    PROFILES[name]=profile
    profile_rows.append({"dataset":name,"parameters":profile["parameters"],
        "MACs/ảnh":profile["macs"],"FLOPs/ảnh":profile["flops"],"GFLOPs":profile["flops"]/1e9,
        "<=6M":profile["parameters"]<=6_000_000,"<1G":profile["flops"]<1e9,
        "forward/backward":"OK","weight FP32 MiB":sum(p.numel()*p.element_size() for p in check_model.parameters())/2**20})
    layers=pd.DataFrame(profile["layers"])
    modules=dict(check_model.named_modules())
    layers["settings"]=[str(modules[n]) for n in layers["name"]]
    layers.to_csv(OUTPUT_ROOT/f"{name}_layers.csv",index=False,encoding="utf-8-sig")
    display(Markdown(f"**{name}: đủ {len(layers)} leaf layers — shape đo bằng forward N=1**"))
    with pd.option_context("display.max_rows",300):
        display(layers)
    summary=[]
    for prefix in ("stem","block01","block02","t1","block03","block04","block05","t2",
                   *[f"block{i:02}" for i in range(6,14)],"head","classifier"):
        selected=[(n,m) for n,m in check_model.named_children() if n==prefix or n.startswith(prefix+"_")]
        summary.append({"khối":prefix,"parameters":sum(p.numel() for _,m in selected for p in m.parameters())})
    assert sum(row["parameters"] for row in summary)==profile["parameters"]
    display(pd.DataFrame(summary))
    del check_model
PROFILE_TABLE=pd.DataFrame(profile_rows)
display(PROFILE_TABLE)
PROFILE_TABLE.to_csv(OUTPUT_ROOT/"model_profiles.csv",index=False)
'''

DIAGRAM_CELL = r'''
def draw_flow(labels,title,filename):
    fig,ax=plt.subplots(figsize=(11,max(4,len(labels)*.55)))
    ax.set_xlim(0,10); ax.set_ylim(0,len(labels)*1.2); ax.axis("off")
    for i,label in enumerate(labels):
        y=(len(labels)-i-1)*1.2+.2
        ax.add_patch(FancyBboxPatch((1,y),8,.8,boxstyle="round,pad=.07",facecolor="#e8f2ff",edgecolor="#2d5e91"))
        ax.text(5,y+.4,label,ha="center",va="center",fontsize=10)
        if i<len(labels)-1:
            ax.annotate("",xy=(5,y-.25),xytext=(5,y),arrowprops={"arrowstyle":"->","color":"#2d5e91"})
    ax.set_title(title,fontsize=15,pad=15)
    fig.tight_layout(); fig.savefig(OUTPUT_ROOT/filename,bbox_inches="tight")
    plt.show(); plt.close(fig)

draw_flow([
    "Input RGB [N,3,32,32]",
    "Stem Conv3×3: 3→40, s=1,p=1 → [N,40,32,32]",
    "Stage 1: Block01–02, C=40 → [N,40,32,32]",
    "BN → SiLU → AvgPool2×2 → Conv1×1 40→112 → [N,112,16,16]",
    "Stage 2: Block03–05, C=112 → [N,112,16,16]",
    "BN → SiLU → AvgPool2×2 → Conv1×1 112→256 → [N,256,8,8]",
    "Stage 3: Block06–13, C=256 → [N,256,8,8]",
    "Head BN → SiLU → GlobalAvgPool → [N,256,1,1]",
    "Flatten → Dropout(p=0) → [N,256]",
    "Linear 256→K → logits [N,K], K=10 hoặc 100",
],"CompactMix-L — cấu hình gốc", "architecture.png")
draw_flow([
    "x [N,C,H,W] — giữ một bản cho đường skip",
    "BN(C) → SiLU → Conv1×1 C→2C → BN(2C) → SiLU",
    "chunk(2,dim=1) → hai nhánh [N,C,H,W]",
    "Nhánh A: DW3×3,p=1 | Nhánh B: DW5×5,p=2",
    "Concat theo kênh → [N,2C,H,W] → BN → SiLU",
    "Grouped Conv3×3,groups=8,p=1 → BN → SiLU → feature",
    "SE: GAP → Conv 2C→D → SiLU → Conv D→2C → Sigmoid",
    "feature × gate [N,2C,1,1] → [N,2C,H,W]",
    "Projection Conv1×1 2C→C → DropPath → [N,C,H,W]",
    "ADD với skip x → [N,C,H,W]",
],"MixBlock — C=40/112/256, D=8/14/32", "mixblock.png")
'''

LOSS_CELL = r'''
cfg=CONFIGS[0]
seed_everything(cfg["seed"],cfg["deterministic"])
source,split=DATA_PREVIEW[cfg["dataset"]]
demo_images,demo_labels=next(iter(make_loader(source,split["training"][:8],cfg,0,training=True)))
demo_images,soft_targets,paired,coefficient=mix_batch(demo_images,demo_labels,split["classes"],cfg,0)
demo_logits=torch.randn(len(demo_labels),split["classes"])
demo_loss=-(soft_targets*F.log_softmax(demo_logits,dim=1)).sum(1).mean()
print("images",tuple(demo_images.shape),"labels",tuple(demo_labels.shape),
      "soft targets",tuple(soft_targets.shape),"logits",tuple(demo_logits.shape),"loss",tuple(demo_loss.shape))
print("Ví dụ loss với logits ngẫu nhiên:",round(demo_loss.item(),4),"— chỉ minh họa công thức")
lr_cfg={**cfg,"epochs":600}
schedule=[learning_rate(lr_cfg,epoch,0,100) for epoch in range(600)]
fig,ax=plt.subplots(figsize=(8,2.7)); ax.plot(schedule)
ax.set(xlabel="Epoch",ylabel="LR",title="Warmup + cosine — lịch full 600 epoch")
plt.show(); plt.close(fig)
'''

TRAIN_CELL = r'''
def run_requested_training():
    global REQUEST_STOP
    REQUEST_STOP=False
    results={}
    if MODE=="report_only":
        return results
    for cfg in notebook_configs():
        # Preflight im lặng, không sửa shape và không dùng model kiểm tra để train.
        inspect_model(make_model(10 if cfg["dataset"]=="cifar10" else 100),
                      10 if cfg["dataset"]=="cifar10" else 100,backward=True)
        results[cfg["dataset"]]=run_experiment(cfg,
            stop_after_epochs=DEMO_STOP_AFTER if MODE=="demo" else None)
    return results

NEW_RESULTS=run_requested_training()
'''

NEW_CURVES_CELL = r'''
def plot_history(history,title,filename):
    frame=pd.DataFrame(history)
    if frame.empty:
        return
    fig,axes=plt.subplots(1,3,figsize=(14,3.3))
    for key,label in (("train_loss","Train soft-label"),("val_loss","Val raw"),("ema_val_loss","Val EMA")):
        if key in frame: axes[0].plot(frame.epoch,frame[key],label=label)
    for key,label in (("train_mixed_accuracy","Train mixed"),("val_accuracy","Val raw"),("ema_val_accuracy","Val EMA")):
        if key in frame: axes[1].plot(frame.epoch,frame[key],label=label)
    if "lr" in frame: axes[2].plot(frame.epoch,frame.lr)
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.grid(alpha=.2)
    axes[0].set_ylabel("Loss"); axes[1].set_ylabel("Accuracy (%)"); axes[2].set_ylabel("LR")
    axes[0].legend(); axes[1].legend(); fig.suptitle(title)
    fig.tight_layout(); fig.savefig(OUTPUT_ROOT/filename,bbox_inches="tight")
    plt.show(); plt.close(fig)

checkpoint_rows=[]
for cfg in notebook_configs():
    folder=Path(cfg["run_dir"])
    if not (folder/"last.pt").exists(): continue
    last=load_checkpoint(folder/"last.pt"); best=load_checkpoint(folder/"best.pt")
    assert {p.name for p in folder.glob("*.pt")}=={"best.pt","last.pt"}
    checkpoint_rows.append({"dataset":cfg["dataset"],"last epoch":last["epoch"],
        "best epoch":best["epoch"],"best val %":best["accuracy"],"weights":best["kind"],
        "run":folder.name,"subset":bool(cfg["limit_train"] or cfg["limit_val"]),"synthetic":cfg["smoke"]})
    plot_history(last["history"],cfg["dataset"]+" — lượt mới, "+MODE,cfg["dataset"]+"_new_curves.png")
if checkpoint_rows: display(pd.DataFrame(checkpoint_rows))
'''

ARCHIVE_CELL = r'''
def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def archived_bundle(name):
    folder=REPORT_DIR/"evidence"/name
    required=[folder/"metrics.csv",folder/"split.json",folder/"test_evaluation.json",
              folder/"result.json",folder/"config.json",REPORT_DIR/"checkpoints"/name/"best.pt"]
    missing=[str(p) for p in required if not p.exists()]
    if missing: raise FileNotFoundError("Thiếu evidence cũ: "+", ".join(missing))
    history=pd.read_csv(folder/"metrics.csv")
    split=read_json(folder/"split.json")
    test=read_json(folder/"test_evaluation.json")
    result=read_json(folder/"result.json")
    checkpoint=REPORT_DIR/"checkpoints"/name/"best.pt"
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest!=test["checkpoint_sha256"]: raise ValueError("Archived best SHA mismatch: "+name)
    assert history.epoch.tolist()==list(range(1,601))
    assert len(split["training"])==45000 and len(split["validation"])==5000
    assert not set(split["training"]) & set(split["validation"])
    assert sorted(split["training"]+split["validation"])==list(range(50000))
    cm=np.asarray(test["test_metrics"]["confusion_matrix"])
    assert int(cm.sum())==test["samples"]==10000
    assert int(cm.trace())==test["correct"]
    assert math.isclose(100*cm.trace()/cm.sum(),test["test_metrics"]["accuracy"],abs_tol=1e-9)
    assert test["training_images"]==45000 and test["is_frozen_final_retraining"] is False
    return {"history":history,"split":split,"test":test,"result":result,
            "config":read_json(folder/"config.json"),"checkpoint":checkpoint,
            "per_class":pd.read_csv(folder/"per_class_metrics.csv")}

ARCHIVED={}
archive_rows=[]
for name in DATASETS:
    bundle=archived_bundle(name); ARCHIVED[name]=bundle
    result,test=bundle["result"],bundle["test"]
    archive_rows.append({"dataset":name,"scope":"Archived best train 45k",
        "best epoch":test["epoch"],"weights":test["weight_kind"],
        "validation top1 %":test["validation_accuracy"],"test top1 %":test["test_metrics"]["accuracy"],
        "test top5 %":test["test_metrics"]["top5_accuracy"],"test macro-F1":test["test_metrics"]["macro_f1"],
        "correct/10000":test["correct"],"incorrect":test["incorrect"],
        "parameters":result["parameters"],"FLOPs":result["flops"],"SHA verified":True})
ARCHIVED_TABLE=pd.DataFrame(archive_rows)
display(ARCHIVED_TABLE)
ARCHIVED_TABLE.to_csv(OUTPUT_ROOT/"archived_summary.csv",index=False)
'''

ARCHIVE_CURVES_CELL = r'''
for name,bundle in ARCHIVED.items():
    test,result=bundle["test"],bundle["result"]
    plot_history(bundle["history"],name+f" — archive 600 epoch, best @{test['epoch']}",name+"_archived_curves.png")
    config=pd.DataFrame({"setting":list(bundle["config"]),"value":[str(v) for v in bundle["config"].values()]})
    display(Markdown("**"+name+" — learning settings đã lưu**")); display(config)
    display(pd.DataFrame([{"epoch count":result["epochs"],"best epoch":result["best_epoch"],
        "training seconds":result["training_time"],"mean seconds/epoch":result["training_time"]/result["epochs"],
        "AMP skips":int(bundle["history"].amp_skips.sum()),"test seconds":test["evaluation_seconds"],
        "test precision":test["precision"],"test batch":test["evaluation_batch_size"],
        "test augmentation":test["augmentation"],"TTA":test["test_time_augmentation"]}]))
'''

ARCHIVE_METRICS_CELL = r'''
def per_class_from_confusion(cm,names):
    cm=np.asarray(cm,dtype=np.int64)
    tp=np.diag(cm); support=cm.sum(1); predicted=cm.sum(0)
    recall=tp/np.maximum(support,1); precision=tp/np.maximum(predicted,1)
    f1=2*precision*recall/np.maximum(precision+recall,1e-12)
    return pd.DataFrame({"class_id":range(len(cm)),"class_name":names,"support":support,
        "correct":tp,"precision_percent":100*precision,"recall_percent":100*recall,"f1_percent":100*f1})

for name,bundle in ARCHIVED.items():
    cm=np.asarray(bundle["test"]["test_metrics"]["confusion_matrix"])
    names=bundle["per_class"].class_name.tolist()
    per_class=per_class_from_confusion(cm,names)
    assert math.isclose(per_class.f1_percent.mean()/100,bundle["test"]["test_metrics"]["macro_f1"],abs_tol=1e-6)
    np.testing.assert_allclose(per_class.recall_percent,bundle["per_class"].recall_percent,atol=1e-6)
    per_class.to_csv(OUTPUT_ROOT/f"{name}_per_class.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(cm,index=names,columns=names).to_csv(OUTPUT_ROOT/f"{name}_confusion.csv",encoding="utf-8-sig")
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    im=axes[0].imshow(100*cm/np.maximum(cm.sum(1,keepdims=True),1),vmin=0,vmax=100,cmap="Blues")
    fig.colorbar(im,ax=axes[0],label="% theo hàng")
    axes[0].set(xlabel="Predicted class",ylabel="True class",title=name+" — đủ "+str(len(cm))+" lớp")
    if len(cm)==10:
        axes[0].set_xticks(range(10),names,rotation=70); axes[0].set_yticks(range(10),names)
        for i in range(10):
            for j in range(10):
                axes[0].text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=6,color="white" if i==j else "black")
    worst=per_class.nsmallest(min(15,len(cm)),"recall_percent").sort_values("recall_percent")
    axes[1].barh(worst.class_name,worst.recall_percent,color="#397bb5")
    axes[1].set(xlim=(0,100),xlabel="Recall (%)",title="Các lớp recall thấp nhất")
    fig.tight_layout(); fig.savefig(OUTPUT_ROOT/f"{name}_archived_confusion_recall.png",bbox_inches="tight")
    plt.show(); plt.close(fig)
    display(Markdown("**"+name+" — toàn bộ chỉ số từng lớp**"))
    with pd.option_context("display.max_rows",110): display(per_class)
'''

ADAPTER_CELL = r'''
def load_archived_model(name):
    bundle=ARCHIVED[name]
    checkpoint=load_checkpoint(bundle["checkpoint"])
    model=make_model(10 if name=="cifar10" else 100).eval()
    profile=inspect_model(model,10 if name=="cifar10" else 100,backward=False)
    assert profile["parameters"]==bundle["result"]["parameters"]
    assert profile["flops"]==bundle["result"]["flops"]
    prefixes={"stem.0":"stem","stages.1.0":"t1_bn","stages.1.3":"t1_conv",
              "stages.2.0":"t2_bn","stages.2.3":"t2_conv","head.0":"head_bn","head.4":"classifier"}
    suffixes={"pre.0":"bn_pre","pre.2":"expand","pre.3":"bn_expand",
              "local":"local","context":"context","mix.0":"bn_mix","mix.2":"grouped",
              "mix.3":"bn_gate","mix.5.gate.0":"se_reduce","mix.5.gate.2":"se_expand","mix.6":"project"}
    block_locations=[(0,0),(0,1),(1,4),(1,5),(1,6),*[(2,i) for i in range(4,12)]]
    for number,(stage,position) in enumerate(block_locations,1):
        for old,new in suffixes.items():
            prefixes[f"stages.{stage}.{position}.{old}"]=f"block{number:02}.{new}"
    converted={}
    for old_key,value in checkpoint["model"].items():
        prefix,suffix=old_key.rsplit(".",1)
        if prefix not in prefixes: raise ValueError("Không có ánh xạ weight: "+old_key)
        new_key=prefixes[prefix]+"."+suffix
        if new_key in converted: raise ValueError("Weight mapping trùng: "+new_key)
        converted[new_key]=value
    model.load_state_dict(converted,strict=True)
    return model.to(DEVICE).eval()
'''

PREDICT_CELL = r'''
for name,bundle in ARCHIVED.items():
    model=load_archived_model(name)
    cfg={**bundle["config"],"workers":0,"batch_size":8,"device":DEVICE,
         "data_root":str(DATA_ROOT),"download":False,"smoke":False,"limit_train":0,"limit_val":0}
    constructor=datasets.CIFAR10 if name=="cifar10" else datasets.CIFAR100
    source=constructor(root=str(DATA_ROOT),train=True,download=False)
    indices=bundle["split"]["validation"][:8]
    images,labels=next(iter(make_loader(source,indices,cfg,0,training=False)))
    with torch.no_grad(): predictions=model(images.to(DEVICE)).argmax(1).cpu()
    fig,axes=plt.subplots(2,4,figsize=(11,5))
    rows=[]
    for ax,index,truth,prediction in zip(axes.flat,indices,labels.tolist(),predictions.tolist()):
        image,_=source[index]; ax.imshow(image); ax.axis("off")
        ax.set_title("True: "+source.classes[truth]+"\nPred: "+source.classes[prediction],
                     color="green" if truth==prediction else "red",fontsize=9)
        rows.append({"index training set":index,"true":source.classes[truth],"predicted":source.classes[prediction]})
    fig.suptitle(name+" — archived best, 8 ảnh VALIDATION để minh họa")
    fig.tight_layout(); fig.savefig(OUTPUT_ROOT/f"{name}_validation_predictions.png",bbox_inches="tight")
    plt.show(); plt.close(fig); display(pd.DataFrame(rows)); del model
'''

RECHECK_CELL = r'''
LIVE_ARCHIVED_TEST={}
if RECHECK_ARCHIVED_TEST:
    for name,bundle in ARCHIVED.items():
        digest=hashlib.sha256(bundle["checkpoint"].read_bytes()).hexdigest()
        if digest!=bundle["test"]["checkpoint_sha256"]: raise ValueError("Archived checkpoint đã đổi")
        cache=OUTPUT_ROOT/f"{name}_archived_live_test.json"
        if cache.exists():
            measured=read_json(cache)
            if measured["checkpoint_sha256"]!=digest or measured["source_sha256"]!=source_fingerprint():
                raise ValueError("Cache test không khớp source/checkpoint; giữ file để kiểm tra")
        else:
            model=load_archived_model(name)
            constructor=datasets.CIFAR10 if name=="cifar10" else datasets.CIFAR100
            source=constructor(root=str(DATA_ROOT),train=False,download=False)
            assert len(source)==10000
            cfg={**bundle["config"],"workers":0,"batch_size":64,"device":DEVICE}
            started=time.monotonic()
            metrics=evaluate(model,make_loader(source,list(range(len(source))),cfg,0),DEVICE,extended=True)
            measured={"scope":"archived selection45k test, not final50k","dataset":name,"samples":len(source),
                "checkpoint_sha256":digest,"source_sha256":source_fingerprint(),"metrics":metrics,
                "seconds":time.monotonic()-started,"environment":environment(cfg)}
            atomic_json(cache,measured); del model
        LIVE_ARCHIVED_TEST[name]=measured
    display(pd.DataFrame([{"dataset":n,"scope":r["scope"],"samples":r["samples"],
                          **{k:r["metrics"][k] for k in ("accuracy","top5_accuracy","macro_f1")}}
                         for n,r in LIVE_ARCHIVED_TEST.items()]))
'''

MATRIX_CELL = r'''
MATRIX=[("sgd",.05),("sgd",.1),("sgd",.15),("adam",.0003),("adam",.001),("adam",.003)]
comparison=[]
for optimizer,lr in MATRIX:
    row={"optimizer":optimizer,"initial LR":lr}
    for name in DATASETS:
        row[name+" archived val/test %"]=(f"{ARCHIVED[name]['test']['validation_accuracy']:.2f} / "
            f"{ARCHIVED[name]['test']['test_metrics']['accuracy']:.2f}" if optimizer=="sgd" and lr==.1 else "Chưa có full-data trong PDF")
    comparison.append(row)
display(pd.DataFrame(comparison))
MATRIX_RESULTS=[]
if RUN_MATRIX:
    for optimizer,lr in MATRIX:
        for cfg in notebook_configs("full",optimizer,lr):
            result=run_experiment(cfg)
            MATRIX_RESULTS.append(result)
    display(pd.DataFrame([{"dataset":r["dataset"],"optimizer":r["optimizer"],"lr":r["learning_rate"],
                           "validation %":r["best_val_accuracy"],"test %":r["test_accuracy"]} for r in MATRIX_RESULTS]))
'''

TIMING_CELL = r'''
timing_rows=[]
for name in DATASETS:
    model=make_model(10 if name=="cifar10" else 100).to(DEVICE).eval()
    sample=torch.zeros(1,3,32,32,device=DEVICE)
    def sync():
        if DEVICE=="cuda": torch.cuda.synchronize()
    times=[]
    with torch.no_grad():
        for _ in range(5): model(sample)
        sync()
        for _ in range(20):
            sync(); start=time.perf_counter(); model(sample); sync()
            times.append(1000*(time.perf_counter()-start))
    timing_rows.append({"dataset":name,"device":DEVICE,"batch":1,"dtype":"FP32","repetitions":20,
        "median ms":float(np.median(times)),"p95 ms":float(np.percentile(times,95)),
        "mean images/s":1000/float(np.mean(times)),"scope":"technical forward timing, not accuracy"})
    del model
TIMING_TABLE=pd.DataFrame(timing_rows); display(TIMING_TABLE)
TIMING_TABLE.to_csv(OUTPUT_ROOT/"live_inference_timing.csv",index=False)
'''

EXPORT_CELL = r'''
receipt={"created_at_utc":datetime.now(timezone.utc).isoformat(),"training_source_sha256":source_fingerprint(),
    "mode":MODE,"new_run_config":notebook_configs(),"runtime":environment(CONFIGS[0]),
    "profiles":PROFILE_TABLE.to_dict(orient="records"),"historical_archived_results":archive_rows,
    "live_forward_timing":timing_rows,
    "notes":["Archived test belongs to selection45k, not final50k",
             "Demo/subset/synthetic scores are not full-data benchmark evidence",
             "Full-data Adam/multi-LR results remain absent in the original PDF snapshot"]}
atomic_json(OUTPUT_ROOT/"presentation_receipt.json",receipt)
print("Output:",OUTPUT_ROOT)
print("Receipt:",OUTPUT_ROOT/"presentation_receipt.json")
display(Markdown("**Trình bày:** dữ liệu → block/forward → budget → loss/optimizer → một epoch/resume → kết quả 600 epoch và test đã lưu → giới hạn của bài làm."))
'''


if __name__ == "__main__":
    build_notebook()
