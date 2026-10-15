"""Export measured short-run comparisons, without training or opening test data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]


def build(directory):
    from datetime import datetime, timedelta, timezone
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.loads((directory / "comparison.json").read_text(encoding="utf-8"))
    if not rows:
        raise ValueError("No completed comparisons to report")
    results = {r["variant"]: json.loads((directory / r["variant"] / "result.json").read_text(encoding="utf-8"))
               for r in rows}
    baseline = next(r for r in rows if r["variant"] == "baseline")
    config = results["baseline"]["config"]
    suite = json.loads((directory / "suite.json").read_text(encoding="utf-8"))
    timing = json.loads((directory / "timing.json").read_text(encoding="utf-8"))
    split = json.loads((directory / "baseline/split.json").read_text(encoding="utf-8"))
    for name, result in results.items():
        assert result["subset"] and not result["synthetic"] and result["test_accuracy"] is None
        assert result["epochs"] == config["epochs"]
        assert len(result["history"]) == config["epochs"]
        assert result["metadata"]["source_sha256"] == suite["source_sha256"]
        other = json.loads((directory / name / "split.json").read_text(encoding="utf-8"))
        assert other == split
    report = ROOT / "report"
    figures = report / "figures"
    figures.mkdir(exist_ok=True)
    base = "THU_NGHIEM_NHANH_MOMENTUM_LR_SCHEDULER"
    csv_path = report / (base + ".csv")
    shutil.copyfile(directory / "comparison.csv", csv_path)
    evidence_path = report / (base + ".json")
    evidence = {"suite": suite, "split": split, "timing": timing,
                "results": results, "summary": rows, "official_test_opened": False,
                "scope": "CIFAR-10 subset, single seed, short schedule; not final benchmark evidence"}
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    groups = (
        ("Momentum (LR=0.1, cosine)", ["momentum_080", "baseline", "momentum_095"], "momentum"),
        ("Base LR (momentum=0.9, cosine)", ["lr_005", "baseline", "lr_015"], "base_lr"),
        ("Scheduler (momentum=0.9, LR=0.1)", ["baseline", "scheduler_constant", "scheduler_step"], "scheduler"),
    )
    by_name = {r["variant"]: r for r in rows}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, (title, names, field) in zip(axes, groups):
        selected = [by_name[n] for n in names if n in by_name]
        values = [r["best_val_pct"] for r in selected]
        bars = ax.bar([str(r[field]) for r in selected], values,
                      color=["#2563eb" if r["variant"] == "baseline" else "#94a3b8" for r in selected])
        ax.bar_label(bars, fmt="%.2f%%", padding=4)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, min(100, max(r["best_val_pct"] for r in rows) + 12))
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Best validation accuracy (%) — raw or EMA")
    fig.suptitle(f"CIFAR-10: {config['limit_train']:,} train / {config['limit_val']:,} validation, "
                 f"{config['epochs']} epochs, seed 42 | subset evidence; official test sealed", fontsize=11)
    fig.tight_layout()
    bars_path = figures / "SO_SANH_NHANH_MOMENTUM_LR_SCHEDULER.png"
    fig.savefig(bars_path, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, result in results.items():
        history = result["history"]
        axes[0].plot([r["epoch"] for r in history],
                     [max(r["val_accuracy"], r["ema_val_accuracy"]) for r in history],
                     marker="o", markersize=3, label=name)
    axes[0].set(title="Validation across completed epochs (raw/EMA max)", xlabel="Completed epoch", ylabel="Accuracy (%)")
    from run_quick_recipe_comparison import learning_rate
    for name in ("baseline", "scheduler_constant", "scheduler_step", "lr_005", "lr_015"):
        if name not in results:
            continue
        c = results[name]["config"]
        batches = (c["limit_train"] + c["batch_size"] - 1) // c["batch_size"]
        x = [step / batches for step in range(c["epochs"] * batches)]
        y = [learning_rate(c, step // batches, step % batches, batches)
             for step in range(c["epochs"] * batches)]
        axes[1].plot(x, y, label=name)
    axes[1].set(title="Actual per-batch learning-rate schedules", xlabel="Epoch progress", ylabel="Learning rate")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.tight_layout()
    curves_path = figures / "DUONG_CONG_NHANH_VALIDATION_VA_LR.png"
    fig.savefig(curves_path, dpi=160)
    plt.close(fig)

    lines = ["# So sánh nhanh momentum, initial LR và scheduler", "",
             "**Đây là thực nghiệm ngắn trên validation CIFAR-10, không phải kết quả test hoặc bằng chứng tối ưu cho 600 epoch.**", "",
             "## Cách chạy", "",
             f"- CompactMix-L gốc: {suite['profile']['parameters']:,} tham số; {suite['profile']['flops']/1e9:.9f} GFLOPs/ảnh.",
             f"- {config['limit_train']:,} ảnh train và {config['limit_val']:,} ảnh validation, lấy cân bằng từng lớp từ split 45k/5k, không giao nhau.",
             f"- Mỗi cấu hình khởi tạo mới cùng seed 42, split seed 2026, batch 128, {config['epochs']} epoch; chỉ đổi một yếu tố so với baseline.",
             f"- Hoàn thành {len(rows)} cấu hình trong {timing['wall_seconds']/60:.2f} phút (không tính chuẩn bị, đo tốc độ ban đầu và xuất báo cáo).",
             "- Cùng SGD + Nesterov=True, weight decay 0,0005 (trừ nhóm bias/BN), AutoAugment, Mixup/CutMix, smoothing 0,05, EMA, AMP và DropPath.",
             f"- Warmup thực tế = min(5, epochs/10) = **{suite['effective_warmup_epochs']:g} epoch**: LR từ 10% lên 100% base LR. Đây không phải warmup 5 epoch của run 600 epoch.",
             "- Cosine giảm về 0,001×base LR; constant giữ base LR sau warmup; step giảm ×0,1 ở giữa đoạn sau warmup. Hai lịch constant/step chỉ thêm trong script thử nghiệm, không thay trainer gốc.",
             "- Với lịch ngắn dưới 10 epoch, code gốc không có đoạn cuối tắt Mixup/CutMix; kết quả không mô phỏng đầy đủ recipe 600 epoch.",
             "- Best chọn theo validation raw/EMA như engine gốc; có cả số cuối epoch để nhìn độ ổn định. Chỉ giữ best.pt và last.pt cho mỗi run.",
             f"- GPU: {suite['environment']['gpu']}; Python {suite['environment']['python']}, PyTorch {suite['environment']['pytorch']}.", "",
             "## Kết quả đo", "",
             "| Cấu hình | Momentum | Base LR | Scheduler | Best val % | Δ so baseline (điểm %) | Epoch/weights best | Val raw cuối % | Val EMA cuối % | Raw loss cuối | AMP skips |",
             "|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|"]
    for r in rows:
        delta = r["best_val_pct"] - baseline["best_val_pct"]
        lines.append(f"| {r['variant']} | {r['momentum']:g} | {r['base_lr']:g} | {r['scheduler']} | "
                     f"{r['best_val_pct']:.2f} | {delta:+.2f} | {r['best_epoch']}/{r['best_weights']} | "
                     f"{r['final_raw_val_pct']:.2f} | {r['final_ema_val_pct']:.2f} | "
                     f"{r['final_raw_val_loss']:.4f} | {r['amp_skips']} |")
    lines.extend(["", f"![So sánh ba yếu tố](figures/{bars_path.name})", "",
                  f"![Đường cong validation và LR](figures/{curves_path.name})", "",
                  "## Có thể giải thích lựa chọn trong bài như thế nào?", ""])
    for title, names, field in groups:
        selected = [by_name[n] for n in names if n in by_name]
        winner = max(selected, key=lambda r: r["best_val_pct"])
        lines.append(f"- **{field}:** trong phép thử này, mức `{winner[field]}` có best validation cao nhất "
                     f"({winner['best_val_pct']:.2f}%). Baseline đạt {baseline['best_val_pct']:.2f}%. "
                     "Nếu có điểm bằng nhau thì không kết luận hơn/kém.")
    lines.extend(["",
                  "> Em chọn SGD momentum 0,9, base LR 0,1 và cosine làm cấu hình nền. Momentum giúp tích lũy hướng gradient; warmup giảm bước cập nhật ban đầu; cosine giảm LR để tinh chỉnh về cuối. Em đã chạy thêm các cấu hình đối chiếu trên cùng tập con và ngân sách. Bảng cho thấy kết quả học ban đầu; em không dùng nó để khẳng định cấu hình gốc tối ưu cho toàn bộ quá trình train.", "",
                  "Nếu baseline không đứng đầu ở một yếu tố, trình bày đúng rằng mức khác học tốt hơn trong phép thử ngắn. Không đổi kết quả hoặc viết lý do để biến cấu hình đã chọn thành người thắng.", "",
                  "## Giới hạn và cách đọc", "",
                  "- Một seed, ít ảnh và ít epoch: thứ hạng có thể thay đổi khi train dài hoặc đổi seed. Chênh lệch nhỏ chưa chứng minh lợi ích ổn định.",
                  "- So sánh scheduler đồng thời thay đổi tổng mức LR tích lũy; đây là so sánh toàn bộ lịch, không tách riêng hình dạng khỏi tổng bước cập nhật.",
                  "- AMP có thể bỏ cập nhật nếu gradient không hữu hạn; cột AMP skips công khai số lần, nên cùng số batch không luôn đồng nghĩa cùng số cập nhật thành công.",
                  "- Không mở official test; không gán các số này cho điểm test 97,18%/82,98%. Không có kết luận mới cho CIFAR-100 hoặc Adam.",
                  "- Thử nghiệm tập con này không hoàn thành yêu cầu SGD/Adam và nhiều LR trên cả hai dataset của Final exam.docx.", "",
                  "## File và lệnh tái lập", "",
                  f"- CSV: [{csv_path.name}]({csv_path.name}); JSON có split indices, config, environment, source hash và đầy đủ history: [{evidence_path.name}]({evidence_path.name}).",
                  f"- Run/checkpoint/log: `{directory.relative_to(ROOT).as_posix()}/<variant>/`.",
                  f"- Split SHA256: `{suite['split_sha256']}`; initial weights SHA256: `{suite['initial_weights_sha256']}`.",
                  "- Lượt đo tốc độ ban đầu 8 epoch trong `experiments/quick_recipe_20261016` đã dừng sớm, giữ checkpoint nhưng không đưa vào bảng. Chỉ các run hoàn chỉnh cùng ngân sách ở thư mục 5epoch được so sánh.", "",
                  "```powershell",
                  r"& 'C:\Users\anhna\AppData\Local\Microsoft\WindowsApps\python3.13.exe' -X utf8 -u scripts/run_quick_recipe_comparison.py --epochs 5 --output experiments/quick_recipe_20261016_5epoch --budget-seconds 660",
                  "```", "",
                  "Chạy lại đúng lệnh sẽ dùng checkpoint cùng source/config; đổi recipe/source thì chọn --output mới. Không cần tạo môi trường Python khác."])
    md_path = report / (base + ".md")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_path = report / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stamp = datetime.now(timezone(timedelta(hours=7))).isoformat()
    for path in (md_path, csv_path, evidence_path, bars_path, curves_path):
        name = path.relative_to(report).as_posix()
        payload = path.read_bytes()
        manifest["files"][name] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        manifest["supplement_updates"][name] = stamp
    manifest["file_count"] = len(manifest["files"])
    manifest_path.write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").replace("\n", "\r\n").encode())
    print(f"Exported {len(rows)} measured comparisons to {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    build(parser.parse_args().directory.resolve())
