"""Artifact export and PDF reporting; input data is measured, never fabricated."""

import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                               Table, TableStyle, PageBreak)

from metrics import paired_bootstrap
from utils import write_json


def save_figure(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=140, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def training_figures(history_path, directory):
    if not Path(history_path).exists():
        return
    history = pd.read_csv(history_path)
    timing = Path(history_path).parent / "timing.csv"
    if timing.exists():
        history = history.merge(pd.read_csv(timing), on="epoch", how="left", validate="one_to_one")
    for name, columns in (
        ("learning_loss", ["train_loss", "validation_loss"]),
        ("learning_accuracy", ["train_weighted_accuracy", "validation_accuracy"]),
        ("learning_rate", ["lr_first", "lr_last"]),
        ("epoch_time", ["epoch_wall_seconds", "train_seconds", "validation_seconds"]),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        for column in columns:
            if column in history and history[column].notna().any():
                ax.plot(history.epoch, history[column], label=column)
        ax.set(xlabel="Epoch", title=name.replace("_", " "))
        ax.legend()
        ax.grid(alpha=.25)
        save_figure(fig, directory, name)


def evaluation_figures(metrics, probabilities, labels, directory):
    classes = len(metrics["per_class"])
    names = [row["class_name"] for row in metrics["per_class"]]
    for field, name in (("confusion_matrix", "confusion_counts"),
                        ("confusion_matrix_normalized", "confusion_normalized")):
        matrix = np.asarray(metrics[field])
        fig, ax = plt.subplots(figsize=(9, 8))
        image = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
        fig.colorbar(image, ax=ax)
        ax.set(xlabel="Predicted", ylabel="True", title=name)
        if classes <= 10:
            ax.set_xticks(range(classes), names, rotation=45, ha="right")
            ax.set_yticks(range(classes), names)
            for i in range(classes):
                for j in range(classes):
                    ax.text(j, i, f"{matrix[i,j]:.2f}" if "normalized" in name else str(matrix[i,j]),
                            ha="center", va="center", fontsize=7,
                            color="white" if matrix[i,j] > matrix.max()/2 else "black")
        save_figure(fig, directory, name)
    per = pd.DataFrame(metrics["per_class"])
    selected = per if classes <= 10 else per.nsmallest(20, "f1")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(selected))
    for offset, metric in zip((-.25, 0, .25), ("precision", "recall", "f1")):
        ax.bar(x + offset, selected[metric], width=.25, label=metric)
    ax.set_xticks(x, selected.class_name, rotation=60, ha="right")
    ax.set(ylim=(0, 1), title="Per-class scores" if classes <= 10 else "20 lowest-F1 classes")
    ax.legend()
    save_figure(fig, directory, "class_scores")
    nonempty = [row for row in metrics["calibration"] if row["count"]]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.plot([r["confidence"] for r in nonempty], [r["accuracy"] for r in nonempty], "o-")
    ax.set(xlim=(0,1), ylim=(0,1), xlabel="Mean confidence", ylabel="Empirical accuracy",
           title="Reliability (15 equal-width bins)")
    save_figure(fig, directory, "reliability")
    confidence = probabilities.max(1)
    correct = probabilities.argmax(1) == labels
    fig, ax = plt.subplots(figsize=(7,4))
    for mask, name in ((correct, "correct"), (~correct, "wrong")):
        if mask.any():
            ax.hist(confidence[mask], bins=np.linspace(0,1,21), alpha=.6, label=name)
    ax.set(xlabel="Confidence", ylabel="Samples")
    ax.legend()
    save_figure(fig, directory, "confidence")
    for name in ("roc", "precision_recall"):
        fig, ax = plt.subplots(figsize=(7,5))
        curves = metrics["curves"]
        keys = [str(i) for i in range(classes)] if classes <= 10 else [str(int(i)) for i in per.nsmallest(5,"f1").class_id]
        for key in keys + ["micro"] + (["macro_roc"] if name == "roc" else []):
            if key not in curves:
                continue
            curve = curves[key]
            label = names[int(key)] if key.isdigit() else key
            if name == "roc":
                ax.plot(curve["fpr"], curve["tpr"], label=label)
            else:
                ax.plot(curve["recall"], curve["precision"], label=label)
        ax.set(xlim=(0,1), ylim=(0,1), xlabel="FPR" if name == "roc" else "Recall",
               ylabel="TPR" if name == "roc" else "Precision", title=name)
        ax.legend(fontsize=7)
        save_figure(fig, directory, name)


def error_examples(raw_images, indices, labels, probabilities, names, directory):
    wrong = np.flatnonzero(probabilities.argmax(1) != labels)
    if not len(wrong):
        return
    selected = wrong[np.argsort(-probabilities[wrong].max(1), kind="stable")[:16]]
    fig, axes = plt.subplots(4, 4, figsize=(11,10))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, position in zip(axes.ravel(), selected):
        ax.imshow(raw_images[indices[position]])
        pred = probabilities[position].argmax()
        ax.set_title(f"#{indices[position]} true: {names[labels[position]]}\npred: {names[pred]} ({probabilities[position,pred]:.2f})", fontsize=8)
    save_figure(fig, directory, "high_confidence_errors")


def build_pdf(output, title, paragraphs, tables, images):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    story = [Paragraph(escape(title), styles["Title"]), Spacer(1,12)]
    for text in paragraphs:
        story += [Paragraph(escape(str(text)), styles["BodyText"]), Spacer(1,8)]
    for heading, rows in tables:
        story += [Paragraph(escape(heading), styles["Heading2"])]
        cells = [[Paragraph(escape(str(cell)), styles["BodyText"]) for cell in row] for row in rows]
        table = Table(cells, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
                                   ("GRID",(0,0),(-1,-1),.3,colors.grey),
                                   ("VALIGN",(0,0),(-1,-1),"TOP")]))
        story += [table, Spacer(1,12)]
    for path in images:
        story += [PageBreak(), Paragraph(escape(path.stem), styles["Heading2"])]
        image = Image(str(path))
        factor = min(6.5*inch/image.imageWidth, 8*inch/image.imageHeight)
        image.drawWidth, image.drawHeight = image.imageWidth*factor, image.imageHeight*factor
        story.append(image)
    SimpleDocTemplate(str(output), rightMargin=36,leftMargin=36,topMargin=36,bottomMargin=36).build(story)


def export_evaluation(directory, metrics, logits, labels, indices, probabilities,
                      names, metadata, history_path=None, raw_images=None):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite evaluation: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    figures = directory / "figures"
    figures.mkdir()
    order = np.argsort(-probabilities, axis=1, kind="stable")[:, :min(5,len(names))]
    np.savez_compressed(directory / "predictions.npz", logits=logits, labels=labels,
                        indices=indices, probabilities=probabilities, class_names=np.asarray(names))
    predictions = pd.DataFrame({"dataset_index": indices, "true_label": labels,
                                "prediction": probabilities.argmax(1), "confidence": probabilities.max(1),
                                "correct": probabilities.argmax(1) == labels})
    for i in range(order.shape[1]):
        predictions[f"top{i+1}_class"] = order[:,i]
        predictions[f"top{i+1}_probability"] = probabilities[np.arange(len(labels)),order[:,i]]
    predictions.to_csv(directory / "predictions.csv", index=False)
    predictions.loc[~predictions.correct].sort_values("confidence", ascending=False).to_csv(directory / "errors.csv", index=False)
    pd.DataFrame(metrics["per_class"]).to_csv(directory / "per_class.csv", index=False)
    pd.DataFrame(metrics["classification_report"]).T.to_csv(directory / "classification_report.csv")
    for field in ("confusion_matrix", "confusion_matrix_normalized"):
        pd.DataFrame(metrics[field], index=names, columns=names).to_csv(directory / f"{field}.csv")
    pd.DataFrame([metrics["summary"]]).to_csv(directory / "summary.csv", index=False)
    pd.DataFrame(metrics["calibration"]).to_csv(directory / "calibration.csv", index=False)
    evaluation_figures(metrics, probabilities, labels, figures)
    if history_path:
        training_figures(history_path, figures)
    if raw_images is not None:
        error_examples(raw_images, indices, labels, probabilities, names, figures)
    result = {"metadata": metadata, **metrics}
    write_json(directory / "metrics.json", result)
    rows = [["Metric", "Value"]] + [[key, f"{value:.6g}" if isinstance(value, float) else str(value)]
                                    for key,value in metrics["summary"].items()]
    per_rows = [["Class", "Support", "Precision", "Recall", "F1", "AUC", "AP"]] + [
        [row["class_name"],row["support"]] + [f"{row[key]:.4f}" if row[key] is not None else "undefined"
                                             for key in ("precision","recall","f1","roc_auc_ovr","average_precision_ovr")]
        for row in metrics["per_class"]]
    readable_metadata = {key:value for key,value in metadata.items() if key not in ("config","profile","training_efficiency")}
    resources = [["Measurement", "Value"]]
    if "model_weights_bytes" in metadata:
        resources.append(["model-only weights bytes",metadata["model_weights_bytes"]])
    for key in ("learnable_parameters","macs","flops","convention"):
        if key in metadata.get("profile",{}):
            resources.append([key,metadata["profile"][key]])
    for heading in ("latency_batch1","throughput_batch128"):
        for key,value in metadata.get("profile",{}).get(heading,{}).items():
            resources.append([f"{heading}: {key}",value])
    for key,value in metadata.get("training_efficiency",{}).items():
        resources.append([key,value])
    build_pdf(directory / "evaluation_report.pdf", f"{metadata['dataset']} / {metadata['variant']} / {metadata['split']}",
              [readable_metadata, metrics["units"], metrics["notes"],
               "Validation reports are development results, not final official test performance. Training weighted accuracy may use augmentation/mixing; it is not deterministic hard-label training accuracy."],
              [("Metrics", rows), ("Measured efficiency/stability", resources),
               ("Per-class metrics", per_rows)], sorted(figures.glob("*.png")))
    return directory


def compare_prediction_files(path_a, path_b, output, metadata_a, metadata_b):
    a, b = np.load(path_a, allow_pickle=False), np.load(path_b, allow_pickle=False)
    for key in ("dataset", "split", "split_fingerprint"):
        if metadata_a.get(key) != metadata_b.get(key):
            raise ValueError(f"Cannot pair different evaluation {key}")
    if not np.array_equal(a["class_names"], b["class_names"]):
        raise ValueError("Class ordering mismatch")
    order_a, order_b = np.argsort(a["indices"]), np.argsort(b["indices"])
    if (not np.array_equal(a["indices"][order_a], b["indices"][order_b])
            or not np.array_equal(a["labels"][order_a], b["labels"][order_b])
            or len(np.unique(a["indices"])) != len(order_a)):
        raise ValueError("Cannot pair unmatched/duplicate image indices or labels")
    labels = a["labels"][order_a]
    result = paired_bootstrap(labels, a["logits"][order_a].argmax(1), b["logits"][order_b].argmax(1))
    write_json(output, {"a": metadata_a, "b": metadata_b, **result})
    return result


def aggregate_study(root):
    """Aggregate only metrics actually exported; explicitly label provisional progress."""
    root = Path(root)
    rows = []
    results = []
    for path in sorted(root.glob("runs/*/evaluation_validation/metrics.json")) + sorted(root.glob("runs/*/evaluation_test/metrics.json")):
        if not (path.parent / "complete.json").exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        results.append((path,result))
        meta = result["metadata"]
        profile = meta.get("profile",{})
        efficiency = meta.get("training_efficiency",{})
        rows.append({"run": path.parents[1].name, **{key: meta.get(key) for key in
                    ("dataset","variant","split","optimizer","lr","seed","epoch","configured_epochs","run_completed","full_train")},
                    **result["summary"], "learnable_parameters": meta.get("profile",{}).get("learnable_parameters"),
                    "flops": profile.get("flops"), "macs": profile.get("macs"),
                    "latency_batch1_median_ms":profile.get("latency_batch1",{}).get("batch_latency_median_ms"),
                    "latency_batch1_p95_ms":profile.get("latency_batch1",{}).get("batch_latency_p95_ms"),
                    "forward_batch128_images_per_second":profile.get("throughput_batch128",{}).get("images_per_second"),
                    "model_weights_bytes":meta.get("model_weights_bytes"), **efficiency,
                    "artifact_path": str(path.parent)})
    destination = root / "reports"
    destination.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "all_evaluations.csv", index=False)
    write_json(destination / "all_evaluations.json", rows)
    tables, images = [], []
    if len(frame):
        tables.append(("Measured evaluations", [["Run","Split","Top1 %","F1 macro","NLL"]] +
                       [[row.run,row.split,f"{row.top1_accuracy:.3f}",f"{row.f1_macro:.4f}",f"{row.nll:.4f}"] for row in frame.itertuples()]))
        tables.append(("Measured efficiency", [["Run", "Params", "GFLOPs", "B1 median ms", "Train epoch s"]] +
                       [[row.run,row.learnable_parameters,
                         f"{row.flops/1e9:.4f}" if pd.notna(row.flops) else "unavailable",
                         f"{row.latency_batch1_median_ms:.3f}" if pd.notna(row.latency_batch1_median_ms) else "unavailable",
                         getattr(row,"train_seconds_epoch_mean","unavailable")] for row in frame.itertuples()]))
        final = frame[(frame.split == "test") & frame.run_completed & (frame.full_train == True)]
        if len(final):
            group = ["dataset","variant","optimizer","lr","configured_epochs"]
            if final.duplicated(group+["seed"]).any():
                raise ValueError("Duplicate final seed in aggregation; do not inflate seed statistics")
            stats = final.groupby(group)[["top1_accuracy","f1_macro","nll"]].agg(["count","mean","std"])
            stats.to_csv(destination / "test_seed_statistics.csv")
        for dataset in frame.dataset.unique():
            selected = frame[(frame.dataset == dataset) & (frame.split == "validation") & frame.run_completed]
            if len(selected):
                fig, ax = plt.subplots(figsize=(10,5))
                ax.bar(np.arange(len(selected)), selected.top1_accuracy)
                ax.set_xticks(np.arange(len(selected)), selected.run, rotation=90, fontsize=6)
                ax.set(ylabel="Validation top1 (%)", title=f"{dataset}: completed runs only")
                save_figure(fig,destination,f"{dataset}_validation_comparison")
                images.append(destination / f"{dataset}_validation_comparison.png")
        # Compare matched architecture ablations only, not independently tuned winners.
        for path_a, result_a in results:
            a = result_a["metadata"]
            if a.get("variant") != "no_fusion" or not a.get("run_completed"):
                continue
            for path_b, result_b in results:
                b = result_b["metadata"]
                if b.get("variant") != "fusion" or not b.get("run_completed"):
                    continue
                if any(a.get(key) != b.get(key) for key in ("dataset","split","optimizer","lr","seed","configured_epochs")):
                    continue
                config_a, config_b = a.get("config",{}), b.get("config",{})
                ignored = {"variant","output_dir"}
                if any(config_a.get(key) != config_b.get(key) for key in (set(config_a)|set(config_b))-ignored):
                    continue
                pair = f"paired_{path_a.parents[1].name}__{path_b.parents[1].name}_{a['split']}.json"
                if not (destination/pair).exists():
                    compare_prediction_files(path_a.parent/"predictions.npz",path_b.parent/"predictions.npz",
                                             destination/pair,a,b)
    progress = []
    budget_path = root / "budget.json"
    if budget_path.exists():
        budget = json.loads(budget_path.read_text())
        progress.append(f"Measured budget: workers={budget['selected_workers']}; conservative epoch={budget['conservative_epoch_seconds']:.2f}s; grid with25% reserve={budget['grid_hours_with_reserve']:.1f}h. Laptop thermals/other jobs may alter timings.")
        tables.append(("Real-data smoke measurements (NOT final ranking)",
                       [["Dataset","Variant","Workers","Epoch seconds","Skipped AMP"]] +
                       [[row["dataset"],row["variant"],row["workers"],
                         ", ".join(f"{value:.2f}" for value in row["epoch_wall_seconds"]),
                         row.get("skipped_amp_steps","unavailable")] for row in budget["measurements"]]))
    status_path = root / "status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        progress.append(f"Status snapshot: {status.get('phase')}; {status.get('completed_grid_runs',0)}/{status.get('total_grid_runs',24)} grid runs completed. Current run: {status.get('current_run','none')}. Snapshot updated: {status.get('updated_utc','unknown')}.")
    build_pdf(destination / "comparison_report.pdf", "Two-model study - measured progress",
              ["PROVISIONAL until grid, recipe selection, final training and official test are complete.",
               "Validation is used for selection. Do not compare smoke/unfinished budgets as final performance. Test seeds must be predetermined, never selected by test accuracy.",
               f"Exported development/final evaluations: {len(rows)}. No results are inferred for missing runs."]+progress,tables,images)
    return destination


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rebuild provisional study CSV/JSON/PDF from completed exports")
    parser.add_argument("--study-root", required=True)
    args = parser.parse_args()
    print(aggregate_study(args.study_root))