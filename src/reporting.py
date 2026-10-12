"""Tables and a PDF report drawn from measured results, with missing scores blank."""
import csv
import json
import os
from pathlib import Path

from .config import ROOT
from .engine import atomic_write
from .models import build_model, profile_model

COLUMNS = ['experiment_id', 'dataset', 'model_name', 'architecture_version', 'seed', 'parameters',
           'flops', 'optimizer', 'learning_rate', 'weight_decay', 'scheduler', 'augmentation',
           'epochs', 'best_val_accuracy', 'final_val_accuracy', 'test_accuracy', 'training_time',
           'checkpoint', 'phase', 'synthetic', 'subset', 'notes']


def collect_results(roots):
    documents = {}
    for root in roots:
        root = Path(root)
        for path in root.rglob('result.json') if root.exists() else []:
            data = json.loads(path.read_text(encoding='utf-8'))
            documents[str(path.resolve())] = data
    return list(documents.values())


def score(value):
    return '' if value is None else f'{value:.2f}'


def build_reports(roots=None, report_root=None, result_path=None):
    roots = roots or [ROOT / 'checkpoints', ROOT / 'experiments' / 'smoke', ROOT / 'experiments' / 'verification']
    report_root = Path(report_root or ROOT / 'reports' / 'generated')
    result_path = Path(result_path or ROOT / 'experiments' / 'results' / 'results.csv')
    report_root.mkdir(parents=True, exist_ok=True)
    results = collect_results(roots)
    def write_csv(stream):
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    atomic_write(result_path, write_csv)
    real = [r for r in results if not r['synthetic'] and not r['subset']]
    real.sort(key=lambda r: (r['dataset'], -(r['best_val_accuracy'] or -1)))
    lines = ['# Measured experiment leaderboard', '',
             'Selection uses validation only. Test is blank until frozen final retraining/evaluation.', '',
             '| Rank | Dataset | Model | Params | Val (%) | Test (%) | Optimizer/LR | Experiment |',
             '|---:|---|---|---:|---:|---:|---|---|']
    for i, r in enumerate(real, 1):
        lines.append(f'| {i} | {r["dataset"]} | {r["model_name"]} | {r["parameters"]} | {score(r["best_val_accuracy"])} | {score(r["test_accuracy"])} | {r["optimizer"]}/{r["learning_rate"]} | {r["experiment_id"]} |')
    if not real:
        lines += ['', 'No completed full-data research runs yet. No benchmark accuracy has been demonstrated.']
    lines += ['', f'Excluded synthetic/subset checks: {len(results)-len(real)}. They remain in the CSV for audit.']
    atomic_write(report_root / 'leaderboard.md', lambda stream: stream.write('\n'.join(lines)+'\n'))
    from fpdf import FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, 'CIFAR Final Examination - CompactMix-L', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', size=10)
    def paragraph(text):
        # ASCII makes the artifact portable without depending on global fonts.
        pdf.multi_cell(0, 5, text.encode('ascii', 'replace').decode(), new_x='LMARGIN', new_y='NEXT')
        pdf.ln(2)
    paragraph('Status: automatically generated from persisted experiment records. Accuracy targets (CIFAR-10 >=98%; CIFAR-100 >=90%) are research goals, not guaranteed or measured results.')
    paragraph('Exam constraints: <=6,000,000 learnable parameters and <1,000,000,000 inference FLOPs. RGB input 32x32. FLOP estimate uses 2 FLOPs/MAC plus normalization, activations, pooling, channel gates and residual additions. Memory-only operations are excluded.')
    for dataset, classes in [('cifar10', 10), ('cifar100', 100)]:
        stats = profile_model(build_model({'family': 'compactmix'}, classes))
        paragraph(f'{dataset}: {stats["parameters"]:,} parameters; {stats["flops"]/1e9:.6f} GFLOPs; {stats["macs"]/1e9:.6f} GMACs.')
    paragraph('Architecture: 3x3 stride-1 stem; stage resolutions 32/16/8; widths 40/112/256; depths 2/3/8. Each block expands channels, splits 3x3 and 5x5 depthwise paths, concatenates, mixes with an 8-group 3x3 convolution, applies SE, projects and adds a residual with stochastic depth. This is a custom composition of established components, not a claim of novel research or uniqueness among student groups.')
    paragraph('Protocol: stratified 45,000 training / 5,000 validation images (split seed 2026). All architecture, LR, optimizer, augmentation and best-weight decisions use validation. Freeze the winning recipe and epoch count, retrain from scratch on all 50,000 training images, then evaluate the official 10,000-image test once. No test-set tuning, pretrained weights, external teacher, ensemble or TTA.')
    paragraph('Recipe: 600 selection epochs; batch 128; seed 42; SGD LR 0.1, momentum 0.9, Nesterov, weight decay 0.0005. Adam option: LR 0.001, betas 0.9/0.999, epsilon 1e-8, weight decay 0.0001. Norm/bias parameters have zero decay. Warmup then cosine to 0.1% LR, label smoothing 0.05, crop/flip/AutoAugment, Mixup alpha 0.2 and CutMix alpha 1.0 with probability 0.75; mixing disabled for final min(30, epochs//10) epochs. Gradient clipping 5; adaptive EMA capped at 0.999; CUDA float16 AMP; deterministic mode by default.')
    paragraph('Required optimizer/LR comparison: --matrix runs SGD 0.05/0.1/0.15 and Adam 0.0003/0.001/0.003 independently for each dataset. Report every run and choose by validation. This comparison is not yet measured unless full-data records below demonstrate it.')
    paragraph(f'Completed full-data experiments: {len(real)}. Synthetic/subset technical checks excluded: {len(results)-len(real)}.')
    for r in real:
        paragraph(f'{r["experiment_id"]}: dataset={r["dataset"]}, phase={r["phase"]}, params={r["parameters"]}, FLOPs={r["flops"]}, best validation={score(r["best_val_accuracy"]) or "not used"}%, test={score(r["test_accuracy"]) or "not evaluated"}%, time={r["training_time"]:.1f}s. Recipe: {json.dumps(r["config"], sort_keys=True)}')
        paragraph(f'Reproducibility: {json.dumps(r["metadata"], sort_keys=True)}')
    paragraph('Ablation plan: compare the baseline, local 3x3/3x3 variant, no gate, no Mixup/CutMix, no EMA and no DropPath under equal budgets and seeds. No successful/failed accuracy hypothesis can be declared before those experiments. Repeat finalists on multiple seeds and report mean/std rather than only the best seed.')
    paragraph('Oral exam preparation: explain depthwise vs grouped convolution, residual gradient flow, receptive fields, SE, parameter/FLOP arithmetic, SGD momentum vs Adam moments, cosine warmup, regularization, mixed-label training accuracy, validation/test leakage and exactly what is restored from a checkpoint.')
    paragraph('Sources: Final exam.docx; github.com/akira-n-28/convnext-muon-cifar10 (README claims 98.10%, 12.37M parameters; not independently verified and over this exam budget); github.com/huggingface/pytorch-image-models; arxiv.org/abs/1512.03385; arxiv.org/abs/1709.01507; arxiv.org/abs/1710.09412; arxiv.org/abs/1905.04899; docs.pytorch.org/docs/stable/notes/randomness.html.')
    atomic_write(report_root / 'final_report.pdf', lambda stream: stream.write(bytes(pdf.output())), binary=True)
    print(f'REPORT PDF={report_root / "final_report.pdf"} CSV={result_path}', flush=True)
    return results
