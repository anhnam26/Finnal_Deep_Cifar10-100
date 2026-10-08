import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluate_full import evaluate_checkpoint
from metrics import classification_metrics, paired_bootstrap, wilson_interval
from prepare_study import prepare
from reporting import aggregate_study, compare_prediction_files, export_evaluation
from study import measure_budget, run_queue
from utils import load_config


def test_amp_persistent_overflow_fails_before_checkpoint():
    import torch
    import train
    from data import make_loader
    from torch import nn
    from torch.utils.data import TensorDataset
    from utils import DEFAULT_CONFIG
    class SkippingScaler:
        scale_value=65536
        def is_enabled(self):
            return False
        def get_scale(self):
            return self.scale_value
        def scale(self,loss):
            return loss
        def step(self,optimizer):
            pass
        def update(self):
            self.scale_value/=2
    model=nn.Sequential(nn.Flatten(),nn.Linear(3072,10))
    config={**DEFAULT_CONFIG,"epochs":2,"warmup_epochs":0,"amp":False}
    optimizer=train.build_optimizer(model,config)
    loader=make_loader(TensorDataset(torch.zeros(2,3,32,32),torch.zeros(2,dtype=torch.long)),2,0)
    with pytest.raises(FloatingPointError,match="AMP skipped"):
        train.train_epoch(model,loader,optimizer,SkippingScaler(),config,torch.device("cpu"),0)


def test_checkpoint_evaluation_uses_validation_only(tmp_path, monkeypatch):
    import evaluate_full
    import torch
    from torch import nn
    from data import split_fingerprint, stratified_split
    from utils import DEFAULT_CONFIG
    calls=[]
    class FakeCIFAR:
        classes=[f"class{i}" for i in range(10)]
        def __init__(self, root, train, download, transform):
            calls.append(train)
            assert train and not download
            self.targets=np.repeat(np.arange(10),501).tolist()
            self.data=np.zeros((5010,32,32,3),dtype=np.uint8)
            self.transform=transform
        def __len__(self):
            return len(self.targets)
        def __getitem__(self,index):
            return torch.zeros(3,32,32), self.targets[index]
    def make_model(classes,variant):
        return nn.Sequential(nn.Flatten(),nn.Linear(3072,classes))
    monkeypatch.setitem(evaluate_full.DATASETS,"cifar10",(FakeCIFAR,10))
    monkeypatch.setattr(evaluate_full,"LightNet",make_model)
    monkeypatch.setattr(evaluate_full,"profile_operations",lambda model:{"learnable_parameters":30730,"flops":61450})
    received={}
    def export(destination,metrics,logits,labels,indices,probabilities,names,metadata,*args):
        received.update({"indices":indices,"labels":labels,"metadata":metadata})
        destination.mkdir()
        (destination/"metrics.json").write_text(json.dumps({"metadata":metadata,**metrics}))
    monkeypatch.setattr(evaluate_full,"export_evaluation",export)
    labels=np.repeat(np.arange(10),501)
    training,validation=stratified_split(labels,10,500,2026)
    config={**DEFAULT_CONFIG,"workers":0,"epochs":2,"data_root":str(tmp_path),"full_train":False}
    state={"config":config,"epoch":1,"model":make_model(10,"baseline").state_dict(),
           "optimizer":{},"split_fingerprint":split_fingerprint(training,validation)}
    checkpoint=tmp_path/"best.pt"
    torch.save(state,checkpoint)
    (tmp_path/"summary.json").write_text('{"completed_epochs":2}')
    summary=evaluate_checkpoint(checkpoint,tmp_path/"export",device_name="cpu",measure_cost=False)
    assert summary["samples"] == 5000
    assert calls == [True]
    assert np.array_equal(received["indices"],validation)
    assert received["metadata"]["run_completed"]
    assert (tmp_path/"export"/"model_weights.pt").exists()
    assert (tmp_path/"export"/"complete.json").exists()
    state["config"]["full_train"]=True
    torch.save(state,checkpoint)
    with pytest.raises(ValueError,match="held-out"):
        evaluate_checkpoint(checkpoint,tmp_path/"no-val",device_name="cpu",measure_cost=False)
    state["epoch"]=0
    torch.save(state,checkpoint)
    with pytest.raises(ValueError,match="epoch budget"):
        evaluate_checkpoint(checkpoint,tmp_path/"no-test",split="test",confirm_final_test=True,measure_cost=False)
    state["config"]["full_train"]=False
    state["config"]["max_train_batches"]=1
    torch.save(state,checkpoint)
    with pytest.raises(ValueError,match="debug"):
        evaluate_checkpoint(checkpoint,tmp_path/"no-debug",measure_cost=False)


def fixture_metrics():
    logits = np.array([[4,1,0], [1,4,0], [4,1,0], [0,1,4], [0,4,1], [1,0,4]],dtype=float)
    labels = np.array([0,1,2,2,1,0])
    names = ["a","b","c"]
    result, probabilities = classification_metrics(logits,labels,names)
    return result, probabilities, logits, labels, names


def test_metrics_known_counts():
    result, probabilities, logits, labels, names = fixture_metrics()
    summary = result["summary"]
    assert summary["top1_accuracy"] == pytest.approx(100*4/6)
    assert summary["top5_accuracy"] == 100
    assert summary["balanced_accuracy"] == pytest.approx(summary["top1_accuracy"])
    assert summary["f1_micro"] == pytest.approx(4/6)
    assert np.array_equal(result["confusion_matrix"],[[1,0,1],[0,2,0],[1,0,1]])
    assert np.allclose(probabilities.sum(1),1)
    expected_nll = -np.log(probabilities[np.arange(len(labels)), labels]).mean()
    assert summary["nll"] == pytest.approx(expected_nll)
    assert summary["brier_sum_over_classes"] == pytest.approx(((probabilities-np.eye(3)[labels])**2).sum(1).mean())
    assert sum(row["count"] for row in result["calibration"]) == 6
    assert summary["ece_15"] == pytest.approx(abs(probabilities.max(1).mean() - 4/6))
    assert summary["auc_defined_classes"] == 3
    assert summary["roc_auc_ovr_micro"] >= 0
    assert "macro_roc" in result["curves"]
    low, high = summary["accuracy_wilson_95_percent"]
    assert low < summary["top1_accuracy"] < high


def test_missing_class_and_extreme_logits():
    result, probabilities = classification_metrics(np.array([[10000,0,-10000],[0,10000,-10000]]),
                                                   np.array([0,1]),["a","b","missing"])
    assert np.isfinite(result["summary"]["nll"])
    assert result["per_class"][2]["roc_auc_ovr"] is None
    assert result["per_class"][2]["average_precision_ovr"] is None
    assert result["summary"]["auc_defined_classes"] == 2
    assert result["summary"]["balanced_accuracy"] == 100
    assert result["summary"]["ece_15"] == 0
    assert result["summary"]["top5_accuracy"] == 100


@pytest.mark.parametrize("logits,labels", [(np.ones((0,3)),np.array([],dtype=int)),
                                         (np.ones((2,3)),np.array([0,3])),
                                         (np.array([[0,np.nan,0]]),np.array([0])),
                                         (np.ones((2,3)),np.array([.1,1.]))])
def test_invalid_metrics(logits,labels):
    with pytest.raises(ValueError):
        classification_metrics(logits,labels,["a","b","c"])


def test_paired_bootstrap_and_wilson():
    labels=np.array([0,1,2,0])
    a=np.array([0,1,0,0])
    b=np.array([0,1,2,0])
    result=paired_bootstrap(labels,a,b)
    assert result["delta_b_minus_a_percentage_points"] == 25
    assert result["a_wrong_b_correct"] == 1
    assert result == paired_bootstrap(labels,a,b)
    assert paired_bootstrap(labels,a,a)["paired_bootstrap_95_percent"] == [0,0]
    assert wilson_interval(0,100)[0] == pytest.approx(0,abs=1e-12)
    assert wilson_interval(100,100)[1] == pytest.approx(100)


def test_export_and_aggregate(tmp_path):
    result,probabilities,logits,labels,names=fixture_metrics()
    destination=tmp_path/"runs"/"demo"/"evaluation_validation"
    metadata={"dataset":"cifar10","variant":"baseline","split":"validation",
              "split_fingerprint":"fixed", "run_completed":True, "epoch":2,
              "configured_epochs":2, "seed":42,"optimizer":"sgd","lr":.1}
    export_evaluation(destination,result,logits,labels,np.arange(6),probabilities,names,metadata)
    # Incomplete export not aggregated; queue publishes marker after all files.
    aggregate_study(tmp_path)
    assert json.loads((tmp_path/"reports"/"all_evaluations.json").read_text()) == []
    (destination/"complete.json").write_text('{"complete":true}')
    aggregate_study(tmp_path)
    assert len(pd.read_csv(tmp_path/"reports"/"all_evaluations.csv")) == 1
    assert (destination/"evaluation_report.pdf").read_bytes().startswith(b"%PDF-")
    assert (tmp_path/"reports"/"comparison_report.pdf").stat().st_size > 1000
    predictions=np.load(destination/"predictions.npz",allow_pickle=False)
    assert np.array_equal(predictions["indices"],np.arange(6))
    assert len(pd.read_csv(destination/"predictions.csv")) == 6
    assert (destination/"figures"/"reliability.png").exists()
    with pytest.raises(FileExistsError):
        export_evaluation(destination,result,logits,labels,np.arange(6),probabilities,names,metadata)
    pair=compare_prediction_files(destination/"predictions.npz",destination/"predictions.npz",
                                  tmp_path/"paired.json",metadata,metadata)
    assert pair["delta_b_minus_a_percentage_points"] == 0
    with pytest.raises(ValueError,match="dataset"):
        compare_prediction_files(destination/"predictions.npz",destination/"predictions.npz",
                                  tmp_path/"bad.json",metadata,{**metadata,"dataset":"cifar100"})


def test_variant_grid_and_no_overwrite(tmp_path):
    manifest_path=prepare(tmp_path/"configs",tmp_path/"outputs")
    manifest=json.loads(manifest_path.read_text())
    assert len(manifest["grid"]) == 2
    assert len(manifest["smoke"]) == 4
    configs=[load_config(path) for path in manifest["grid"]]
    assert len({(c["dataset"],c["variant"],c["optimizer"],c["lr"]) for c in configs}) == 2
    assert all(c["epochs"] == 150 and c["split_seed"] == 2026 and c["seed"] == 42
               and c["max_train_batches"] is None and c["max_val_batches"] is None for c in configs)
    assert all(not c["full_train"] for c in configs)
    with pytest.raises(FileExistsError):
        prepare(tmp_path/"configs",tmp_path/"outputs")


def test_evaluation_gate_before_reading_checkpoint(tmp_path):
    with pytest.raises(ValueError,match="confirm-final-test"):
        evaluate_checkpoint(tmp_path/"missing.pt",tmp_path/"test",split="test")
    assert not (tmp_path/"test").exists()


def test_smoke_budget_counts_and_workers(tmp_path):
    manifest=json.loads(prepare(tmp_path/"configs",tmp_path/"outputs").read_text())
    for path in manifest["smoke"]:
        config=load_config(path)
        output=Path(config["output_dir"])
        output.mkdir(parents=True)
        (output/"summary.json").write_text('{"completed_epochs":2}')
        pd.DataFrame([{"epoch":e,"train_samples":45000,"validation_samples":5000,
                       "train_loss":1.,"validation_loss":1.,"skipped_amp_steps":0,
                       "images_per_second":1000,"peak_allocated_bytes":100,
                       "peak_reserved_bytes":200} for e in (1,2)]).to_csv(output/"metrics.csv",index=False)
        pd.DataFrame([{"epoch":e,"epoch_wall_seconds":20 if config["workers"] == 2 else 40}
                      for e in (1,2)]).to_csv(output/"timing.csv",index=False)
    budget=measure_budget(manifest)
    assert budget["selected_workers"] == 0
    assert budget["grid_hours_with_reserve"] == pytest.approx(40*300/3600*1.25)
    c100_only = {**manifest, "smoke":[p for p in manifest["smoke"] if load_config(p)["dataset"] == "cifar100"]}
    assert measure_budget(c100_only)["selected_workers"] == 0
    c10_only = {**manifest, "smoke":[p for p in manifest["smoke"] if load_config(p)["dataset"] == "cifar10"]}
    assert measure_budget(c10_only)["selected_workers"] == 0
    config=load_config(manifest["smoke"][0])
    file=Path(config["output_dir"])/"metrics.csv"
    frame=pd.read_csv(file)
    frame.loc[0,"skipped_amp_steps"]=1
    frame.to_csv(file,index=False)
    assert measure_budget(manifest)["measurements"][0]["skipped_amp_steps"] == [1,0]
    frame.loc[0,"skipped_amp_steps"]=4
    frame.to_csv(file,index=False)
    with pytest.raises(ValueError,match="AMP"):
        measure_budget(manifest)
    frame.loc[0,"skipped_amp_steps"]=0
    frame.loc[0,"train_samples"]=128
    frame.to_csv(file,index=False)
    with pytest.raises(ValueError,match="sample-count"):
        measure_budget(manifest)


def test_queue_lock_and_stop(tmp_path):
    path=prepare(tmp_path/"configs",tmp_path/"outputs")
    root=tmp_path/"outputs"
    root.mkdir()
    (root/"queue.lock").write_text('{"pid":1}')
    with pytest.raises(FileExistsError):
        run_queue(path)
    assert (root/"queue.lock").exists()
    (root/"queue.lock").unlink()
    (root/"STOP_AFTER_RUN").touch()
    run_queue(path)
    assert json.loads((root/"status.json").read_text())["phase"] == "stopped_between_runs"
    assert not (root/"queue.lock").exists()