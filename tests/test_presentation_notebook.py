"""Check the delivered notebook without depending on hidden project imports."""
import ast
import copy
from pathlib import Path

import nbformat
import pytest
import torch
from torch import nn


NOTEBOOK = Path(__file__).resolve().parents[1]/"report/CompactMix_CIFAR10_CIFAR100_BAO_CAO.ipynb"


@pytest.fixture
def notebook_namespace(tmp_path):
    notebook=nbformat.read(NOTEBOOK,as_version=4)
    ns={"__name__":"notebook_test"}
    bootstrap=next(c.source for c in notebook.cells if c.get("id")=="bootstrap")
    tree=ast.parse(bootstrap)
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom)) or (
                isinstance(node,ast.FunctionDef) and node.name=="ensure_python"):
            exec(compile(ast.Module(body=[node],type_ignores=[]),"bootstrap","exec"),ns)
    ns["ensure_python"]()
    ns.update(ROOT=NOTEBOOK.parents[1],REPORT_DIR=NOTEBOOK.parent,
              DATA_ROOT=NOTEBOOK.parents[1]/"datasets",OUTPUT_ROOT=tmp_path,
              CHECKPOINT_VERSION=1,REQUEST_STOP=False,NOTEBOOK_TRAINING_CELLS={"bootstrap":bootstrap},
              MODE="demo",DATASETS=("cifar10","cifar100"),RUN_NAME="test_notebook",
              DEVICE="cpu",DOWNLOAD=False,SYNTHETIC_CHECK=True,
              DEMO_TOTAL_EPOCHS=2,DEMO_TRAIN_IMAGES=32,DEMO_VAL_IMAGES=16,FULL_EPOCHS=600)
    settings=next(c.source for c in notebook.cells if c.get("id")=="settings")
    for node in ast.parse(settings).body:
        if isinstance(node,ast.Assign) and any(
                isinstance(target,ast.Name) and target.id in {"TRAIN_SETTINGS","STATS"}
                for target in node.targets):
            exec(compile(ast.Module(body=[node],type_ignores=[]),"settings","exec"),ns)
    ns["TRAIN_SETTINGS"].update(workers=0,tensorboard=False)
    for cell in notebook.cells:
        if cell.cell_type!="code" or not cell.source.startswith("_remember_cell("):
            continue
        ns["_remember_cell"]=lambda tag,source=cell.source: ns["NOTEBOOK_TRAINING_CELLS"].update({tag:source})
        exec(compile(cell.source,cell.id,"exec"),ns)
    for cell_id in ("data-preview","archived-evidence","archive-weight-adapter","cumulative-block-table"):
        cell=next(c for c in notebook.cells if c.get("id")==cell_id)
        for node in ast.parse(cell.source).body:
            if isinstance(node,ast.FunctionDef):
                exec(compile(ast.Module(body=[node],type_ignores=[]),cell_id,"exec"),ns)
    torch.set_num_threads(2)
    return ns


def test_notebook_is_valid_compilable_and_has_no_project_imports():
    import sys
    notebook=nbformat.read(NOTEBOOK,as_version=4)
    nbformat.validate(notebook)
    allowed=set(sys.stdlib_module_names)|{"torch","torchvision","numpy","pandas","matplotlib","IPython","reportlab","PIL"}
    code_cells=[c for c in notebook.cells if c.cell_type=="code"]
    assert len(code_cells)>=35
    for cell in code_cells:
        compile(cell.source,cell.id,"exec")
        for node in ast.walk(ast.parse(cell.source)):
            if isinstance(node,ast.Import):
                assert all(alias.name.split(".")[0] in allowed for alias in node.names)
            elif isinstance(node,ast.ImportFrom):
                assert node.level==0 and node.module.split(".")[0] in allowed
    joined="\n".join(c.source for c in code_cells)
    assert "pip install" not in joined and "venv" not in joined
    assert 'RECHECK_ARCHIVED_TEST = False' in joined and 'RUN_FINAL = False' in joined


@pytest.mark.parametrize("dataset,parameters,flops",[
    ("cifar10",5_248_172,842_406_944),("cifar100",5_271_302,842_453_114)])
def test_archived_weights_map_to_explicit_blocks_without_changing_outputs(notebook_namespace,dataset,parameters,flops):
    ns=notebook_namespace
    from src.models import build_model
    classes=10 if dataset=="cifar10" else 100
    bundle=ns["archived_bundle"](dataset)
    ns["ARCHIVED"]={dataset:bundle}
    actual=ns["load_archived_model"](dataset)
    reference=build_model({"family":"compactmix"},classes).eval()
    checkpoint=ns["load_checkpoint"](bundle["checkpoint"])
    reference.load_state_dict(checkpoint["model"],strict=True)
    images=torch.randn(2,3,32,32)
    with torch.no_grad():
        torch.testing.assert_close(actual(images),reference(images),rtol=0,atol=0)
    profile=ns["inspect_model"](actual,classes,backward=False)
    assert profile["parameters"]==parameters and profile["flops"]==flops


@pytest.mark.parametrize("optimizer",["sgd","adam"])
def test_notebook_resume_preserves_rng_optimizer_and_rejects_edited_cell(notebook_namespace,tmp_path,capsys,optimizer):
    ns=notebook_namespace
    def tiny(classes):
        return nn.Sequential(nn.Conv2d(3,8,3,padding=1,bias=False),nn.BatchNorm2d(8),
                             nn.SiLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(.2),nn.Linear(8,classes))
    ns["make_model"]=tiny
    config=ns["notebook_configs"](optimizer=optimizer)[0]
    config["batch_size"]=8
    partial=ns["run_experiment"](config,stop_after_epochs=1)
    assert partial["status"]=="interrupted" and len(capsys.readouterr().out.splitlines())==1
    directory=Path(config["run_dir"])
    before=(directory/"last.pt").read_bytes()
    original=ns["NOTEBOOK_TRAINING_CELLS"]["block01"]
    ns["NOTEBOOK_TRAINING_CELLS"]["block01"]=original+"\n# unsaved notebook edit"
    with pytest.raises(ValueError,match="source/version"):
        ns["run_experiment"](config)
    assert (directory/"last.pt").read_bytes()==before
    ns["NOTEBOOK_TRAINING_CELLS"]["block01"]=original
    resumed=ns["run_experiment"](config)
    assert len(capsys.readouterr().out.splitlines())==1
    whole_config=copy.deepcopy(config); whole_config["run_dir"]=str(tmp_path/"whole")
    whole=ns["run_experiment"](whole_config)
    assert resumed["history"]==whole["history"]
    last=ns["load_checkpoint"](directory/"last.pt")
    whole_last=ns["load_checkpoint"](Path(whole_config["run_dir"])/"last.pt")
    for key in ("model","ema","best_weights"):
        assert all(torch.equal(value,whole_last[key][name]) for name,value in last[key].items())
    for parameter_id,state in last["optimizer"]["state"].items():
        for name,value in state.items():
            other=whole_last["optimizer"]["state"][parameter_id][name]
            assert torch.equal(value,other) if torch.is_tensor(value) else value==other
    assert {p.name for p in directory.glob("*.pt")}=={"best.pt","last.pt"}
    assert resumed["synthetic"] is True and resumed["test_accuracy"] is None
    with pytest.raises(ValueError,match="smoke/subset"):
        ns["finalize"](config["run_dir"])


def test_inconsistent_conv_is_reported_before_training(notebook_namespace):
    ns=notebook_namespace
    model=ns["make_model"](10)
    model.block01.context=nn.Conv2d(40,40,7,padding=2,groups=40,bias=False)
    with pytest.raises(ns["ShapeError"],match="Block01.concat"):
        ns["inspect_model"](model,10)


def test_exam_export_matches_docx_and_keeps_unmeasured_results_blank(notebook_namespace,tmp_path):
    import json
    import pymupdf
    ns=notebook_namespace
    notebook=nbformat.read(NOTEBOOK,as_version=4)
    for cell_id in ("optimizer-matrix","export-exam-report"):
        cell=next(c for c in notebook.cells if c.id==cell_id)
        for node in ast.parse(cell.source).body:
            if isinstance(node,ast.FunctionDef) or (isinstance(node,ast.Assign) and any(
                    isinstance(t,ast.Name) and t.id=="MATRIX" for t in node.targets)):
                exec(compile(ast.Module(body=[node],type_ignores=[]),cell_id,"exec"),ns)
    archived={name:ns["archived_bundle"](name) for name in ns["DATASETS"]}
    ns["ARCHIVED"]=archived
    ns["PROFILES"]={name:ns["inspect_model"](ns["make_model"](10 if name=="cifar10" else 100),
                    10 if name=="cifar10" else 100,backward=False) for name in ns["DATASETS"]}
    rows=ns["exam_comparison_rows"]()
    assert len(rows)==12
    assert sum(row["test_percent"] is not None for row in rows)==2
    assert all(row["validation_percent"] is None and row["test_percent"] is None
               for row in rows if row["optimizer"]=="adam")
    ns["archived_bundle"]=lambda name:archived[name]
    ns["REPORT_DIR"]=tmp_path
    files=ns["export_exam_report"]()
    audit=json.loads(Path(files["audit"]).read_text())
    assert audit["complete_optimizer_lr_evidence"] is False
    assert "multiple seeds" in audit["project_extensions_not_explicit_exam_requirements"]
    assert "50k retraining" in audit["project_extensions_not_explicit_exam_requirements"]
    assert "SGD and Adam with different initial LRs" in audit["explicit_exam_requirements"]
    with pymupdf.open(files["pdf"]) as pdf:
        text="\n".join(page.get_text() for page in pdf)
    for token in ("CHƯA HOÀN THÀNH","97.18","82.98","not over 6M","under 1G","Adam","0.15"):
        assert token in text,token
    assert Path(files["markdown"]).exists() and Path(files["comparison_csv"]).exists()


@pytest.mark.parametrize("classes,parameters,macs,flops",[
    (10,5_248_172,409_492_864,842_406_944),(100,5_271_302,409_515_904,842_453_114)])
def test_cumulative_accounting_counts_each_block_once_and_matches_profile(notebook_namespace,classes,parameters,macs,flops):
    ns=notebook_namespace
    model=ns["make_model"](classes)
    rows,profile=ns["cumulative_model_accounting"](model,classes)
    assert len(rows)==18 and model.training is True
    leaves=[leaf for row in rows for leaf in row["leaves"]]
    assert len(leaves)==len({leaf["name"] for leaf in leaves})==261
    for key,total in (("parameters",parameters),("macs",macs),("flops",flops)):
        assert sum(row[key] for row in rows)==rows[-1][key+"_cumulative"]==profile[key]==total
        assert [row[key+"_cumulative"] for row in rows]==list(__import__('itertools').accumulate(row[key] for row in rows))
    assert rows[0]["input"]=="3×32×32" and rows[0]["output"]=="40×32×32"
    assert rows[0]["parameters"]==1080 and rows[0]["macs"]==1_105_920
    assert next(row for row in rows if row["block"]=="head")["macs"]==0
    for block_name,c,d,h in (("block01",40,8,32),("block03",112,14,16),("block06",256,32,8)):
        row=next(row for row in rows if row["block"]==block_name)
        e=2*c
        assert row["parameters"]==2*c*e+34*c+9*e*e//8+2*e*d+d+e+2*(c+3*e)
        assert row["macs"]==h*h*(2*c*e+34*c+9*e*e//8)+2*e*d
        assert row["calculation"].startswith("B(")
    assert rows[-1]["parameters"]==256*classes+classes and rows[-1]["macs"]==256*classes
