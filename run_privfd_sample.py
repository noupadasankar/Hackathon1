#!/usr/bin/env python3
"""Reproducible PrivFD-IDS sampled N-BaIoT experiment.

Data: pinned community-prepared N-BaIoT sample mirror; not the full UCI archive.
No scores are hard-coded. All metrics and figures are created from executed runs.
"""
from __future__ import annotations
import io, os, re, json, time, hashlib, random, platform, sys, warnings
from pathlib import Path
from copy import deepcopy

import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
    matthews_corrcoef, precision_score, recall_score, confusion_matrix,
    average_precision_score, roc_auc_score, precision_recall_curve)
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from opacus import PrivacyEngine

REPO = "huylq/N-BaIoT-Sample"
DATA_COMMIT = "c06df3ace3c11fa88ad085185311bf9f797a8626"
API_URL = f"https://api.github.com/repos/{REPO}/contents/data/dataset20000?ref={DATA_COMMIT}"
ROOT = Path("privfd_outputs")
FIG = ROOT / "figures"
CACHE = Path("data_cache/nbaiot_sample")
ROOT.mkdir(exist_ok=True, parents=True); FIG.mkdir(exist_ok=True, parents=True); CACHE.mkdir(exist_ok=True, parents=True)
SEEDS = [0, 1, 2, 3, 4]
ROUNDS = 5
BATCH_SIZE = 256
LOCAL_EPOCHS = 1
LR = 0.025
TOP_K = 50
MAX_PER_ATTACK_CAPTURE = 250
ROWS_PER_CLASS_CLIENT = 1000
DELTA_DP = 1e-5
CLIP_NORM = 1.0
NOISES = [0.7, 1.0, 1.5]
MALICIOUS_CLIENTS_BY_SEED = {}  # filled after client names are known
torch.set_num_threads(1)
DEVICE = torch.device("cpu")

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def parse_name(name):
    m = re.match(r"^(.+?)_(\d+)sample_(.+)\.csv$", name)
    if not m: return None
    return m.group(1), m.group(2), m.group(3).lower()

def get_dataset_listing():
    r = requests.get(API_URL, timeout=60)
    r.raise_for_status()
    items = r.json()
    return [x for x in items if x.get("type") == "file" and x.get("name","").endswith(".csv") and parse_name(x["name"])]

def read_capture(item, seed):
    path = CACHE / item["name"]
    if not path.exists() or path.stat().st_size != int(item.get("size", -1)):
        url = f"https://raw.githubusercontent.com/{REPO}/{DATA_COMMIT}/{item['path']}"
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        content = r.content
        if len(content) != int(item["size"]):
            raise RuntimeError(f"Size mismatch for {item['name']}: expected {item['size']} got {len(content)}")
        path.write_bytes(content)
    df = pd.read_csv(path)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    # Keep columns containing at least one finite numeric value; the dataset should have 115 features.
    df = df.dropna(axis=1, how="all")
    if len(df.columns) != 115:
        raise RuntimeError(f"{item['name']} yielded {len(df.columns)} features, expected 115")
    df = df.fillna(df.median(numeric_only=True)).fillna(0.0)
    arr = df.to_numpy(dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    _, n, capture = parse_name(item["name"])
    label = 0 if capture == "benign" else 1
    cap = min(len(arr), MAX_PER_ATTACK_CAPTURE if label == 1 else ROWS_PER_CLASS_CLIENT)
    rng = np.random.default_rng(seed)
    if len(arr) > cap:
        idx = rng.choice(len(arr), size=cap, replace=False)
        arr = arr[np.sort(idx)]
    return arr, label

def load_sampled_clients():
    listing = get_dataset_listing()
    by_device = {}
    source_manifest = []
    for item in listing:
        parsed = parse_name(item["name"])
        if not parsed: continue
        device, _, capture = parsed
        by_device.setdefault(device, []).append((item, capture))
    expected = 9
    if len(by_device) != expected:
        raise RuntimeError(f"Expected 9 device clients, found {len(by_device)}: {sorted(by_device)}")
    clients = {}
    audit = []
    for di, device in enumerate(sorted(by_device)):
        benign_parts, attack_parts = [], []
        used = []
        for item, capture in sorted(by_device[device], key=lambda z: z[0]["name"]):
            # The mirror has one benign capture and multiple attack captures per device.
            arr, label = read_capture(item, seed=7000 + di * 113 + len(used))
            used.append(item["name"])
            source_manifest.append({"device": device, "file": item["name"], "git_blob_sha": item.get("sha"),
                "bytes": item.get("size"), "label": int(label), "rows_after_capture_cap": int(len(arr))})
            if label == 0: benign_parts.append(arr)
            else: attack_parts.append(arr)
        if not benign_parts or not attack_parts:
            raise RuntimeError(f"Client {device} lacks one of the two labels")
        benign = np.concatenate(benign_parts, axis=0)
        attack = np.concatenate(attack_parts, axis=0)
        rng = np.random.default_rng(15000 + di)
        n0 = min(ROWS_PER_CLASS_CLIENT, len(benign))
        n1 = min(ROWS_PER_CLASS_CLIENT, len(attack))
        if n0 < 500 or n1 < 500:
            raise RuntimeError(f"Not enough examples for {device}: benign={n0}, attack={n1}")
        benign = benign[np.sort(rng.choice(len(benign), size=n0, replace=False))]
        attack = attack[np.sort(rng.choice(len(attack), size=n1, replace=False))]
        X = np.vstack([benign, attack]).astype(np.float32)
        y = np.concatenate([np.zeros(n0, dtype=np.int64), np.ones(n1, dtype=np.int64)])
        perm = rng.permutation(len(y)); X, y = X[perm], y[perm]
        # A disjoint 10% calibration slice is reserved for local scaling and MI selection.
        X_dev, X_cal, y_dev, y_cal = train_test_split(X, y, test_size=0.10, random_state=900+di,
                                                        stratify=y)
        # Keep 15% test and 15% validation, leaving 60% train.
        X_rem, X_test, y_rem, y_test = train_test_split(X_dev, y_dev, test_size=1/6,
                                                        random_state=1000+di, stratify=y_dev)
        X_train, X_val, y_train, y_val = train_test_split(X_rem, y_rem, test_size=0.20,
                                                        random_state=1100+di, stratify=y_rem)
        scaler = StandardScaler().fit(X_cal)
        X_cal_s = np.nan_to_num(scaler.transform(X_cal), nan=0.0, posinf=0.0, neginf=0.0)
        scores = mutual_info_classif(X_cal_s, y_cal, random_state=di, n_neighbors=3)
        selected = np.argsort(scores)[-TOP_K:]
        mask = np.zeros(X.shape[1], dtype=np.float32); mask[selected] = 1.0
        def transform(a):
            z = scaler.transform(a)
            z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            z = np.clip(z, -10.0, 10.0).astype(np.float32)
            return z
        clients[device] = {
            "X_train": transform(X_train), "y_train": y_train.astype(np.int64),
            "X_val": transform(X_val), "y_val": y_val.astype(np.int64),
            "X_test": transform(X_test), "y_test": y_test.astype(np.int64),
            "X_cal": transform(X_cal), "y_cal": y_cal.astype(np.int64),
            "mask": mask, "files": used, "n0": int(n0), "n1": int(n1),
            "train_n": int(len(y_train)), "val_n": int(len(y_val)), "test_n": int(len(y_test)),
            "features_selected": int(mask.sum())
        }
        audit.append({"client": device, "source_files": len(used), "benign_selected": n0,
            "attack_selected": n1, "calibration_rows": len(y_cal), "train_rows": len(y_train),
            "validation_rows": len(y_val), "test_rows": len(y_test),
            "features_selected": int(mask.sum()), "train_benign": int((y_train==0).sum()),
            "train_attack": int((y_train==1).sum()), "test_benign": int((y_test==0).sum()),
            "test_attack": int((y_test==1).sum())})
        print(f"Loaded {device}: n={len(y)} train={len(y_train)} val={len(y_val)} test={len(y_test)}")
    pd.DataFrame(source_manifest).to_csv(ROOT / "source_file_manifest.csv", index=False)
    pd.DataFrame(audit).to_csv(ROOT / "client_data_audit.csv", index=False)
    (ROOT / "data_provenance.json").write_text(json.dumps({
        "dataset":"N-BaIoT sample mirror", "repository":f"https://github.com/{REPO}",
        "pinned_commit":DATA_COMMIT, "data_directory":"data/dataset20000",
        "device_clients":sorted(clients), "feature_count":115,
        "sampling":{"benign_rows_per_device":ROWS_PER_CLASS_CLIENT,
            "attack_rows_per_device":ROWS_PER_CLASS_CLIENT,
            "max_rows_per_attack_capture_before_within_device_sampling":MAX_PER_ATTACK_CAPTURE},
        "label_mapping":{"benign":0,"all_attack_captures":1},
        "split":{"calibration":0.10,"train":0.60,"validation":0.15,"test":0.15},
        "note":"This is a pinned, community-prepared sample mirror derived from N-BaIoT, not the complete official UCI archive."
    }, indent=2), encoding="utf-8")
    return clients

class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(115,128), nn.ReLU(), nn.Dropout(0.10),
                                 nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.10),
                                 nn.Linear(64,2))
    def forward(self, x): return self.net(x)

def core_model(m):
    return getattr(m, "_module", m)

def batch_xy(client, split, mi):
    X = client[f"X_{split}"]
    if mi: X = X * client["mask"][None,:]
    return X, client[f"y_{split}"]

def evaluate_probs(y, p):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y,pred)),
        "macro_f1": float(f1_score(y,pred,average="macro",zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y,pred)),
        "mcc": float(matthews_corrcoef(y,pred)),
        "precision_attack": float(precision_score(y,pred,pos_label=1,zero_division=0)),
        "recall_attack": float(recall_score(y,pred,pos_label=1,zero_division=0)),
        "fpr": float(fp/(fp+tn)) if tn+fp else float("nan"),
        "pr_auc": float(average_precision_score(y,p)) if len(np.unique(y))==2 else float("nan"),
        "roc_auc": float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }

@torch.no_grad()
def pred_prob(model, X, mi=False, mask=None, batch_size=4096):
    model.eval()
    if mi and mask is not None: X = X * mask[None,:]
    out=[]
    for i in range(0,len(X),batch_size):
        xt=torch.as_tensor(X[i:i+batch_size],dtype=torch.float32,device=DEVICE)
        out.extend(torch.softmax(model(xt),dim=1)[:,1].cpu().numpy().tolist())
    return np.asarray(out)

def pooled_evaluate(model, clients, mi):
    ys=[]; ps=[]; per=[]
    device_outputs={}
    for name,c in clients.items():
        X,y=batch_xy(c,"test",mi)
        p=pred_prob(model,X,False)
        met=evaluate_probs(y,p)
        per.append({"client":name,**met})
        ys.append(y); ps.append(p); device_outputs[name]={"y":y,"p":p}
    return evaluate_probs(np.concatenate(ys),np.concatenate(ps)), pd.DataFrame(per), device_outputs

def train_plain(model, X, y, epochs, seed, lr=0.025, batch_size=BATCH_SIZE):
    seed_all(seed)
    ds=TensorDataset(torch.as_tensor(X,dtype=torch.float32),torch.as_tensor(y,dtype=torch.long))
    loader=DataLoader(ds,batch_size=batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed))
    opt=torch.optim.SGD(model.parameters(),lr=lr)
    model.train()
    for _ in range(epochs):
        for xb,yb in loader:
            xb=xb.to(DEVICE); yb=yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(xb),yb)
            loss.backward(); opt.step()
    return model

def flattened(delta):
    return torch.cat([delta[k].reshape(-1).float() for k in sorted(delta)]).numpy()

def aggregate_deltas(deltas, aggregator, f=1):
    names=list(deltas)
    if aggregator=="fedavg":
        selected=names[:]
    elif aggregator=="multikrum":
        n=len(names)
        if n < 2*f+3: raise RuntimeError("Multi-Krum needs n >= 2f+3")
        vec=np.stack([flattened(deltas[nm]) for nm in names])
        sq=((vec[:,None,:]-vec[None,:,:])**2).sum(axis=2)
        neighbours=n-f-2
        scores=[]
        for i in range(n):
            d=np.delete(sq[i],i)
            scores.append(float(np.sort(d)[:neighbours].sum()))
        m=n-f-2
        selected=[names[i] for i in np.argsort(scores)[:m]]
    else: raise ValueError(aggregator)
    keys=list(deltas[names[0]])
    agg={k:torch.stack([deltas[n][k].float() for n in selected]).mean(0) for k in keys}
    return agg, selected

def train_local_update(global_model, client, config, seed, privacy_engine=None,
                       flip_labels=False):
    seed_all(seed)
    X,y=batch_xy(client,"train",config["mi"])
    y=y.copy()
    if flip_labels: y=1-y
    ds=TensorDataset(torch.as_tensor(X,dtype=torch.float32),torch.as_tensor(y,dtype=torch.long))
    loader=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True,
        generator=torch.Generator().manual_seed(seed),drop_last=False)
    model=Detector().to(DEVICE)
    base={k:v.detach().cpu().clone() for k,v in global_model.state_dict().items()}
    model.load_state_dict(base)
    teacher=deepcopy(global_model).to(DEVICE).eval() if config["kd"] else None
    opt=torch.optim.SGD(model.parameters(),lr=LR)
    if config["dp"]:
        model,opt,loader=privacy_engine.make_private(
            module=model,optimizer=opt,data_loader=loader,
            noise_multiplier=config["sigma"],max_grad_norm=CLIP_NORM,poisson_sampling=True)
    model.train()
    for _ in range(LOCAL_EPOCHS):
        for xb,yb in loader:
            if len(yb)==0: continue
            xb=xb.to(DEVICE); yb=yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits=model(xb)
            loss_each=F.cross_entropy(logits,yb,reduction="none")
            if teacher is not None:
                with torch.no_grad(): tlogits=teacher(xb)
                kd=F.kl_div(F.log_softmax(logits/2.0,dim=1),
                    F.softmax(tlogits/2.0,dim=1),reduction="none").sum(1)*4.0
                loss_each=loss_each+0.25*kd
            loss=loss_each.mean()
            loss.backward(); opt.step()
            if config["prox"]:
                with torch.no_grad():
                    core=core_model(model)
                    for nm,p in core.named_parameters():
                        p.add_(base[nm].to(DEVICE)-p,alpha=LR*0.01)
    final={k:v.detach().cpu().clone() for k,v in core_model(model).state_dict().items()}
    delta={k:final[k]-base[k] for k in base}
    del model,teacher,opt,loader
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return delta

def run_federated(clients, config, seed, scenario="clean", capture_preds=False):
    seed_all(seed)
    global_model=Detector().to(DEVICE)
    global_model.eval()
    client_names=list(clients)
    privacy_engines={name:PrivacyEngine(accountant="rdp") for name in client_names} if config["dp"] else {}
    attacker=client_names[(seed*2+2)%len(client_names)]
    per_round=[]; selection_rows=[]; privacy_rows=[]
    malicious_start=2
    for rnd in range(ROUNDS):
        deltas={}
        for name in client_names:
            attack_active=(scenario!="clean" and rnd>=malicious_start and name==attacker)
            delta=train_local_update(global_model,clients[name],config,seed*100000+rnd*100+client_names.index(name),
                privacy_engine=privacy_engines.get(name),flip_labels=(scenario=="label_flip" and attack_active))
            if scenario=="sign_flip" and attack_active:
                delta={k:(-2.0*v) for k,v in delta.items()}
            deltas[name]=delta
        agg,selected=aggregate_deltas(deltas,config["aggregator"],f=1)
        with torch.no_grad():
            for k,p in global_model.state_dict().items():
                p.copy_(p.cpu()+agg[k])
        rejected=[n for n in client_names if n not in selected]
        for name in client_names:
            selection_rows.append({"seed":seed,"model":config["name"],"scenario":scenario,"round":rnd+1,
                "client":name,"is_malicious":int(scenario!="clean" and name==attacker and rnd>=malicious_start),
                "selected":int(name in selected),"aggregator":config["aggregator"]})
            if config["dp"]:
                eps=float(privacy_engines[name].get_epsilon(delta=DELTA_DP))
                privacy_rows.append({"seed":seed,"model":config["name"],"scenario":scenario,"round":rnd+1,
                    "client":name,"epsilon":eps,"delta":DELTA_DP,"noise_multiplier":config["sigma"],
                    "max_grad_norm":CLIP_NORM,"train_records":clients[name]["train_n"],
                    "sample_rate_approx":min(1.0,BATCH_SIZE/max(1,clients[name]["train_n"]))})
        metrics,_,_=pooled_evaluate(global_model,clients,config["mi"])
        per_round.append({"seed":seed,"model":config["name"],"scenario":scenario,"round":rnd+1,**metrics,
            "selected_clients":";".join(selected),"rejected_clients":";".join(rejected),
            "malicious_client":attacker if scenario!="clean" else ""})
        print(f"[{scenario}] seed={seed} {config['name']} round={rnd+1}/{ROUNDS} macroF1={metrics['macro_f1']:.4f}")
    metrics, per_device, device_outputs=pooled_evaluate(global_model,clients,config["mi"])
    # Basic black-box confidence membership diagnostic; not a DP proof.
    mia_auc=None
    if capture_preds:
        member_scores=[]; nonmember_scores=[]
        for name,c in clients.items():
            Xm,_=batch_xy(c,"train",config["mi"]); Xt,_=batch_xy(c,"test",config["mi"])
            if config["mi"]:
                Xm=Xm*c["mask"][None,:]; Xt=Xt*c["mask"][None,:]
            def max_conf(X):
                vals=[]
                with torch.no_grad():
                    for i in range(0,len(X),1024):
                        logits=global_model(torch.as_tensor(X[i:i+1024],dtype=torch.float32))
                        vals.extend(torch.softmax(logits,dim=1).max(1).values.numpy().tolist())
                return vals
            member_scores.extend(max_conf(Xm)); nonmember_scores.extend(max_conf(Xt))
        try: mia_auc=float(roc_auc_score([1]*len(member_scores)+[0]*len(nonmember_scores),member_scores+nonmember_scores))
        except Exception: mia_auc=float("nan")
    else: mia_auc=float("nan")
    max_eps=max([x["epsilon"] for x in privacy_rows if x["seed"]==seed and x["model"]==config["name"] and x["scenario"]==scenario],default=float("nan"))
    final={**metrics,"seed":seed,"model":config["name"],"scenario":scenario,
        "aggregator":config["aggregator"],"dp":config["dp"],"noise_multiplier":config.get("sigma",None),
        "mia_confidence_auc":mia_auc,"max_client_epsilon":max_eps,
        "delta":DELTA_DP if config["dp"] else None,"selected_feature_count":TOP_K if config["mi"] else 115,
        "malicious_client":attacker if scenario!="clean" else ""}
    return final,per_round,selection_rows,privacy_rows,per_device,global_model,device_outputs

def make_configs():
    cfgs=[
        {"name":"FedAvg","mi":False,"prox":False,"kd":False,"dp":False,"aggregator":"fedavg"},
        {"name":"FD-IDS-style","mi":True,"prox":True,"kd":True,"dp":False,"aggregator":"fedavg"},
        {"name":"FD-IDS + DP-SGD","mi":True,"prox":True,"kd":True,"dp":True,"aggregator":"fedavg","sigma":1.0},
        {"name":"FD-IDS + Multi-Krum","mi":True,"prox":True,"kd":True,"dp":False,"aggregator":"multikrum"},
        {"name":"PrivFD-IDS sigma=0.7","mi":True,"prox":True,"kd":True,"dp":True,"aggregator":"multikrum","sigma":0.7},
        {"name":"PrivFD-IDS sigma=1.0","mi":True,"prox":True,"kd":True,"dp":True,"aggregator":"multikrum","sigma":1.0},
        {"name":"PrivFD-IDS sigma=1.5","mi":True,"prox":True,"kd":True,"dp":True,"aggregator":"multikrum","sigma":1.5},
    ]
    return cfgs

def make_figures(metrics_df, rounds_df, privacy_df, selection_df, clients, full_preds, full_model, full_seed):
    # Figure 1: dataset audit
    audit=pd.read_csv(ROOT/"client_data_audit.csv")
    fig,ax=plt.subplots(figsize=(9,4.6))
    x=np.arange(len(audit)); width=.36
    ax.bar(x-width/2,audit["train_benign"],width,label="Benign")
    ax.bar(x+width/2,audit["train_attack"],width,label="Attack")
    ax.set_xticks(x); ax.set_xticklabels([s.replace("_","\\n") for s in audit["client"]],fontsize=7)
    ax.set_ylabel("Training records"); ax.set_title("Sampled per-device training class counts"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG/"01_dataset_audit.png",dpi=220); plt.close(fig)
    # Figure 2: clean comparisons
    clean=metrics_df[metrics_df.scenario=="clean"]
    means=clean.groupby("model")["macro_f1"].mean().sort_values()
    stds=clean.groupby("model")["macro_f1"].std().reindex(means.index).fillna(0)
    fig,ax=plt.subplots(figsize=(9,4.8)); ax.barh(means.index,means.values,xerr=stds.values,capsize=3)
    ax.set_xlabel("Macro-F1 (higher is better)"); ax.set_xlim(0,1); ax.set_title("Clean-test macro-F1 across completed seeds")
    fig.tight_layout(); fig.savefig(FIG/"02_clean_model_comparison.png",dpi=220); plt.close(fig)
    # Figure 3: privacy vs utility
    dp=clean[clean.dp.astype(bool)]
    fig,ax=plt.subplots(figsize=(6.5,4.5))
    if len(dp):
        g=dp.groupby("model").agg(eps=("max_client_epsilon","mean"),f1=("macro_f1","mean"),sd=("macro_f1","std")).reset_index()
        for _,r in g.iterrows():
            ax.errorbar(r["eps"],r["f1"],yerr=0 if pd.isna(r["sd"]) else r["sd"],fmt="o",capsize=3,label=r["model"])
            ax.annotate(r["model"].replace("PrivFD-IDS ",""),(r["eps"],r["f1"]),fontsize=7,xytext=(4,4),textcoords="offset points")
        ax.set_xlabel("Maximum per-client epsilon (mean across seeds; delta=1e-5)")
        ax.set_ylabel("Macro-F1"); ax.set_ylim(0,1); ax.legend(fontsize=7)
    ax.set_title("Measured privacy-utility trade-off")
    fig.tight_layout(); fig.savefig(FIG/"03_privacy_utility.png",dpi=220); plt.close(fig)
    # Figure 4: poisoning robustness
    attacked=metrics_df[metrics_df.scenario!="clean"]
    fig,ax=plt.subplots(figsize=(8.5,4.8))
    if len(attacked):
        p=attacked.groupby(["scenario","model"])["macro_f1"].mean().unstack()
        p.plot(kind="bar",ax=ax)
    ax.set_ylabel("Macro-F1"); ax.set_ylim(0,1); ax.set_title("Test macro-F1 under controlled client poisoning"); ax.tick_params(axis="x",rotation=0)
    fig.tight_layout(); fig.savefig(FIG/"04_poisoning_robustness.png",dpi=220); plt.close(fig)
    # Figure 5: full method confusion matrix final selected seed
    ys=[]; ps=[]
    for name,c in clients.items():
        y=c["y_test"]; X=c["X_test"]*c["mask"][None,:]
        p=pred_prob(full_model,X)
        ys.extend(y.tolist()); ps.extend(p.tolist())
    pred=(np.asarray(ps)>=.5).astype(int)
    cm=confusion_matrix(ys,pred,labels=[0,1])
    fig,ax=plt.subplots(figsize=(5.2,4.5)); im=ax.imshow(cm)
    ax.set_xticks([0,1],["Pred. benign","Pred. attack"]); ax.set_yticks([0,1],["True benign","True attack"])
    for (i,j),v in np.ndenumerate(cm): ax.text(j,i,f"{v:,}",ha="center",va="center")
    ax.set_title(f"PrivFD-IDS confusion matrix (seed {full_seed})"); fig.colorbar(im,ax=ax); fig.tight_layout()
    fig.savefig(FIG/"05_privfd_confusion_matrix.png",dpi=220); plt.close(fig)
    # Figure 6: pooled precision-recall curve
    y=np.asarray(ys); p=np.asarray(ps)
    precision,recall,_=precision_recall_curve(y,p)
    fig,ax=plt.subplots(figsize=(5.8,4.4)); ax.plot(recall,precision)
    ax.set_xlabel("Recall (attack)"); ax.set_ylabel("Precision (attack)"); ax.set_title("PrivFD-IDS precision-recall curve")
    fig.tight_layout(); fig.savefig(FIG/"06_privfd_precision_recall.png",dpi=220); plt.close(fig)
    # Figure 7: convergence
    conv=rounds_df[rounds_df.scenario=="clean"].groupby(["model","round"])["macro_f1"].mean().reset_index()
    show=["FedAvg","FD-IDS-style","PrivFD-IDS sigma=1.0"]
    fig,ax=plt.subplots(figsize=(7.2,4.5))
    for m in show:
        g=conv[conv.model==m]
        if len(g): ax.plot(g["round"],g["macro_f1"],marker="o",label=m)
    ax.set_xlabel("Communication round"); ax.set_ylabel("Macro-F1"); ax.set_ylim(0,1); ax.set_title("Federated convergence on held-out test clients"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG/"07_validation_convergence.png",dpi=220); plt.close(fig)

def main():
    t0=time.time()
    print("Python",sys.version); print("Torch",torch.__version__); print("Opacus",__import__("opacus").__version__)
    print("Pinned source",REPO,DATA_COMMIT)
    clients=load_sampled_clients()
    device_names=list(clients)
    configs=make_configs()
    metrics=[]; rounds=[]; selections=[]; privacies=[]; device_metrics=[]
    final_for_fig={}
    # Seed-level clean baselines: centralized and local-only are reference points.
    for seed in SEEDS:
        seed_all(seed)
        Xpool=np.concatenate([c["X_train"] for c in clients.values()])
        ypool=np.concatenate([c["y_train"] for c in clients.values()])
        central=Detector().to(DEVICE)
        central=train_plain(central,Xpool,ypool,epochs=ROUNDS,seed=seed,lr=LR)
        ys=[]; ps=[]
        for c in clients.values():
            ys.extend(c["y_test"].tolist()); ps.extend(pred_prob(central,c["X_test"]).tolist())
        m=evaluate_probs(ys,ps)
        metrics.append({**m,"seed":seed,"model":"Centralized reference","scenario":"clean","aggregator":"centralized",
            "dp":False,"noise_multiplier":None,"mia_confidence_auc":float("nan"),"max_client_epsilon":float("nan"),
            "delta":None,"selected_feature_count":115,"malicious_client":""})
        # Local-only: one model per device, evaluated only on that device's held-out test split.
        local_outputs=[]; local_dev=[]
        for ci,(name,c) in enumerate(clients.items()):
            Xtr,ytr=batch_xy(c,"train",False)
            model=train_plain(Detector().to(DEVICE),Xtr,ytr,epochs=ROUNDS,seed=seed*100+ci,lr=LR)
            p=pred_prob(model,c["X_test"]); local_outputs.append((c["y_test"],p))
            local_dev.append({"client":name,**evaluate_probs(c["y_test"],p)})
        yl=np.concatenate([x[0] for x in local_outputs]); pl=np.concatenate([x[1] for x in local_outputs])
        m=evaluate_probs(yl,pl)
        metrics.append({**m,"seed":seed,"model":"Local-only","scenario":"clean","aggregator":"local",
            "dp":False,"noise_multiplier":None,"mia_confidence_auc":float("nan"),"max_client_epsilon":float("nan"),
            "delta":None,"selected_feature_count":115,"malicious_client":""})
        for d in local_dev: device_metrics.append({"seed":seed,"model":"Local-only","scenario":"clean",**d})
        for cfg in configs:
            final, rnd, sel, pri, perdev, model, outputs=run_federated(clients,cfg,seed,scenario="clean",
                capture_preds=(cfg["name"] in ["FD-IDS-style","FD-IDS + DP-SGD","PrivFD-IDS sigma=1.0"]))
            metrics.append(final); rounds.extend(rnd); selections.extend(sel); privacies.extend(pri)
            perdev["seed"]=seed; perdev["model"]=cfg["name"]; perdev["scenario"]="clean"; device_metrics.extend(perdev.to_dict("records"))
            if cfg["name"]=="PrivFD-IDS sigma=1.0" and seed==SEEDS[0]:
                final_for_fig={"model":model,"seed":seed}
        # Poisoning tests: compare FD-style FedAvg and complete PrivFD-IDS, only one Byzantine client.
        for scen in ["sign_flip","label_flip"]:
            for cfg in [configs[1],configs[5]]:
                final,rnd,sel,pri,perdev,model,outputs=run_federated(clients,cfg,seed,scenario=scen,capture_preds=False)
                metrics.append(final); rounds.extend(rnd); selections.extend(sel); privacies.extend(pri)
                perdev["seed"]=seed; perdev["model"]=cfg["name"]; perdev["scenario"]=scen; device_metrics.extend(perdev.to_dict("records"))
    metrics_df=pd.DataFrame(metrics)
    rounds_df=pd.DataFrame(rounds)
    selection_df=pd.DataFrame(selections)
    privacy_df=pd.DataFrame(privacies)
    dev_df=pd.DataFrame(device_metrics)
    metrics_df.to_csv(ROOT/"run_metrics.csv",index=False)
    rounds_df.to_csv(ROOT/"round_metrics.csv",index=False)
    selection_df.to_csv(ROOT/"client_selection.csv",index=False)
    privacy_df.to_csv(ROOT/"privacy_accounting.csv",index=False)
    dev_df.to_csv(ROOT/"per_device_metrics.csv",index=False)
    # Summary mean, SD, and normal-approximation 95% CI across fixed-split independent seeds.
    rows=[]
    for (scenario,model),g in metrics_df.groupby(["scenario","model"],dropna=False):
        row={"scenario":scenario,"model":model,"runs":int(len(g))}
        for col in ["accuracy","macro_f1","balanced_accuracy","mcc","precision_attack","recall_attack","fpr","pr_auc","roc_auc","max_client_epsilon","mia_confidence_auc"]:
            vals=pd.to_numeric(g[col],errors="coerce").dropna()
            row[col+"_mean"]=float(vals.mean()) if len(vals) else None
            row[col+"_std"]=float(vals.std(ddof=1)) if len(vals)>1 else (0.0 if len(vals)==1 else None)
            sd=row[col+"_std"]; mean=row[col+"_mean"]
            row[col+"_ci95_low"]=max(-1.0,mean-1.96*sd/np.sqrt(len(vals))) if len(vals)>1 else mean
            row[col+"_ci95_high"]=min(1.0,mean+1.96*sd/np.sqrt(len(vals))) if len(vals)>1 else mean
        rows.append(row)
    summary=pd.DataFrame(rows)
    summary.to_csv(ROOT/"summary_metrics.csv",index=False)
    # Honest-client rejection rate for Multi-Krum, clean and attacked; malicious selection separately.
    if len(selection_df):
        s=[]
        for keys,g in selection_df.groupby(["model","scenario","seed","round"]):
            honest=g[g.is_malicious==0]
            s.append({"model":keys[0],"scenario":keys[1],"seed":keys[2],"round":keys[3],
                "honest_rejection_rate":float((honest.selected==0).mean()) if len(honest) else np.nan,
                "malicious_selection_rate":float(g.loc[g.is_malicious==1,"selected"].mean()) if (g.is_malicious==1).any() else np.nan})
        pd.DataFrame(s).to_csv(ROOT/"selection_diagnostics.csv",index=False)
    # Privacy summary is accountant-derived across all private training rounds.
    if len(privacy_df):
        privacy_df.groupby(["model","scenario","seed"]).agg(
            max_epsilon=("epsilon","max"),mean_epsilon=("epsilon","mean"),max_round=("round","max"),
            clients=("client","nunique"),delta=("delta","max")).reset_index().to_csv(ROOT/"privacy_summary.csv",index=False)
    make_figures(metrics_df,rounds_df,privacy_df,selection_df,clients,
        full_preds=None,full_model=final_for_fig["model"],full_seed=final_for_fig["seed"])
    runtime={
        "run_id":time.strftime("%Y%m%dT%H%M%SZ",time.gmtime()),"python":sys.version,"platform":platform.platform(),
        "torch":torch.__version__,"opacus":__import__("opacus").__version__,"numpy":np.__version__,
        "pandas":pd.__version__,"sklearn":__import__("sklearn").__version__,"device":"CPU",
        "data_source_repository":f"https://github.com/{REPO}","data_commit":DATA_COMMIT,
        "clients":device_names,"feature_count":115,"seeds":SEEDS,"rounds":ROUNDS,"local_epochs":LOCAL_EPOCHS,
        "batch_size":BATCH_SIZE,"learning_rate":LR,"mi_top_k":TOP_K,"dp_noise_multipliers":NOISES,
        "clip_norm":CLIP_NORM,"delta":DELTA_DP,"multi_krum_f":1,"multi_krum_selected_n":6,
        "poisoning_scenarios":["sign_flip","label_flip"],"warmup_rounds":2,
        "wall_time_seconds":time.time()-t0,
        "important":"Sampled mirror, not full UCI archive; DP conditional on fixed calibration-derived scaler and MI mask."
    }
    (ROOT/"run_config.json").write_text(json.dumps(runtime,indent=2),encoding="utf-8")
    (ROOT/"RUN_STATUS.txt").write_text(
        "COMPLETED: all configured experiment loops and exports reached the end of main().\n"
        "Review raw run_metrics.csv, summary_metrics.csv, privacy_accounting.csv, per_device_metrics.csv, "
        "client_selection.csv, selection_diagnostics.csv, source_file_manifest.csv, data_provenance.json, "
        "run_config.json and figures before use in a manuscript.\n"
        "These are experiments on a pinned community-prepared N-BaIoT sample mirror, not the complete UCI archive.\n",
        encoding="utf-8")
    print("\nEXPERIMENT COMPLETE")
    print(summary[["scenario","model","runs","macro_f1_mean","macro_f1_std","max_client_epsilon_mean"]].to_string(index=False))
    print("Wall time seconds:",round(time.time()-t0,1))

if __name__=="__main__":
    main()
