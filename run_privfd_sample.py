#!/usr/bin/env python3
"""Reproducible PrivFD-IDS experiment on a sampled set of raw captures from the official UCI N-BaIoT archive.

The complete official archive is downloaded, SHA-256 recorded, and source capture files audited; model fitting uses a deterministic bounded sample rather than all 7,062,606 records. No scores are hard-coded; metrics and figures come from executed runs.
"""
from __future__ import annotations
import io, os, re, json, time, hashlib, random, platform, sys, warnings, zipfile, subprocess, secrets, math, shutil
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

OFFICIAL_DATA_URL = "https://archive.ics.uci.edu/static/public/442/detection+of+iot+botnet+attacks+n+baiot.zip"
DATASET_DOI = "10.24432/C5RC8J"
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/442/detection"
ROOT = Path("privfd_outputs")
FIG = ROOT / "figures"
CACHE = Path("data_cache/official_nbaiot")
ARCHIVE_PATH = CACHE / "n_baiot_official.zip"
RAW_ROOT = CACHE / "extracted"
ROOT.mkdir(exist_ok=True, parents=True); FIG.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
ROUNDS = 5
BATCH_SIZE = 256
LOCAL_EPOCHS = 1
LR = 0.025
TOP_K = 50
BENIGN_CAPTURE_CAP = 10000
ATTACK_CAPTURE_CAP = 2000
CAPTURE_EMBARGO_SAMPLED_ROWS = 5
DELTA_DP = 1e-5
CLIP_NORM = 1.0
NOISES = [0.7, 1.0, 1.5]
DEVICE_NAMES = [
    "Danmini_Doorbell", "Ecobee_Thermostat", "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor", "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera", "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
]
torch.set_num_threads(1)
DEVICE = torch.device("cpu")

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _sha256_file(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def prepare_official_dataset():
    CACHE.mkdir(parents=True, exist_ok=True); RAW_ROOT.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE_PATH.exists() or ARCHIVE_PATH.stat().st_size < 1_000_000_000:
        tmp = ARCHIVE_PATH.with_suffix(".zip.part")
        if tmp.exists(): tmp.unlink()
        print("Downloading the official UCI N-BaIoT archive (large file)...")
        h = hashlib.sha256()
        with requests.get(OFFICIAL_DATA_URL, stream=True, timeout=(60, 300)) as response:
            response.raise_for_status()
            with open(tmp, "wb") as dst:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        dst.write(chunk); h.update(chunk)
        if tmp.stat().st_size < 1_000_000_000:
            raise RuntimeError(f"Official UCI ZIP unexpectedly small: {tmp.stat().st_size} bytes")
        tmp.replace(ARCHIVE_PATH); archive_sha = h.hexdigest()
    else:
        archive_sha = _sha256_file(ARCHIVE_PATH)
    (CACHE / "n_baiot_official.sha256").write_text(f"{archive_sha}  {ARCHIVE_PATH.name}\n", encoding="utf-8")
    sentinel = RAW_ROOT / ".uci_zip_extracted"
    if not sentinel.exists():
        print(f"Extracting official UCI ZIP ({ARCHIVE_PATH.stat().st_size:,} bytes)...")
        with zipfile.ZipFile(ARCHIVE_PATH) as zf:
            # Zip extraction verifies CRC while extracting; a separate full testzip scan is unnecessary.
            zf.extractall(RAW_ROOT)
        sentinel.write_text("completed\n", encoding="utf-8")
    rar_files = sorted(RAW_ROOT.rglob("*.rar"))
    if not rar_files:
        raise RuntimeError("No attack RAR archives found in the official UCI ZIP")
    seven_zip = shutil.which("7zz") or shutil.which("7z")
    if seven_zip is None:
        raise RuntimeError("7-Zip CLI missing; workflow must install package '7zip'")
    for rar in rar_files:
        out_dir = rar.parent / (rar.stem + "_extracted")
        if out_dir.exists() and any(out_dir.rglob("*.csv")):
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting attack archive {rar.relative_to(RAW_ROOT)}")
        proc = subprocess.run([seven_zip, "x", str(rar), f"-o{out_dir}", "-y", "-bso0", "-bsp0"],
                              text=True, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"7-Zip failed on {rar}: {proc.stderr[-2500:]}")
    csv_files=[]
    for path in sorted(RAW_ROOT.rglob("*.csv")):
        if any(x in path.name.lower() for x in ["readme","metadata","description"]):
            continue
        if not any(dev in str(path) for dev in DEVICE_NAMES):
            continue
        try:
            if len(pd.read_csv(path,nrows=0).columns)==115:
                csv_files.append(path)
        except Exception as exc:
            print(f"Skipping unreadable CSV {path}: {exc}")
    if len(csv_files) < 50:
        raise RuntimeError(f"Too few official 115-feature capture CSVs: {len(csv_files)}")
    return csv_files, archive_sha

def _device_name(path):
    matches=[device for device in DEVICE_NAMES if device in str(path)]
    if len(matches)!=1:
        raise RuntimeError(f"Could not determine device from {path}: {matches}")
    return matches[0]

def _count_csv_data_rows(path):
    with open(path,"r",encoding="utf-8",errors="replace",newline="") as fh:
        return max(0,sum(1 for _ in fh)-1)

def _systematic_capture_sample(path, cap):
    total_rows=_count_csv_data_rows(path)
    if total_rows < 100:
        raise RuntimeError(f"Too few rows in official capture {path}: {total_rows}")
    n_take=min(int(cap),int(total_rows))
    target_idx=np.linspace(0,total_rows-1,n_take,dtype=np.int64)
    samples=[]; offsets=[]; start=0
    for chunk in pd.read_csv(path,chunksize=100000):
        end=start+len(chunk)
        lo=np.searchsorted(target_idx,start,side="left")
        hi=np.searchsorted(target_idx,end,side="left")
        if hi>lo:
            local=target_idx[lo:hi]-start
            part=chunk.iloc[local].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
            if part.shape[1]!=115:
                raise RuntimeError(f"{path} yielded {part.shape[1]} features, expected 115")
            part=part.fillna(part.median(numeric_only=True)).fillna(0.0)
            arr=np.nan_to_num(part.to_numpy(dtype=np.float32),nan=0.0,posinf=0.0,neginf=0.0)
            samples.append(arr); offsets.extend(target_idx[lo:hi].tolist())
        start=end
    if not samples: raise RuntimeError(f"No sampled rows from {path}")
    return np.vstack(samples).astype(np.float32),np.asarray(offsets,dtype=np.int64),int(total_rows)

def _split_capture_sequence(X,row_indices,source_file):
    n=len(X); c1=int(.10*n); c2=int(.70*n); c3=int(.85*n); gap=CAPTURE_EMBARGO_SAMPLED_ROWS
    ranges={"cal":(0,max(0,c1-gap)),
            "train":(min(n,c1+gap),max(c1+gap,c2-gap)),
            "val":(min(n,c2+gap),max(c2+gap,c3-gap)),
            "test":(min(n,c3+gap),n)}
    out={}
    for split,(a,b) in ranges.items():
        a=min(max(0,a),n); b=min(max(a,b),n)
        out[split]={"X":X[a:b],"row_indices":row_indices[a:b],
                    "ids":[f"{source_file}#{int(v)}" for v in row_indices[a:b]]}
    return out

def _cat_capture_parts(parts,label):
    if not parts: return np.empty((0,115),dtype=np.float32),np.empty(0,dtype=np.int64),[]
    X=np.vstack([p["X"] for p in parts]).astype(np.float32)
    y=np.full(len(X),label,dtype=np.int64)
    ids=sum([p["ids"] for p in parts],[])
    return X,y,ids

def _balance_training(X,y,ids,seed):
    i0=np.flatnonzero(y==0); i1=np.flatnonzero(y==1); n=min(len(i0),len(i1))
    if n<100: raise RuntimeError(f"Insufficient train classes: benign={len(i0)}, attack={len(i1)}")
    rng=np.random.default_rng(seed)
    take=np.concatenate([np.sort(rng.choice(i0,n,replace=False)),np.sort(rng.choice(i1,n,replace=False))])
    rng.shuffle(take)
    return X[take],y[take],[ids[i] for i in take]

def load_sampled_clients():
    csv_files,archive_sha=prepare_official_dataset()
    capture_rows=[]; capture_manifest=[]; raw_records_total=0
    for fi,path in enumerate(csv_files):
        device=_device_name(path); label=0 if "benign_traffic" in path.name.lower() else 1
        cap=BENIGN_CAPTURE_CAP if label==0 else ATTACK_CAPTURE_CAP
        rel=str(path.relative_to(RAW_ROOT)).replace("\\\\","/")
        X,row_idx,total_rows=_systematic_capture_sample(path,cap)
        raw_records_total+=total_rows
        pieces=_split_capture_sequence(X,row_idx,rel)
        row={"device":device,"source_file":rel,"label":label,"rows_in_source":total_rows,
             "file_bytes":int(path.stat().st_size),"sampled_rows":int(len(row_idx)),
             "cal_rows":len(pieces["cal"]["ids"]),"train_rows":len(pieces["train"]["ids"]),
             "validation_rows":len(pieces["val"]["ids"]),"test_rows":len(pieces["test"]["ids"]),
             "purged_sampled_rows":int(len(row_idx)-sum(len(pieces[k]["ids"]) for k in pieces)),
             "split_policy":"chronological 10/60/15/15 blocks; 5 sampled-row embargo"}
        capture_manifest.append({"device":device,"file":rel,"label":label,"rows_in_official_capture":total_rows,
            "sampled_rows":len(row_idx),"file_bytes":int(path.stat().st_size),
            "sampling":"systematic, temporal order preserved"})
        capture_rows.append((device,label,pieces,row))
        print(f"Audited {device}/{path.name}: raw_rows={total_rows:,}, sampled={len(row_idx):,}, label={label}")
    if set(r[0] for r in capture_rows)!=set(DEVICE_NAMES):
        raise RuntimeError(f"Official archive did not yield expected nine devices: {sorted(set(r[0] for r in capture_rows))}")
    clients={}; client_audit=[]; capture_audit=[]; overlap_audit=[]
    for di,device in enumerate(DEVICE_NAMES):
        by_split={s:{0:[],1:[]} for s in ["cal","train","val","test"]}
        files=[]
        for dname,label,pieces,row in capture_rows:
            if dname!=device: continue
            files.append(row["source_file"]); capture_audit.append(row)
            for split in by_split:
                by_split[split][label].append(pieces[split])
        merged={}
        for split in ["cal","train","val","test"]:
            X0,y0,ids0=_cat_capture_parts(by_split[split][0],0)
            X1,y1,ids1=_cat_capture_parts(by_split[split][1],1)
            X=np.vstack([X0,X1]).astype(np.float32); y=np.concatenate([y0,y1]); ids=ids0+ids1
            if split=="train": X,y,ids=_balance_training(X,y,ids,24000+di)
            merged[split]={"X":X,"y":y,"ids":ids}
        idsets={s:set(merged[s]["ids"]) for s in merged}
        for a,b in [("cal","train"),("cal","val"),("cal","test"),("train","val"),("train","test"),("val","test")]:
            ov=idsets[a].intersection(idsets[b])
            overlap_audit.append({"device":device,"split_a":a,"split_b":b,"overlap_rows":len(ov)})
            if ov: raise RuntimeError(f"Source-row identity leakage for {device}: {a}/{b}: {len(ov)} rows")
        # Fit scaling/MI only on the explicit calibration partition; it is outside the DP-protected train partition.
        Xcal,ycal=merged["cal"]["X"],merged["cal"]["y"]
        scaler=StandardScaler().fit(Xcal)
        Xcal_s=np.nan_to_num(scaler.transform(Xcal),nan=0.0,posinf=0.0,neginf=0.0)
        scores=mutual_info_classif(Xcal_s,ycal,random_state=di,n_neighbors=3)
        mask=np.zeros(115,dtype=np.float32); mask[np.argsort(scores)[-TOP_K:]]=1.0
        def transform(a):
            z=scaler.transform(a)
            return np.clip(np.nan_to_num(z,nan=0.0,posinf=0.0,neginf=0.0),-10.0,10.0).astype(np.float32)
        rng=np.random.default_rng(25000+di); perm=rng.permutation(len(merged["train"]["y"]))
        Xtrain=merged["train"]["X"][perm]; ytrain=merged["train"]["y"][perm]
        clients[device]={
            "X_train":transform(Xtrain),"y_train":ytrain,
            "X_val":transform(merged["val"]["X"]),"y_val":merged["val"]["y"],
            "X_test":transform(merged["test"]["X"]),"y_test":merged["test"]["y"],
            "X_cal":transform(Xcal),"y_cal":ycal,"mask":mask,"files":files,
            "train_n":int(len(ytrain)),"val_n":int(len(merged["val"]["y"])),
            "test_n":int(len(merged["test"]["y"])),"features_selected":int(mask.sum()),
            "_dataset_cache":{}
        }
        n0=int((ytrain==0).sum()); n1=int((ytrain==1).sum())
        client_audit.append({
            "client":device,"source_files":len(files),
            "benign_selected":int(sum(len(pieces[s]["ids"]) for dname,label,pieces,_ in capture_rows if dname==device and label==0 for s in ["cal","train","val","test"])),
            "attack_selected":int(sum(len(pieces[s]["ids"]) for dname,label,pieces,_ in capture_rows if dname==device and label==1 for s in ["cal","train","val","test"])),
            "calibration_rows":len(ycal),"train_rows":len(ytrain),"validation_rows":len(merged["val"]["y"]),
            "test_rows":len(merged["test"]["y"]),"features_selected":int(mask.sum()),
            "train_benign":n0,"train_attack":n1,
            "test_benign":int((merged["test"]["y"]==0).sum()),"test_attack":int((merged["test"]["y"]==1).sum()),
            "train_balance":"balanced by deterministic downsampling",
            "test_policy":"all non-purged held-out sequence-block rows retained"
        })
        print(f"Loaded {device}: train={len(ytrain)} val={len(merged['val']['y'])} test={len(merged['test']['y'])}, train class counts {n0}/{n1}")
    pd.DataFrame(capture_manifest).to_csv(ROOT/"source_file_manifest.csv",index=False)
    pd.DataFrame(capture_audit).to_csv(ROOT/"capture_split_audit.csv",index=False)
    pd.DataFrame(overlap_audit).to_csv(ROOT/"split_leakage_audit.csv",index=False)
    pd.DataFrame(client_audit).to_csv(ROOT/"client_data_audit.csv",index=False)
    (ROOT/"data_provenance.json").write_text(json.dumps({
        "dataset":"Official UCI N-BaIoT archive","dataset_page":DATASET_PAGE,"dataset_doi":DATASET_DOI,
        "download_url":OFFICIAL_DATA_URL,"archive_sha256":archive_sha,"source_capture_files":len(capture_manifest),
        "raw_records_counted_across_csv_files":int(raw_records_total),"device_clients":DEVICE_NAMES,"feature_count":115,
        "sampling":{"benign_capture_cap":BENIGN_CAPTURE_CAP,"attack_cap_per_capture":ATTACK_CAPTURE_CAP,
            "method":"systematic, ordered sample from each original CSV capture; original row identities preserved",
            "training_balance":"per-client training downsampled to equal benign/attack counts"},
        "labels":{"benign_traffic.csv":0,"all attack CSV captures":1},
        "split":{"calibration_fraction":0.10,"train_fraction":0.60,"validation_fraction":0.15,"test_fraction":0.15,
            "method":"chronological blocks per capture with 5 sampled-row embargo at partition boundaries",
            "row_identity_overlap":"pairwise source-path + original-row-ID audit; all intersections must be zero"},
        "privacy_scope":"Record-level DP covers only client-local training-partition records. Public calibration rows used for scaling and MI are excluded from the protected set; the statement is conditional on this preprocessing.",
        "limit":"The full official archive is downloaded and audited, but model training uses a bounded deterministic sample from all available original captures; this is not an all-7,062,606-record training run."
    },indent=2),encoding="utf-8")
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

def pooled_evaluate(model, clients, mi, split="test"):
    ys=[]; ps=[]; per=[]
    device_outputs={}
    for name,c in clients.items():
        X,y=batch_xy(c,split,mi)
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
                       flip_labels=False, noise_generator=None):
    random.seed(seed); np.random.seed(seed)
    if not config["dp"]: torch.manual_seed(seed)
    key=(bool(config["mi"]),bool(flip_labels))
    cache=client.setdefault("_dataset_cache",{})
    if key not in cache:
        X=client["X_train"]
        if config["mi"]: X=X*client["mask"][None,:]
        y=client["y_train"].copy()
        if flip_labels: y=1-y
        cache[key]=TensorDataset(torch.as_tensor(X,dtype=torch.float32),torch.as_tensor(y,dtype=torch.long))
    ds=cache[key]
    loader=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True,
        generator=torch.Generator().manual_seed(seed),drop_last=False)
    model=Detector().to(DEVICE)
    base={k:v.detach().cpu().clone() for k,v in global_model.state_dict().items()}
    model.load_state_dict(base)
    teacher=deepcopy(global_model).to(DEVICE).eval() if config["kd"] else None
    opt=torch.optim.SGD(model.parameters(),lr=LR)
    exact_q=None; steps_this_round=0
    if config["dp"]:
        model,opt,loader=privacy_engine.make_private(
            module=model,optimizer=opt,data_loader=loader,
            noise_multiplier=config["sigma"],max_grad_norm=CLIP_NORM,poisson_sampling=True,
            noise_generator=noise_generator)
        exact_q=float(getattr(loader,"sample_rate",1.0/len(loader)))
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
            if config["dp"]: steps_this_round+=1
            if config["prox"]:
                with torch.no_grad():
                    core=core_model(model)
                    for nm,p in core.named_parameters():
                        p.add_(base[nm].to(DEVICE)-p,alpha=LR*0.01)
    final={k:v.detach().cpu().clone() for k,v in core_model(model).state_dict().items()}
    delta={k:final[k]-base[k] for k in base}
    del model,teacher,opt,loader
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return delta,steps_this_round,exact_q

def run_federated(clients, config, seed, scenario="clean", capture_preds=False):
    seed_all(seed)
    global_model=Detector().to(DEVICE)
    global_model.eval()
    client_names=list(clients)
    privacy_engines={name:PrivacyEngine(accountant="rdp") for name in client_names} if config["dp"] else {}
    noise_generators={name:torch.Generator(device=DEVICE).manual_seed(secrets.randbits(63))
                      for name in client_names} if config["dp"] else {}
    attacker=client_names[(seed*2+2)%len(client_names)]
    per_round=[]; selection_rows=[]; privacy_rows=[]
    malicious_start=2
    dp_steps_cumulative={name:0 for name in client_names}; dp_q_by_client={name:None for name in client_names}
    for rnd in range(ROUNDS):
        deltas={}; steps_by_client={}; q_by_client={}
        for name in client_names:
            attack_active=(scenario!="clean" and rnd>=malicious_start and name==attacker)
            delta,steps_this_round,q_actual=train_local_update(global_model,clients[name],config,seed*100000+rnd*100+client_names.index(name),
                privacy_engine=privacy_engines.get(name),flip_labels=(scenario=="label_flip" and attack_active),
                noise_generator=noise_generators.get(name))
            if config["dp"]:
                dp_steps_cumulative[name]+=steps_this_round; dp_q_by_client[name]=q_actual
                steps_by_client[name]=steps_this_round; q_by_client[name]=q_actual
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
                    "client":name,"epsilon_opacus":eps,"epsilon":eps,"delta":DELTA_DP,"noise_multiplier":config["sigma"],
                    "max_grad_norm":CLIP_NORM,"train_records":clients[name]["train_n"],
                    "sample_rate_actual":dp_q_by_client[name],"steps_this_round":steps_by_client[name],
                    "cumulative_dp_steps":dp_steps_cumulative[name],"accountant":"Opacus RDPAccountant",
                    "noise_rng":"independent high-entropy-seeded per-client torch.Generator"})
        metrics,_,_=pooled_evaluate(global_model,clients,config["mi"],split="val")
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
    ax.set_xlabel("Communication round"); ax.set_ylabel("Validation macro-F1"); ax.set_ylim(0,1); ax.set_title("Federated convergence on validation clients"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG/"07_validation_convergence.png",dpi=220); plt.close(fig)

def main():
    t0=time.time()
    print("Python",sys.version); print("Torch",torch.__version__); print("Opacus",__import__("opacus").__version__)
    print("Official dataset source",DATASET_PAGE,"DOI",DATASET_DOI)
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
    # Independent second implementation of the sampled-Gaussian RDP accountant.
    if len(privacy_df):
        from dp_accounting import dp_event, rdp
        independent_rows=[]
        for (seed_i,model_i,scenario_i,client_i),g in privacy_df.groupby(["seed","model","scenario","client"]):
            for _,rr in g.sort_values("round").iterrows():
                q=float(rr["sample_rate_actual"]); sigma=float(rr["noise_multiplier"])
                steps=int(rr["cumulative_dp_steps"]); delta=float(rr["delta"])
                acc=rdp.RdpAccountant()
                if steps>0:
                    acc.compose(dp_event.PoissonSampledDpEvent(q,dp_event.GaussianDpEvent(sigma)),count=steps)
                    eps_ind=float(acc.get_epsilon(delta=delta))
                else: eps_ind=0.0
                independent_rows.append({"seed":int(seed_i),"model":model_i,"scenario":scenario_i,"client":client_i,
                    "round":int(rr["round"]),"sample_rate_actual":q,"noise_multiplier":sigma,
                    "cumulative_dp_steps":steps,"epsilon_opacus":float(rr["epsilon_opacus"]),
                    "epsilon_dp_accounting":eps_ind,"absolute_difference":abs(float(rr["epsilon_opacus"])-eps_ind),
                    "delta":delta})
        independent_df=pd.DataFrame(independent_rows)
        independent_df.to_csv(ROOT/"privacy_independent_audit.csv",index=False)
        max_error=float(independent_df["absolute_difference"].max()) if len(independent_df) else 0.0
        audit_status="PASS" if max_error<=0.5 else "REVIEW_REQUIRED"
        (ROOT/"privacy_independent_audit_status.txt").write_text(
            f"{audit_status}\nmax_absolute_epsilon_difference={max_error:.6f}\n"
            "Opacus and Google dp-accounting evaluated the same recorded Poisson sample rate, noise multiplier, "
            "cumulative optimizer-step count and delta. Inspect the per-client/per-round CSV; finite RDP order grids can differ.\n",
            encoding="utf-8")
        privacy_df.groupby(["model","scenario","seed"]).agg(
            max_epsilon=("epsilon","max"),mean_epsilon=("epsilon","mean"),max_round=("round","max"),
            clients=("client","nunique"),delta=("delta","max")).reset_index().to_csv(ROOT/"privacy_summary.csv",index=False)
    make_figures(metrics_df,rounds_df,privacy_df,selection_df,clients,
        full_preds=None,full_model=final_for_fig["model"],full_seed=final_for_fig["seed"])
    runtime={
        "run_id":time.strftime("%Y%m%dT%H%M%SZ",time.gmtime()),"python":sys.version,"platform":platform.platform(),
        "torch":torch.__version__,"opacus":__import__("opacus").__version__,"numpy":np.__version__,
        "pandas":pd.__version__,"sklearn":__import__("sklearn").__version__,"device":"CPU",
        "data_source_repository":DATASET_PAGE,"data_doi":DATASET_DOI,"official_archive_url":OFFICIAL_DATA_URL,
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
        "The official UCI N-BaIoT archive was downloaded, SHA-256 hashed, and its raw capture CSVs audited.\n"
        "Training used deterministic temporal samples from the official captures; it is not an all-7,062,606-record training run.\n"
        "Review run_metrics.csv, summary_metrics.csv, privacy_accounting.csv, privacy_independent_audit.csv, split_leakage_audit.csv, capture_split_audit.csv, per_device_metrics.csv, source_file_manifest.csv, data_provenance.json, run_config.json and figures.\n",
        encoding="utf-8")
    print("\nEXPERIMENT COMPLETE")
    print(summary[["scenario","model","runs","macro_f1_mean","macro_f1_std","max_client_epsilon_mean"]].to_string(index=False))
    print("Wall time seconds:",round(time.time()-t0,1))

if __name__=="__main__":
    main()

# Artifact packaging trigger; the validated run source is pinned separately in export_runner.yml.
