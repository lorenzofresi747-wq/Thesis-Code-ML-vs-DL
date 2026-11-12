#!/usr/bin/env python3
"""Validation-selected PyTorch MLP / LSTM / GRU forecasts, run outside R's process.

Input CSV: one row per stock and forecast origin; Sample is Training/Validation/Test.
All features and scaling are fixed from the training sample. Full test dates are
retained for every recurrent architecture using historical context at split edges.
"""
import argparse
import copy
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

META = {"Ticker", "sector", "Date", "TargetDate", "Sample", "Target"}
CONFIGS = [
    {"Model": "MLP", "Config": "MLP_32_16", "Hidden": 32, "Q": 1},
    {"Model": "MLP", "Config": "MLP_64_32", "Hidden": 64, "Q": 1},
    {"Model": "LSTM", "Config": "LSTM_Q5", "Hidden": 32, "Q": 5},
    {"Model": "LSTM", "Config": "LSTM_Q20", "Hidden": 32, "Q": 20},
    {"Model": "GRU", "Config": "GRU_Q5", "Hidden": 32, "Q": 5},
    {"Model": "GRU", "Config": "GRU_Q20", "Hidden": 32, "Q": 20},
]


class ForecastNet(nn.Module):
    def __init__(self, arch: str, features: int, hidden: int):
        super().__init__()
        self.arch = arch
        if arch == "MLP":
            self.network = nn.Sequential(
                nn.Linear(features, hidden), nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(hidden, hidden // 2), nn.ReLU(),
                nn.Linear(hidden // 2, 1),
            )
        elif arch in ("LSTM", "GRU"):
            cls = nn.LSTM if arch == "LSTM" else nn.GRU
            self.recurrent = cls(input_size=features, hidden_size=hidden,
                                 num_layers=1, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(0.10), nn.Linear(hidden, 1))
        else:
            raise ValueError(arch)

    def forward(self, x):
        if self.arch == "MLP":
            return self.network(x).squeeze(-1)
        seq_out, _ = self.recurrent(x)
        return self.head(seq_out[:, -1, :]).squeeze(-1)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_arrays(frame, features, x_mean, x_std, y_mean, y_std, q, arch):
    raw = frame[features].to_numpy(dtype=np.float32)
    xx = (raw - x_mean) / x_std
    target = frame["Target"].to_numpy(dtype=np.float32)
    if q == 1:
        indices = np.arange(len(frame))
        all_x = xx
    else:
        indices = np.arange(q - 1, len(frame))
        all_x = np.stack([xx[i - q + 1:i + 1] for i in indices]).astype(np.float32)
    splits = frame["Sample"].to_numpy()[indices]
    arrays = {}
    for split in ("Training", "Validation", "Test"):
        selected = splits == split
        idx = indices[selected]
        arrays[split] = {
            "x": np.ascontiguousarray(all_x[selected], dtype=np.float32),
            "y": (target[idx] - y_mean) / y_std,
            "actual": target[idx],
            "idx": idx,
        }
    for split in arrays:
        if len(arrays[split]["idx"]) == 0:
            raise ValueError(f"Empty {split} for q={q}")
    return arrays


def predict_scaled(net, xx, batch_size=512):
    net.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(xx), batch_size):
            part = torch.from_numpy(xx[start:start + batch_size])
            predictions.append(net(part).cpu().numpy())
    return np.concatenate(predictions)


def fit_one(spec, arrays, features, seed, epochs, patience, batch_size):
    seed_everything(seed)
    net = ForecastNet(spec["Model"], features, spec["Hidden"])
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    train_x = torch.from_numpy(arrays["Training"]["x"])
    train_y = torch.from_numpy(arrays["Training"]["y"].astype(np.float32))
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    val_x = arrays["Validation"]["x"]
    val_y = arrays["Validation"]["y"]
    best_loss, best_state, best_epoch, stalled = float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        net.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(net(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
        v_pred = predict_scaled(net, val_x)
        v_loss = float(np.mean((val_y - v_pred) ** 2))
        if v_loss < best_loss - 1e-8:
            best_loss = v_loss
            best_state = copy.deepcopy(net.state_dict())
            best_epoch = epoch
            stalled = 0
        else:
            stalled += 1
            if stalled >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training failed to produce a checkpoint")
    net.load_state_dict(best_state)
    return net, best_epoch


def scores(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse_zero = float(np.sqrt(np.mean(actual ** 2)))
    mae_zero = float(np.mean(np.abs(actual)))
    return {
        "N": len(actual), "RMSE": rmse, "MAE": mae,
        "BenchmarkRMSE": rmse_zero, "BenchmarkMAE": mae_zero,
        "RMSEGain": 100 * (1 - rmse / rmse_zero) if rmse_zero else np.nan,
        "MAEGain": 100 * (1 - mae / mae_zero) if mae_zero else np.nan,
        "R2_OOS": 1 - np.sum((actual - predicted) ** 2) / np.sum(actual ** 2)
        if np.any(actual) else np.nan,
        "DirectionalAccuracy": float(np.mean(actual * predicted > 0)),
    }


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    df.to_csv(temporary, index=False)
    os.replace(temporary, path)


def process_stock(stock, features, output, args, stock_index):
    ticker = str(stock["Ticker"].iloc[0])
    sector = str(stock["sector"].iloc[0])
    pred_file = output / "selected_predictions" / f"{ticker}.csv"
    report_file = output / "selected_models" / f"{ticker}.csv"
    all_file = output / "all_predictions" / f"{ticker}.csv"
    arch_file = output / "architecture_results" / f"{ticker}.csv"
    cand_file = output / "validation_candidates" / f"{ticker}.csv"
    needed = (pred_file, report_file, all_file, arch_file, cand_file)
    if not args.force and all(f.exists() for f in needed):
        print(f"[SKIP] {ticker}: results already saved", flush=True)
        return
    stock = stock.sort_values("Date").reset_index(drop=True)
    if stock["Date"].duplicated().any() or stock["TargetDate"].duplicated().any():
        raise ValueError(f"Duplicated dates in {ticker}")
    if not (stock["TargetDate"] > stock["Date"]).all():
        raise ValueError(f"TargetDate not after forecast origin in {ticker}")
    train = stock["Sample"] == "Training"
    xtrain = stock.loc[train, features].to_numpy(dtype=np.float64)
    x_mean = xtrain.mean(axis=0).astype(np.float32)
    x_std = xtrain.std(axis=0, ddof=1).astype(np.float32)
    x_std[(~np.isfinite(x_std)) | (x_std < 1e-12)] = 1.0
    ytrain = stock.loc[train, "Target"].to_numpy(dtype=np.float64)
    y_mean, y_std = float(ytrain.mean()), float(ytrain.std(ddof=1))
    if not np.isfinite(y_std) or y_std <= 0:
        raise ValueError(f"Target variance zero for {ticker}")

    validations = []
    arch_winners = {}
    for ci, spec in enumerate(CONFIGS):
        arrays = get_arrays(stock, features, x_mean, x_std, y_mean, y_std,
                            spec["Q"], spec["Model"])
        # Fixed seed per configuration, identical protocol for every stock.
        seed = args.seed + ci
        model, best_epoch = fit_one(spec, arrays, len(features), seed,
                                    args.epochs, args.patience, args.batch_size)
        val_prediction = predict_scaled(model, arrays["Validation"]["x"]) * y_std + y_mean
        val_actual = arrays["Validation"]["actual"]
        val_rmse = float(np.sqrt(np.mean((val_actual - val_prediction) ** 2)))
        val_mae = float(np.mean(np.abs(val_actual - val_prediction)))
        candidate = {"Ticker": ticker, "sector": sector, **spec,
                     "Seed": seed, "BestEpoch": best_epoch,
                     "ValidationRMSE": val_rmse, "ValidationMAE": val_mae}
        validations.append(candidate)
        arch = spec["Model"]
        if arch not in arch_winners or val_rmse < arch_winners[arch]["ValidationRMSE"]:
            arch_winners[arch] = {**candidate, "model": model, "arrays": arrays, "spec": spec}
        print(f"  {ticker}: {spec['Config']} val RMSE={val_rmse:.6f} epoch={best_epoch}", flush=True)

    # ARCHITECTURE selection occurs before any test predictions are calculated.
    best_arch = min(arch_winners, key=lambda name: arch_winners[name]["ValidationRMSE"])
    test_rows, test_preds, best_pred_frame = [], [], None
    for arch in ("MLP", "LSTM", "GRU"):
        winner = arch_winners[arch]
        split = winner["arrays"]["Test"]
        actual = split["actual"]
        predicted = predict_scaled(winner["model"], split["x"]) * y_std + y_mean
        report = {k: v for k, v in winner.items() if k not in ("model", "arrays", "spec")}
        report.update(scores(actual, predicted))
        report["SelectedDL"] = arch == best_arch
        test_rows.append(report)
        original = stock.iloc[split["idx"]]
        pred_df = pd.DataFrame({
            "Ticker": ticker, "sector": sector, "Model": arch,
            "ForecastOrigin": original["Date"].dt.strftime("%Y-%m-%d").to_numpy(),
            "TargetDate": original["TargetDate"].dt.strftime("%Y-%m-%d").to_numpy(),
            "ActualReturn": actual, "Prediction": predicted,
        })
        test_preds.append(pred_df)
        if arch == best_arch:
            best_pred_frame = pred_df
        ckpt = output / "model_checkpoints" / f"{ticker}_{arch}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": winner["model"].state_dict(),
            "spec": winner["spec"], "feature_names": features,
            "x_mean": x_mean, "x_std": x_std,
            "y_mean": y_mean, "y_std": y_std,
            "training_seed": winner["Seed"],
        }, ckpt)

    save_csv(pd.DataFrame(validations), cand_file)
    save_csv(pd.DataFrame(test_rows), arch_file)
    save_csv(pd.concat(test_preds, ignore_index=True), all_file)
    save_csv(pd.DataFrame([row for row in test_rows if row["SelectedDL"]]), report_file)
    save_csv(best_pred_frame, pred_file)  # save this LAST for checkpoint completeness
    print(f"[DONE] {ticker} selected {best_arch}; all results saved", flush=True)


def aggregate(output):
    collections = ["validation_candidates", "architecture_results",
                   "all_predictions", "selected_predictions", "selected_models"]
    for folder in collections:
        files = sorted((output / folder).glob("*.csv"))
        if files:
            combined = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
            combined.to_csv(output / f"dl_{folder}.csv", index=False)
            print(f"Saved {folder}: {len(combined)} rows", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tickers", default="", help="optional comma-separated subset for smoke test")
    parser.add_argument("--force", action="store_true", help="retrain existing outputs")
    args = parser.parse_args()
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input, parse_dates=["Date", "TargetDate"])
    required = META
    if not required.issubset(set(data.columns)):
        raise ValueError("Missing columns: " + ", ".join(sorted(required - set(data.columns))))
    features = [col for col in data.columns if col not in META]
    if not features:
        raise ValueError("No predictors found")
    if not np.isfinite(data[features + ["Target"]].to_numpy(dtype=float)).all():
        raise ValueError("NA/Inf found in Target or predictors: clean input first")
    if not set(data["Sample"].unique()) == {"Training", "Validation", "Test"}:
        raise ValueError("Expected Sample values Training, Validation, Test")
    all_tickers = sorted(data["Ticker"].unique().tolist())
    chosen = all_tickers
    if args.tickers:
        chosen = [s.strip() for s in args.tickers.split(",")]
        if not set(chosen).issubset(all_tickers):
            raise ValueError("Requested ticker not in data")
    print(f"Stocks in dataset: {len(all_tickers)} | Selected to run: {len(chosen)}", flush=True)
    print(f"Predictors: {len(features)} | Device: CPU | Configurations: {len(CONFIGS)}", flush=True)
    for idx, ticker in enumerate(chosen, 1):
        print(f"\n[{idx}/{len(chosen)}] {ticker}", flush=True)
        process_stock(data[data.Ticker == ticker].copy(), features, output, args, idx)
    aggregate(output)
    print("\nFINISHED. Import the dl_*.csv files into R.", flush=True)

if __name__ == "__main__":
    main()
