#!/usr/bin/env python3
"""Thesis DL overfitting diagnostics, without modifying any original model.

Default: reevaluate original saved checkpoint weights on Train / Validation / Test
for all 33 securities x 3 selected architectures, using the exact original
training-sample scaling. Outputs model scores; no training or selection.

Optional --learning-curves: retrains only predeclared demonstration tickers
using their original validation-selected specs / seeds, recording epoch losses.
This is a diagnostic rerun; NOT a modification of the original results.
"""
import argparse
import copy
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def import_trainer(path):
    spec = importlib.util.spec_from_file_location('thesis_dl_trainer_for_diag', path)
    if spec is None or spec.loader is None:
        raise ValueError('Cannot import original trainer')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_checkpoint(path):
    # Never load checkpoint files from an untrusted source; torch serialization
    # can execute arbitrary code when weights_only=False.
    return torch.load(path, map_location='cpu', weights_only=False)


def score_split(arrays, net, y_mean, y_std, trainer, split_name):
    subset = arrays[split_name]
    actual = subset['actual'].astype(np.float64)
    forecast = trainer.predict_scaled(net, subset['x']).astype(np.float64) * y_std + y_mean
    benchmark = float(np.sqrt(np.mean(actual**2)))
    rmse = float(np.sqrt(np.mean((actual - forecast)**2)))
    mae = float(np.mean(np.abs(actual - forecast)))
    return dict(N=len(actual), RMSE=rmse, ZeroRMSE=benchmark,
                RelRMSE=(rmse/benchmark if benchmark > 0 else np.nan),
                RMSEGain=(100*(1-rmse/benchmark) if benchmark > 0 else np.nan),
                MAE=mae)


def checkpoint_diagnostic(args, trainer, data, selected, out):
    results = []
    all_tickers = sorted(selected['Ticker'].unique())
    for count, ticker in enumerate(all_tickers, 1):
        frame = data.loc[data['Ticker'].eq(ticker)].sort_values('Date').reset_index(drop=True)
        chosen = selected.loc[selected['Ticker'].eq(ticker)].iloc[0]
        for arch in ('MLP', 'LSTM', 'GRU'):
            checkpoint_path = Path(args.checkpoints) / f'{ticker}_{arch}.pt'
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f'Missing original checkpoint: {checkpoint_path}')
            ckpt = get_checkpoint(checkpoint_path)
            cfg = ckpt['spec']
            if str(cfg['Model']) != arch:
                raise ValueError(f'Architecture mismatch for {ticker}, {arch}')
            features = ckpt['feature_names']
            expected_features = [c for c in data.columns if c not in trainer.META]
            if features != expected_features:
                raise ValueError(f'Predictor order differs from original checkpoint on {ticker}')
            x_mean = np.asarray(ckpt['x_mean'], dtype=np.float32)
            x_std = np.asarray(ckpt['x_std'], dtype=np.float32)
            y_mean, y_std = float(ckpt['y_mean']), float(ckpt['y_std'])
            arrays = trainer.get_arrays(frame, features, x_mean, x_std, y_mean, y_std,
                                        int(cfg['Q']), str(cfg['Model']))
            net = trainer.ForecastNet(arch, len(features), int(cfg['Hidden']))
            net.load_state_dict(ckpt['model_state'])
            net.eval()
            for split_name in ('Training', 'Validation', 'Test'):
                s = score_split(arrays, net, y_mean, y_std, trainer, split_name)
                results.append(dict(Ticker=ticker, Model=arch, Config=str(cfg['Config']),
                                    SelectedDL=(arch==str(chosen['Model'])),
                                    OriginalSeed=int(ckpt['training_seed']),
                                    Sample=split_name, **s))
        print(f'[{count:2d}/{len(all_tickers)}] {ticker}: evaluated 3 original checkpoints', flush=True)
    all_scores = pd.DataFrame(results)
    all_scores.to_csv(out / 'all_architectures_train_validation_test.csv', index=False)
    best_scores = all_scores.loc[all_scores['SelectedDL']].copy()
    best_scores.to_csv(out / 'selected_DL_train_validation_test_long.csv', index=False)
    wide = best_scores.pivot(index=['Ticker','Model','Config'], columns='Sample',
                             values=['RMSEGain','RelRMSE','RMSE','ZeroRMSE','N'])
    wide.columns = ['_'.join(pair) for pair in wide.columns]
    wide = wide.reset_index()
    wide['TrainValGap_pp'] = wide['RMSEGain_Training'] - wide['RMSEGain_Validation']
    wide['ValTestGap_pp'] = wide['RMSEGain_Validation'] - wide['RMSEGain_Test']
    wide['TrainBetterThanZero_ValWorse'] = ((wide['RMSEGain_Training'] > 0) &
                                            (wide['RMSEGain_Validation'] < 0))
    wide.to_csv(out / 'selected_DL_overfitting_gaps.csv', index=False)
    print('\nOriginal checkpoints evaluated. Saved output in:', out, flush=True)
    print('Model configurations:', len(all_scores)//3, '| selected stocks:', len(wide), flush=True)
    print('Median train minus validation gain gap (percentage points):',
          round(float(wide['TrainValGap_pp'].median()), 4), flush=True)
    print('Number with training gain > 0 and validation gain < 0:',
          int(wide['TrainBetterThanZero_ValWorse'].sum()), flush=True)
    return wide


def fit_curve(trainer, frame, features, cfg, ckpt, max_epochs, patience, batch_size):
    x_mean = np.asarray(ckpt['x_mean'], dtype=np.float32)
    x_std = np.asarray(ckpt['x_std'], dtype=np.float32)
    y_mean, y_std = float(ckpt['y_mean']), float(ckpt['y_std'])
    arrays = trainer.get_arrays(frame, features, x_mean, x_std, y_mean, y_std,
                                int(cfg['Q']), cfg['Model'])
    seed = int(ckpt['training_seed'])
    trainer.seed_everything(seed)
    net = trainer.ForecastNet(cfg['Model'], len(features), int(cfg['Hidden']))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    tx = torch.from_numpy(arrays['Training']['x'])
    ty = torch.from_numpy(arrays['Training']['y'].astype(np.float32))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(tx, ty), shuffle=True, batch_size=batch_size,
                        generator=generator)
    history = []
    best = np.inf
    stalled = 0
    for epoch in range(1, max_epochs+1):
        net.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(net(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
        train_pred = trainer.predict_scaled(net, arrays['Training']['x'])
        val_pred = trainer.predict_scaled(net, arrays['Validation']['x'])
        train_mse = float(np.mean((arrays['Training']['y']-train_pred)**2))
        val_mse = float(np.mean((arrays['Validation']['y']-val_pred)**2))
        history.append(dict(Epoch=epoch, TrainMSE_scaled=train_mse, ValidationMSE_scaled=val_mse,
                            TrainRMSE_scaled=np.sqrt(train_mse), ValidationRMSE_scaled=np.sqrt(val_mse)))
        if val_mse < best - 1e-8:
            best = val_mse
            stalled = 0
        else:
            stalled += 1
            if stalled >= patience:
                break
    return history


def learning_curves(args, trainer, data, selected, out):
    names = [s.strip().upper() for s in args.curve_tickers.split(',') if s.strip()]
    if not names:
        return
    all_tickers = set(selected['Ticker'])
    if not set(names).issubset(all_tickers):
        raise ValueError('One or more --curve-tickers not in selected model results')
    histories = []
    for ticker in names:
        chosen = selected.loc[selected['Ticker'].eq(ticker)].iloc[0]
        arch = str(chosen['Model'])
        ckpt = get_checkpoint(Path(args.checkpoints) / f'{ticker}_{arch}.pt')
        cfg = ckpt['spec']
        features = ckpt['feature_names']
        frame = data.loc[data['Ticker'].eq(ticker)].sort_values('Date').reset_index(drop=True)
        history = fit_curve(trainer, frame, features, cfg, ckpt, args.epochs,
                            args.patience, args.batch_size)
        for row in history:
            histories.append(dict(Ticker=ticker, Model=arch, Config=cfg['Config'],
                                  Seed=ckpt['training_seed'], **row))
        print(f'Learning curve: {ticker} {arch}, {len(history)} epochs', flush=True)
    pd.DataFrame(histories).to_csv(out/'illustrative_learning_curves.csv', index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--selected', required=True)
    p.add_argument('--checkpoints', required=True)
    p.add_argument('--trainer', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--curve-tickers', default='', help='optional: AMT,NVDA,MSFT,KO')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=128)
    args = p.parse_args()
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    trainer = import_trainer(args.trainer)
    data = pd.read_csv(args.input, parse_dates=['Date','TargetDate'])
    selected = pd.read_csv(args.selected)
    required = trainer.META
    if not required.issubset(set(data.columns)):
        raise ValueError('Incomplete input metadata')
    if set(data['Sample']) != {'Training','Validation','Test'}:
        raise ValueError('Unexpected split labels')
    if len(selected)!=33 or selected['Ticker'].nunique()!=33:
        raise ValueError('Expected one selected DL architecture for each of 33 stocks')
    if set(selected['Ticker']) != set(data['Ticker']):
        raise ValueError('Input universe and selected models differ')
    if len([c for c in data.columns if c not in trainer.META]) != 28:
        raise ValueError('Predictor count is not 28')
    print('Inspecting original checkpoints: NO training or model selection.', flush=True)
    checkpoint_diagnostic(args, trainer, data, selected, out)
    if args.curve_tickers:
        print('OPTIONAL illustrative training-only reruns (not replacing original weights).', flush=True)
        learning_curves(args, trainer, data, selected, out)
    print('DONE. No original data, weights or forecasts modified.', flush=True)


if __name__=='__main__':
    main()
