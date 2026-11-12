#!/usr/bin/env python3
"""Seed stability for FROZEN validation-selected DL architectures/hyperparameters.

This script imports exactly the model/preprocessing/training routines from the
original train_dl_33.py. It NEVER retunes architectures, sequence lengths or
hyperparameters on the test set. All new seeds are chosen in advance.
Outputs one CSV per completed (ticker, seed), so reruns resume automatically.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


def atomic_csv(df, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.tmp')
    df.to_csv(tmp, index=False)
    os.replace(tmp, target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--selected', required=True)
    p.add_argument('--training-script', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seeds', default='101,202,303')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--tickers', default='')  # optional technical smoke check
    args = p.parse_args()

    requested_seeds = [int(s.strip()) for s in args.seeds.split(',')]
    if len(requested_seeds) != len(set(requested_seeds)) or not requested_seeds:
        raise ValueError('Seed list must have distinct entries')
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))

    spec = importlib.util.spec_from_file_location('original_thesis_trainer', args.training_script)
    if spec is None or spec.loader is None:
        raise ValueError('Could not load the frozen training script')
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)

    dataset = pd.read_csv(args.input, parse_dates=['Date', 'TargetDate'])
    selected = pd.read_csv(args.selected)
    if len(selected) != 33 and not args.tickers:
        raise ValueError('Expected 33 frozen validation-selected DL models')
    if selected.Ticker.duplicated().any():
        raise ValueError('Duplicate selected model tickers')
    features = [c for c in dataset.columns if c not in trainer.META]
    if len(features) != 28:
        raise ValueError(f'Expected 28 predictors; found {len(features)}')
    if not np.isfinite(dataset[features + ['Target']].to_numpy(dtype=float)).all():
        raise ValueError('Dataset contains non-finite predictors/targets')
    if args.tickers:
        keep = [t.strip() for t in args.tickers.split(',') if t.strip()]
        selected = selected[selected.Ticker.isin(keep)].copy()
        if set(selected.Ticker) != set(keep):
            raise ValueError('Ticker not found among selected DL models')

    outdir = Path(args.output).resolve()
    reports_dir = outdir / 'per_seed_scores'
    preds_dir = outdir / 'per_seed_predictions'
    reports_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    print(f'Frozen configurations: {len(selected)} stocks | new seeds: {requested_seeds}', flush=True)
    print('No architecture/hyperparameter reselection. Test set only scored after training.', flush=True)

    for ii, row in enumerate(selected.itertuples(index=False), 1):
        ticker = str(row.Ticker)
        frame = dataset[dataset.Ticker == ticker].copy().sort_values('Date').reset_index(drop=True)
        if frame.empty:
            raise ValueError(f'Missing input for {ticker}')
        cfg = {'Model': str(row.Model), 'Config': str(row.Config),
               'Hidden': int(row.Hidden), 'Q': int(row.Q)}
        train = frame.Sample.eq('Training')
        xtrain = frame.loc[train, features].to_numpy(dtype=np.float64)
        x_mean = xtrain.mean(axis=0).astype(np.float32)
        x_std = xtrain.std(axis=0, ddof=1).astype(np.float32)
        x_std[(~np.isfinite(x_std)) | (x_std < 1e-12)] = 1.0
        ytrain = frame.loc[train, 'Target'].to_numpy(dtype=np.float64)
        y_mean, y_std = float(ytrain.mean()), float(ytrain.std(ddof=1))
        if y_std <= 0 or not np.isfinite(y_std):
            raise ValueError(f'Target variance invalid for {ticker}')
        arrays = trainer.get_arrays(frame, features, x_mean, x_std, y_mean, y_std,
                                    cfg['Q'], cfg['Model'])
        test_data = frame.iloc[arrays['Test']['idx']]
        if len(test_data) != 1508:
            raise ValueError(f'Test coverage changed for {ticker}: {len(test_data)}')

        for seed in requested_seeds:
            report = reports_dir / f'{ticker}_seed{seed}.csv'
            prediction_path = preds_dir / f'{ticker}_seed{seed}.csv'
            if report.exists() and prediction_path.exists():
                print(f'[{ii}/{len(selected)}] {ticker} seed {seed}: SKIP (saved)', flush=True)
                continue
            model, epoch = trainer.fit_one(
                cfg, arrays, len(features), seed, args.epochs, args.patience, args.batch_size)
            val_pred = trainer.predict_scaled(model, arrays['Validation']['x']) * y_std + y_mean
            test_pred = trainer.predict_scaled(model, arrays['Test']['x']) * y_std + y_mean
            val_actual = arrays['Validation']['actual']
            test_actual = arrays['Test']['actual']
            v_rmse = float(np.sqrt(np.mean((val_actual - val_pred)**2)))
            t_rmse = float(np.sqrt(np.mean((test_actual - test_pred)**2)))
            t_mae = float(np.mean(np.abs(test_actual - test_pred)))
            result = pd.DataFrame([{
                'Ticker': ticker, 'Model': cfg['Model'], 'Config': cfg['Config'],
                'Q': cfg['Q'], 'Hidden': cfg['Hidden'], 'Seed': seed,
                'BestEpoch': epoch, 'ValidationRMSE': v_rmse,
                'RMSE_DL': t_rmse, 'MAE_DL': t_mae,
                'N_Test': len(test_actual),
            }])
            prediction = pd.DataFrame({
                'Ticker': ticker,
                'Seed': seed,
                'ForecastOrigin': test_data.Date.dt.strftime('%Y-%m-%d').to_numpy(),
                'TargetDate': test_data.TargetDate.dt.strftime('%Y-%m-%d').to_numpy(),
                'ActualReturn': test_actual,
                'Prediction': test_pred,
            })
            # Save prediction first, then the score (both required for "skip").
            atomic_csv(prediction, prediction_path)
            atomic_csv(result, report)
            print(f'[{ii}/{len(selected)}] {ticker} seed {seed}: '
                  f'val RMSE={v_rmse:.6f} test RMSE={t_rmse:.6f}', flush=True)

    score_files = sorted(reports_dir.glob('*_seed*.csv'))
    pred_files = sorted(preds_dir.glob('*_seed*.csv'))
    scores = pd.concat([pd.read_csv(f) for f in score_files], ignore_index=True)
    preds = pd.concat([pd.read_csv(f) for f in pred_files], ignore_index=True)
    atomic_csv(scores, outdir / 'dl_seed_stability_scores.csv')
    atomic_csv(preds, outdir / 'dl_seed_stability_predictions.csv')
    expected = len(selected) * len(requested_seeds)
    if len(scores) != expected or len(preds) != expected * 1508:
        raise RuntimeError(f'Unexpected output counts; got {len(scores)} scores, '
                           f'{len(preds)} predictions; expected {expected} and {expected*1508}')
    print(f'DONE. Saved {len(scores)} new training runs and {len(preds)} forecasts.', flush=True)


if __name__ == '__main__':
    main()
