"""Analysis script for Question 2 -- loads data, runs all experiments, prints tables."""

import numpy as np
import pandas as pd
from q2 import (
    preprocess_data,
    eval_linear1,
    eval_linear2,
    eval_ridge1,
    eval_ridge2,
    eval_lasso1,
    eval_lasso2,
)

# ── Load data ────────────────────────────────────────────────────────────
DATA = "/Users/felipecardozo/Desktop/Aulas/534/HW1_export/energydata"
train_df = pd.read_csv(f"{DATA}/energy_train.csv")
val_df = pd.read_csv(f"{DATA}/energy_val.csv")
test_df = pd.read_csv(f"{DATA}/energy_test.csv")

# Separate target (Appliances) and drop date column
trainy = train_df["Appliances"].values.astype(float)
valy = val_df["Appliances"].values.astype(float)
testy = test_df["Appliances"].values.astype(float)

trainx = train_df.drop(columns=["date", "Appliances"]).values.astype(float)
valx = val_df.drop(columns=["date", "Appliances"]).values.astype(float)
testx = test_df.drop(columns=["date", "Appliances"]).values.astype(float)

feature_names = [c for c in train_df.columns if c not in ("date", "Appliances")]

# Preprocess (standardise)
trainx, valx, testx = preprocess_data(trainx, valx, testx)

# ── Part 2c & 2d: Linear Regression ─────────────────────────────────────
print("=" * 70)
print("PART 2e: Linear Regression comparison")
print("=" * 70)
lr1 = eval_linear1(trainx, trainy, valx, valy, testx, testy)
lr2 = eval_linear2(trainx, trainy, valx, valy, testx, testy)

header = f"{'Metric':<12} {'LR1 (train only)':>18} {'LR2 (train+val)':>18}"
print(header)
print("-" * len(header))
for key in ['train-rmse', 'train-r2', 'val-rmse', 'val-r2', 'test-rmse', 'test-r2']:
    print(f"{key:<12} {lr1[key]:>18.4f} {lr2[key]:>18.4f}")

# ── Part 2h: Ridge & Lasso across lambda values ─────────────────────────
alphas = np.logspace(-4, 2, 25)  # 25 values from 1e-4 to 1e2

print("\n" + "=" * 70)
print("PART 2h: Ridge Regression across alpha values (train only)")
print("=" * 70)
ridge_results = {}
for a in alphas:
    ridge_results[a] = eval_ridge1(trainx, trainy, valx, valy, testx, testy, a)

print(f"{'alpha':>10}  {'train-RMSE':>10} {'train-R2':>10} "
      f"{'val-RMSE':>10} {'val-R2':>10} {'test-RMSE':>10} {'test-R2':>10}")
print("-" * 80)
for a in alphas:
    r = ridge_results[a]
    print(f"{a:>10.4f}  {r['train-rmse']:>10.4f} {r['train-r2']:>10.4f} "
          f"{r['val-rmse']:>10.4f} {r['val-r2']:>10.4f} "
          f"{r['test-rmse']:>10.4f} {r['test-r2']:>10.4f}")

best_ridge_alpha = min(alphas, key=lambda a: ridge_results[a]['val-rmse'])
print(f"\n>>> Optimal Ridge alpha (by val-RMSE): {best_ridge_alpha:.6f}")
print(f"    Val-RMSE = {ridge_results[best_ridge_alpha]['val-rmse']:.4f}")

print("\n" + "=" * 70)
print("PART 2h: Lasso Regression across alpha values (train only)")
print("=" * 70)
lasso_results = {}
for a in alphas:
    lasso_results[a] = eval_lasso1(trainx, trainy, valx, valy, testx, testy, a)

print(f"{'alpha':>10}  {'train-RMSE':>10} {'train-R2':>10} "
      f"{'val-RMSE':>10} {'val-R2':>10} {'test-RMSE':>10} {'test-R2':>10}")
print("-" * 80)
for a in alphas:
    r = lasso_results[a]
    print(f"{a:>10.4f}  {r['train-rmse']:>10.4f} {r['train-r2']:>10.4f} "
          f"{r['val-rmse']:>10.4f} {r['val-r2']:>10.4f} "
          f"{r['test-rmse']:>10.4f} {r['test-r2']:>10.4f}")

best_lasso_alpha = min(alphas, key=lambda a: lasso_results[a]['val-rmse'])
print(f"\n>>> Optimal Lasso alpha (by val-RMSE): {best_lasso_alpha:.6f}")
print(f"    Val-RMSE = {lasso_results[best_lasso_alpha]['val-rmse']:.4f}")

# ── Part 2j: Ridge2 & Lasso2 with optimal alpha ─────────────────────────
print("\n" + "=" * 70)
print("PART 2j: Ridge2 & Lasso2 (train+val) with optimal alphas")
print("=" * 70)
ridge2 = eval_ridge2(trainx, trainy, valx, valy, testx, testy, best_ridge_alpha)
lasso2 = eval_lasso2(trainx, trainy, valx, valy, testx, testy, best_lasso_alpha)

header = f"{'Metric':<12} {'Ridge2':>12} {'Lasso2':>12}"
print(header)
print("-" * len(header))
for key in ['train-rmse', 'train-r2', 'val-rmse', 'val-r2', 'test-rmse', 'test-r2']:
    print(f"{key:<12} {ridge2[key]:>12.4f} {lasso2[key]:>12.4f}")

# ── Part 2k: Coefficient paths ──────────────────────────────────────────
print("\n" + "=" * 70)
print("PART 2k: Coefficient paths")
print("=" * 70)

from sklearn.linear_model import Ridge as RidgeSK, Lasso as LassoSK

path_alphas = np.logspace(-4, 2, 50)

ridge_coefs = []
for a in path_alphas:
    m = RidgeSK(alpha=a).fit(trainx, trainy)
    ridge_coefs.append(m.coef_)
ridge_coefs = np.array(ridge_coefs)

lasso_coefs = []
for a in path_alphas:
    m = LassoSK(alpha=a, max_iter=10000).fit(trainx, trainy)
    lasso_coefs.append(m.coef_)
lasso_coefs = np.array(lasso_coefs)

print("\nRidge coefficients at alpha=0.0001 (smallest) vs alpha=100 (largest):")
print(f"{'Feature':<16} {'alpha=1e-4':>12} {'alpha=100':>12}")
print("-" * 44)
for i, name in enumerate(feature_names):
    print(f"{name:<16} {ridge_coefs[0, i]:>12.4f} {ridge_coefs[-1, i]:>12.4f}")

print("\nLasso coefficients at alpha=0.0001 (smallest) vs alpha=100 (largest):")
print(f"{'Feature':<16} {'alpha=1e-4':>12} {'alpha=100':>12}")
print("-" * 44)
for i, name in enumerate(feature_names):
    print(f"{name:<16} {lasso_coefs[0, i]:>12.4f} {lasso_coefs[-1, i]:>12.4f}")

# Count zero coefficients for Lasso at different alphas
print("\nLasso: number of zero coefficients at each alpha:")
for idx in [0, 12, 24, 36, 49]:
    a = path_alphas[idx]
    n_zero = np.sum(np.abs(lasso_coefs[idx]) < 1e-10)
    print(f"  alpha={a:>10.4f} -> {n_zero}/{len(feature_names)} features zeroed out")

# Identify which features Lasso zeros out at optimal alpha
print(f"\nLasso features zeroed out at optimal alpha={best_lasso_alpha:.6f}:")
m_opt = LassoSK(alpha=best_lasso_alpha, max_iter=10000).fit(trainx, trainy)
for i, name in enumerate(feature_names):
    if abs(m_opt.coef_[i]) < 1e-10:
        print(f"  {name}")
n_zero_opt = np.sum(np.abs(m_opt.coef_) < 1e-10)
print(f"Total zeroed: {n_zero_opt}/{len(feature_names)}")
