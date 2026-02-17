"""Experiment script for Question 3 — ElasticNet via SGD."""

import numpy as np
import pandas as pd
from elastic import ElasticNet, loss
from q2 import preprocess_data

np.random.seed(42)

# ── Load and preprocess data ────────────────────────────────────────────
DATA = "/Users/felipecardozo/Desktop/Aulas/534/HW1_export/energydata"
train_df = pd.read_csv(f"{DATA}/energy_train.csv")
val_df = pd.read_csv(f"{DATA}/energy_val.csv")
test_df = pd.read_csv(f"{DATA}/energy_test.csv")

trainy = train_df["Appliances"].values.astype(float)
valy = val_df["Appliances"].values.astype(float)
testy = test_df["Appliances"].values.astype(float)

trainx = train_df.drop(columns=["date", "Appliances"]).values.astype(float)
valx = val_df.drop(columns=["date", "Appliances"]).values.astype(float)
testx = test_df.drop(columns=["date", "Appliances"]).values.astype(float)

feature_names = [c for c in train_df.columns if c not in ("date", "Appliances")]

trainx, valx, testx = preprocess_data(trainx, valx, testx)

N_TRAIN = trainx.shape[0]
print(f"Train: {N_TRAIN} samples, {trainx.shape[1]} features")


def compute_metrics(model, tx, ty, vx, vy, sx, sy):
    results = {}
    for prefix, x, y in [("train", tx, ty), ("val", vx, vy), ("test", sx, sy)]:
        pred = model.predict(x)
        mse = np.mean((y - pred) ** 2)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        results[f"{prefix}-rmse"] = np.sqrt(mse)
        results[f"{prefix}-r2"] = 1.0 - ss_res / ss_tot
    return results


# ── Q3f: Learning rate search ───────────────────────────────────────────
# From Q2: lambda_ridge = 0.0001, lambda_lasso = 3.1623
# We use el = 3.1623 (lasso optimal) with alpha = 0.5 as the primary config
EL = 3.1623
ALPHA = 0.5
BATCH = 32
EPOCHS = 500

print("\n" + "=" * 75)
print(f"Q3f: Learning rate search  (el={EL}, alpha={ALPHA}, "
      f"batch={BATCH}, epochs={EPOCHS})")
print("=" * 75)

etas = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

lr_results = {}
for eta in etas:
    np.random.seed(42)
    model = ElasticNet(el=EL, alpha=ALPHA, eta=eta, batch=BATCH, epoch=EPOCHS)
    hist = model.train(trainx, trainy)
    last_loss = hist[EPOCHS - 1]
    diverged = np.isinf(last_loss) or np.isnan(last_loss) or last_loss > 1e15
    metrics = None if diverged else compute_metrics(
        model, trainx, trainy, valx, valy, testx, testy
    )
    lr_results[eta] = {
        "hist": hist,
        "metrics": metrics,
        "diverged": diverged,
        "model": model,
    }
    status = "DIVERGED" if diverged else "OK"
    m = metrics
    print(f"  eta={eta:.0e}: loss[0]={hist[0]:.2e}  loss[{EPOCHS-1}]={last_loss:.2e}"
          f"  [{status}]")
    if not diverged and m:
        print(f"           train RMSE={m['train-rmse']:.2f}  R2={m['train-r2']:.4f}"
              f"   val RMSE={m['val-rmse']:.2f}  R2={m['val-r2']:.4f}"
              f"   test RMSE={m['test-rmse']:.2f}  R2={m['test-r2']:.4f}")

# Best eta by validation RMSE
valid_etas = {e: r for e, r in lr_results.items() if not r["diverged"]}
best_eta = min(valid_etas, key=lambda e: valid_etas[e]["metrics"]["val-rmse"])
best_m = valid_etas[best_eta]["metrics"]
print(f"\n>>> Best eta = {best_eta:.0e}  "
      f"(val RMSE = {best_m['val-rmse']:.2f})")

# Loss curve for best eta
print(f"\nLoss curve at eta={best_eta:.0e}:")
bh = lr_results[best_eta]["hist"]
for ep in [0, 9, 49, 99, 100, 199, 299, 499]:
    if ep in bh:
        print(f"  Epoch {ep:4d}: {bh[ep]:.2e}")

# ── Q3f (supplement): el = lambda_ridge ─────────────────────────────────
print("\n" + "=" * 75)
print("Q3f (supplement): el=0.0001 (ridge optimal), alpha=0.5, "
      f"batch={BATCH}, epochs={EPOCHS}")
print("=" * 75)
EL_RIDGE = 0.0001
for eta in etas:
    np.random.seed(42)
    model = ElasticNet(el=EL_RIDGE, alpha=ALPHA, eta=eta, batch=BATCH, epoch=EPOCHS)
    hist = model.train(trainx, trainy)
    last = hist[EPOCHS - 1]
    div = np.isinf(last) or np.isnan(last) or last > 1e15
    tag = "DIVERGED" if div else "OK"
    if not div:
        m = compute_metrics(model, trainx, trainy, valx, valy, testx, testy)
        print(f"  eta={eta:.0e}: loss={last:.2e}  "
              f"train={m['train-rmse']:.2f}  val={m['val-rmse']:.2f}  "
              f"test={m['test-rmse']:.2f}  R2_val={m['val-r2']:.4f}")
    else:
        print(f"  eta={eta:.0e}: {tag}")

# ── Q3g: Alpha sweep ────────────────────────────────────────────────────
print("\n" + "=" * 75)
print(f"Q3g: Alpha sweep  (el={EL}, eta={best_eta:.0e}, "
      f"batch={BATCH}, epochs={EPOCHS})")
print("=" * 75)

alphas_to_test = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

print(f"{'alpha':>6}  {'train-RMSE':>10} {'train-R2':>10} "
      f"{'val-RMSE':>10} {'val-R2':>10} {'test-RMSE':>10} {'test-R2':>10}")
print("-" * 80)

alpha_results = {}
for a in alphas_to_test:
    np.random.seed(42)
    model = ElasticNet(el=EL, alpha=a, eta=best_eta, batch=BATCH, epoch=EPOCHS)
    hist = model.train(trainx, trainy)
    last = hist[EPOCHS - 1]
    div = np.isinf(last) or np.isnan(last) or last > 1e15
    if div:
        print(f"{a:>6.2f}  DIVERGED")
        alpha_results[a] = None
    else:
        m = compute_metrics(model, trainx, trainy, valx, valy, testx, testy)
        alpha_results[a] = {"metrics": m, "coefs": model.coef().copy(), "hist": hist}
        print(f"{a:>6.2f}  {m['train-rmse']:>10.2f} {m['train-r2']:>10.4f} "
              f"{m['val-rmse']:>10.2f} {m['val-r2']:>10.4f} "
              f"{m['test-rmse']:>10.2f} {m['test-r2']:>10.4f}")

# ── Q3i: Coefficient analysis ───────────────────────────────────────────
print("\n" + "=" * 75)
print("Q3i: Coefficient comparison (best-test vs best-val)")
print("=" * 75)

best_test_a, best_test_rmse = None, float("inf")
best_val_a, best_val_rmse = None, float("inf")
for a, r in alpha_results.items():
    if r is None:
        continue
    if r["metrics"]["test-rmse"] < best_test_rmse:
        best_test_rmse = r["metrics"]["test-rmse"]
        best_test_a = a
    if r["metrics"]["val-rmse"] < best_val_rmse:
        best_val_rmse = r["metrics"]["val-rmse"]
        best_val_a = a

print(f"Best on test:       alpha={best_test_a}  "
      f"(test RMSE={best_test_rmse:.2f})")
print(f"Best on validation: alpha={best_val_a}  "
      f"(val RMSE={best_val_rmse:.2f})")

coefs_best_test = alpha_results[best_test_a]["coefs"]
coefs_best_val = alpha_results[best_val_a]["coefs"]

print(f"\n{'Feature':<14} {'Best-Test':>12} {'Best-Val':>12} {'Diff':>10}")
print("-" * 52)
for i, name in enumerate(feature_names):
    ct = coefs_best_test[i]
    cv = coefs_best_val[i]
    print(f"{name:<14} {ct:>12.4f} {cv:>12.4f} {ct - cv:>10.4f}")

nz_t = np.sum(np.abs(coefs_best_test) < 1e-10)
nz_v = np.sum(np.abs(coefs_best_val) < 1e-10)
print(f"\nZero coefs:  best-test={nz_t}/25,  best-val={nz_v}/25")
print(f"||beta||_2:  best-test={np.linalg.norm(coefs_best_test):.2f}, "
      f" best-val={np.linalg.norm(coefs_best_val):.2f}")

# ── Q3h: Comparison with sklearn ────────────────────────────────────────
print("\n" + "=" * 75)
print("Q3h: SGD ElasticNet vs sklearn Ridge/Lasso")
print("=" * 75)

from sklearn.linear_model import Ridge, Lasso

sk_models = {
    "sklearn Ridge(0.0001)": Ridge(alpha=0.0001).fit(trainx, trainy),
    "sklearn Lasso(3.1623)": Lasso(alpha=3.1623, max_iter=10000).fit(trainx, trainy),
}

print(f"\n{'Model':<30} {'train-RMSE':>10} {'train-R2':>10} "
      f"{'val-RMSE':>10} {'val-R2':>10} {'test-RMSE':>10} {'test-R2':>10}")
print("-" * 100)

for name, sk in sk_models.items():
    vals = {}
    for prefix, x, y in [("train", trainx, trainy),
                          ("val", valx, valy),
                          ("test", testx, testy)]:
        pred = sk.predict(x)
        rmse = np.sqrt(np.mean((y - pred) ** 2))
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        vals[f"{prefix}-rmse"] = rmse
        vals[f"{prefix}-r2"] = r2
    print(f"{name:<30} {vals['train-rmse']:>10.2f} {vals['train-r2']:>10.4f} "
          f"{vals['val-rmse']:>10.2f} {vals['val-r2']:>10.4f} "
          f"{vals['test-rmse']:>10.2f} {vals['test-r2']:>10.4f}")

for a_val, label in [(1.0, "SGD alpha=1 (Ridge-like)"),
                      (0.0, "SGD alpha=0 (Lasso-like)"),
                      (0.5, "SGD alpha=0.5 (ElasticNet)")]:
    if alpha_results.get(a_val) is not None:
        m = alpha_results[a_val]["metrics"]
        print(f"{label:<30} {m['train-rmse']:>10.2f} {m['train-r2']:>10.4f} "
              f"{m['val-rmse']:>10.2f} {m['val-r2']:>10.4f} "
              f"{m['test-rmse']:>10.2f} {m['test-r2']:>10.4f}")

# Coefficient norm comparison
print("\nCoefficient norms:")
for name, sk in sk_models.items():
    print(f"  {name}: ||beta||_2 = {np.linalg.norm(sk.coef_):.2f}, "
          f"||beta||_1 = {np.sum(np.abs(sk.coef_)):.2f}")
for a_val, label in [(1.0, "SGD alpha=1"), (0.0, "SGD alpha=0"),
                      (0.5, "SGD alpha=0.5")]:
    if alpha_results.get(a_val) is not None:
        c = alpha_results[a_val]["coefs"]
        print(f"  {label}: ||beta||_2 = {np.linalg.norm(c):.2f}, "
              f"||beta||_1 = {np.sum(np.abs(c)):.2f}")
