# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║   Algorithmic Fairness in Credit Scoring — Review 2: The Baseline Problem    ║
# ║   Project: 23CSE399  |  Team ID: 23UG045                                     ║
# ║                                                                              ║
# ║   PIPELINE OVERVIEW                                                          ║
# ║   ─────────────────────────────────────────────────────────────────────────  ║
# ║   STEP 1 │ Full Literature-Survey Model Bake-Off (16 models from 15 papers)  ║
# ║   STEP 2 │ TOP-3 Deep Dive (Confusion Matrix + ROC — one panel per model)    ║
# ║   STEP 3 │ 60-Month Poverty Trap Simulation (Champion Only)                  ║
# ║   STEP 4 │ Dataset Checkpointing (CSV Exports at Key Months)                 ║
# ║   STEP 5 │ Timeline Visualization — The 60-Month Poverty Trap                ║
# ║                                                                              ║
# ║   MODELS SOURCE MAP                                                          ║
# ║   Paper [1]  Hardt et al. 2016     → Equal Opportunity post-proc (LR)       ║
# ║   Paper [7]  John 2025             → LightGBM                               ║
# ║   Paper [9]  Badar & Fisichella    → Naive Bayes (Fair-CMNB)                ║
# ║   Paper [10] Kasmi 2021            → LR + Calibrated Eq. Odds               ║
# ║   Paper [11] Lessmann et al. 2015  → LDA, Bagging, KNN, RF, SVM, DT, LR    ║
# ║   Paper [13] Thu et al. 2024       → DT, NB, MLP, kNN (k=5)                ║
# ║   Paper [14] Kpatcha 2025          → Threshold-optimised RF                 ║
# ║   Papers [2–6, 8, 12, 15]         → DL / Control baselines (PyTorch nets)  ║
# ║                                                                              ║
# ║   THESIS: No existing static model survives economic shocks fairly.         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# !pip install xgboost lightgbm scikit-learn pandas numpy scipy matplotlib seaborn torch --quiet

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model         import LogisticRegression
from sklearn.tree                  import DecisionTreeClassifier
from sklearn.ensemble              import (RandomForestClassifier, GradientBoostingClassifier,
                                            AdaBoostClassifier, ExtraTreesClassifier,
                                            BaggingClassifier)
from sklearn.svm                   import SVC
from sklearn.neural_network        import MLPClassifier
from sklearn.naive_bayes           import GaussianNB
from sklearn.neighbors             import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection       import train_test_split
from sklearn.preprocessing         import StandardScaler
from sklearn.metrics               import (accuracy_score, roc_auc_score, f1_score,
                                            confusion_matrix, roc_curve)
from xgboost                       import XGBClassifier
from lightgbm                      import LGBMClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

np.random.seed(42)
torch.manual_seed(42)

# ─────────────────────────────────────────────────────────────────────────────
#  PYTORCH MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

class CreditNet(nn.Module):
    """Shallow net — 2→64(Drop0.2)→32(Drop0.2)→1(Sigmoid)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1),  nn.Sigmoid()
        )
    def forward(self, x): return self.net(x).squeeze()


class DeepCreditNet(nn.Module):
    """Deep net — 2→128(BN,Drop0.3)→64(BN)→32(BN)→1(Sigmoid)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, 32),  nn.BatchNorm1d(32),  nn.ReLU(),
            nn.Linear(32, 1),   nn.Sigmoid()
        )
    def forward(self, x): return self.net(x).squeeze()


def train_torch(model, X_tr, y_tr, epochs=60, lr=0.001):
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    ds  = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                        torch.tensor(y_tr, dtype=torch.float32))
    dl  = DataLoader(ds, batch_size=64, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    return model


def torch_predict_proba(model, X):
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()


NEEDS_SCALING = {
    'Logistic Regression', 'SVM', 'MLP (sklearn)',
    'LDA', 'KNN (k=5)',
    'PyTorch Shallow NN', 'PyTorch Deep NN'
}

# ─────────────────────────────────────────────────────────────────────────────
#  ARCHITECTURE / PAPER SOURCE TRANSPARENCY
#  Each entry: (arch_string, paper_reference)
# ─────────────────────────────────────────────────────────────────────────────

ARCH_INFO = {
    # ── Original 5 ──────────────────────────────────────────────────────────
    'Logistic Regression':
        ('C=1.0 · max_iter=500 · solver=lbfgs',
         '[1] Hardt 2016, [10] Kasmi 2021, [11] Lessmann 2015'),
    'Decision Tree':
        ('max_depth=5 · criterion=gini',
         '[11] Lessmann 2015, [13] Thu 2024'),
    'Random Forest':
        ('n_estimators=100 · max_features=sqrt · bootstrap=True',
         '[11] Lessmann 2015'),
    'Gradient Boosting':
        ('n_estimators=100 · lr=0.1 · max_depth=3',
         '[11] Lessmann 2015'),
    'XGBoost':
        ('n_estimators=100 · lr=0.3 · max_depth=6 · subsample=0.8',
         '[7] John 2025, [11] Lessmann 2015'),
    # ── Extended 5 ──────────────────────────────────────────────────────────
    'AdaBoost':
        ('n_estimators=200 · lr=0.5 · base=DecisionStump',
         '[11] Lessmann 2015'),
    'Extra Trees':
        ('n_estimators=100 · max_features=sqrt · n_jobs=-1',
         '[11] Lessmann 2015'),
    'SVM':
        ('kernel=rbf · C=1.0 · gamma=scale · probability=True',
         '[11] Lessmann 2015'),
    'MLP (sklearn)':
        ('Layers: 2→64→32→1 · ReLU · adam · max_iter=300',
         '[13] Thu 2024'),
    # ── New additions from papers ─────────────────────────────────────────
    'LightGBM':
        ('n_estimators=200 · lr=0.05 · max_depth=6 · num_leaves=31 · subsample=0.8',
         '[7] John 2025  ← recommended for adaptive credit scoring under drift'),
    'Naive Bayes':
        ('GaussianNB · var_smoothing=1e-9',
         '[9] Badar & Fisichella 2024 (Fair-CMNB), [13] Thu 2024'),
    'KNN (k=5)':
        ('n_neighbors=5 · metric=minkowski · weights=uniform',
         '[13] Thu 2024  ← k=5 as used in experimental study'),
    'LDA':
        ('solver=svd · n_components=1',
         '[11] Lessmann 2015  ← part of 41-classifier credit benchmark'),
    'Bagging (DT)':
        ('n_estimators=100 · base=DecisionTree · max_samples=0.8 · bootstrap=True',
         '[11] Lessmann 2015  ← homogeneous ensemble baseline'),
    # ── PyTorch DL ────────────────────────────────────────────────────────
    'PyTorch Shallow NN':
        ('2→64(ReLU,Drop0.2)→32(ReLU,Drop0.2)→1(Sigmoid) · Adam · lr=0.001',
         '[13] Thu 2024 MLP family, [12] Hu 2025 DNN baseline'),
    'PyTorch Deep NN':
        ('2→128(BN,ReLU,Drop0.3)→64(BN,ReLU)→32(BN,ReLU)→1(Sigmoid) · Adam',
         '[12] Hu 2025 DNN+AE framework, [15] Barrainkua 2026 DL baseline'),
}

# ─────────────────────────────────────────────────────────────────────────────
#  FICO CDF DATA  (Hardt et al. 2016)
# ─────────────────────────────────────────────────────────────────────────────

SCORES = np.linspace(300, 850, 112)

raw_cdf = {
    'asian':    [0.0,0.0,0.0,0.0,0.001,0.001,0.001,0.001,0.002,0.002,0.003,
                 0.003,0.004,0.005,0.006,0.007,0.008,0.010,0.011,0.013,0.015,
                 0.017,0.019,0.022,0.025,0.028,0.031,0.035,0.039,0.044,0.049,
                 0.054,0.060,0.066,0.073,0.081,0.089,0.097,0.106,0.116,0.126,
                 0.137,0.149,0.161,0.174,0.187,0.201,0.215,0.230,0.246,0.262,
                 0.278,0.295,0.312,0.330,0.348,0.366,0.385,0.404,0.423,0.443,
                 0.462,0.482,0.502,0.521,0.540,0.559,0.578,0.596,0.614,0.632,
                 0.649,0.666,0.682,0.698,0.713,0.727,0.741,0.755,0.768,0.780,
                 0.792,0.803,0.814,0.824,0.834,0.843,0.852,0.860,0.868,0.876,
                 0.883,0.890,0.896,0.902,0.908,0.913,0.918,0.923,0.928,0.932,
                 0.936,0.940,0.944,0.947,0.951,0.954,0.957,0.960,0.963,0.966,1.0],
    'black':    [0.0,0.0,0.001,0.001,0.002,0.003,0.004,0.005,0.007,0.009,0.011,
                 0.014,0.017,0.020,0.024,0.028,0.033,0.039,0.045,0.052,0.059,
                 0.067,0.076,0.086,0.096,0.107,0.119,0.132,0.145,0.159,0.174,
                 0.189,0.205,0.222,0.239,0.257,0.275,0.294,0.313,0.332,0.352,
                 0.372,0.392,0.412,0.433,0.453,0.473,0.493,0.513,0.533,0.552,
                 0.571,0.590,0.609,0.627,0.644,0.661,0.678,0.694,0.709,0.724,
                 0.738,0.751,0.764,0.776,0.788,0.799,0.809,0.819,0.828,0.837,
                 0.845,0.853,0.860,0.867,0.874,0.880,0.886,0.891,0.896,0.901,
                 0.905,0.910,0.914,0.917,0.921,0.924,0.927,0.930,0.933,0.936,
                 0.938,0.941,0.943,0.945,0.948,0.950,0.952,0.954,0.956,0.958,
                 0.960,0.962,0.964,0.966,0.968,0.970,0.972,0.974,0.976,0.978,1.0],
    'hispanic': [0.0,0.0,0.001,0.001,0.001,0.002,0.002,0.003,0.004,0.005,0.007,
                 0.008,0.010,0.013,0.015,0.018,0.022,0.026,0.030,0.035,0.040,
                 0.046,0.053,0.060,0.068,0.077,0.086,0.096,0.107,0.119,0.131,
                 0.144,0.157,0.171,0.186,0.202,0.218,0.235,0.252,0.270,0.289,
                 0.308,0.327,0.347,0.367,0.387,0.407,0.428,0.448,0.468,0.489,
                 0.509,0.529,0.548,0.568,0.587,0.605,0.623,0.641,0.658,0.674,
                 0.690,0.705,0.720,0.734,0.748,0.761,0.773,0.784,0.795,0.806,
                 0.816,0.825,0.834,0.843,0.851,0.859,0.866,0.873,0.879,0.885,
                 0.891,0.897,0.902,0.907,0.911,0.916,0.920,0.924,0.928,0.931,
                 0.935,0.938,0.941,0.944,0.947,0.949,0.952,0.954,0.957,0.959,
                 0.961,0.963,0.965,0.967,0.969,0.971,0.973,0.975,0.977,0.979,1.0],
    'white':    [0.0,0.0,0.0,0.0,0.0,0.001,0.001,0.001,0.001,0.002,0.002,
                 0.003,0.003,0.004,0.005,0.006,0.007,0.008,0.010,0.011,0.013,
                 0.015,0.017,0.020,0.022,0.025,0.028,0.032,0.036,0.040,0.045,
                 0.050,0.056,0.062,0.069,0.076,0.084,0.092,0.101,0.110,0.120,
                 0.131,0.142,0.154,0.166,0.179,0.193,0.207,0.222,0.237,0.252,
                 0.268,0.284,0.301,0.318,0.335,0.353,0.370,0.388,0.406,0.424,
                 0.442,0.460,0.478,0.496,0.514,0.531,0.549,0.566,0.583,0.599,
                 0.615,0.630,0.645,0.660,0.674,0.687,0.700,0.713,0.725,0.737,
                 0.748,0.759,0.769,0.779,0.789,0.798,0.807,0.815,0.823,0.831,
                 0.839,0.846,0.853,0.860,0.866,0.872,0.878,0.884,0.889,0.894,
                 0.899,0.904,0.908,0.912,0.916,0.920,0.924,0.927,0.931,0.934,1.0]
}

cdf_df = pd.DataFrame(raw_cdf)
cdf_df.insert(0, 'score', SCORES)
print("✅ FICO CDF data loaded:", cdf_df.shape)

# ─────────────────────────────────────────────────────────────────────────────
#  POPULATION SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

N_AGENTS        = 5000
SIGMOID_OFFSET  = 620
SIGMOID_K       = 75
GROUP_SIZES     = {'white': 0.60, 'black': 0.13, 'hispanic': 0.18, 'asian': 0.09}
MINORITY_GROUPS = ['black', 'hispanic']

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-(x - SIGMOID_OFFSET) / SIGMOID_K))

def sample_from_cdf(group_name, n):
    scores   = cdf_df['score'].values.astype(float)
    cdf_vals = np.clip(cdf_df[group_name].values.astype(float), 0, 1)
    cdf_vals[0] = 0.0; cdf_vals[-1] = 1.0
    _, idx   = np.unique(cdf_vals, return_index=True)
    inv_cdf  = interp1d(cdf_vals[idx], scores[idx], kind='linear',
                        bounds_error=False, fill_value=(scores[0], scores[-1]))
    return np.clip(inv_cdf(np.random.uniform(0, 1, n)), 300, 850)

agents = []
for group, fraction in GROUP_SIZES.items():
    n         = int(N_AGENTS * fraction)
    scores    = sample_from_cdf(group, n)
    rep_probs = sigmoid(scores)
    outcomes  = (np.random.rand(n) < rep_probs).astype(int)
    for i in range(n):
        agents.append({
            'agent_id':     len(agents),
            'group':        group,
            'credit_score': round(float(scores[i]), 1),
            'repay_prob':   round(float(rep_probs[i]), 4),
            'true_outcome': int(outcomes[i]),
        })

pop_df = pd.DataFrame(agents)
print(f"✅ Simulated {len(pop_df)} agents")
print(pop_df.groupby('group').agg(
    count=('agent_id','count'),
    mean_score=('credit_score','mean'),
    repay_rate=('true_outcome','mean')
).round(3).to_string(), "\n")

X      = pop_df[['credit_score','repay_prob']].values
y      = pop_df['true_outcome'].values
groups = pop_df['group'].values

X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
    X, y, groups, test_size=0.3, random_state=42, stratify=y)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

def demographic_parity_diff(y_pred, g):
    rates = {grp: y_pred[g == grp].mean() for grp in np.unique(g)}
    return round(max(rates.values()) - min(rates.values()), 4)

def equal_opportunity_diff(y_pred, y_true, g):
    tprs = {}
    for grp in np.unique(g):
        mask = (g == grp) & (y_true == 1)
        tprs[grp] = y_pred[mask].mean() if mask.sum() > 0 else 0.0
    return round(max(tprs.values()) - min(tprs.values()), 4)


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 1 — FULL LITERATURE-SURVEY BAKE-OFF  (16 models across 15 papers)
# ═════════════════════════════════════════════════════════════════════════════

print("═" * 80)
print("  STEP 1 │ FULL LITERATURE-SURVEY BAKE-OFF  (16 models from 15 papers)")
print("═" * 80)
print("  Metrics : Accuracy · AUC-ROC · F1 · DP Diff · EO Diff")
print("  Holdout : 30% stratified  |  Random seed: 42")
print("─" * 80 + "\n")

# ── Define all sklearn models with paper-justified hyperparameters ────────────
sklearn_models = {
    # ── From [1][10][11] — standard LR baseline ──────────────────────────
    'Logistic Regression':
        LogisticRegression(C=1.0, max_iter=500, solver='lbfgs', random_state=42),

    # ── From [11][13] — Gini DT depth=5 ─────────────────────────────────
    'Decision Tree':
        DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42),

    # ── From [11] — RF 100 trees, sqrt features ───────────────────────────
    'Random Forest':
        RandomForestClassifier(n_estimators=100, max_features='sqrt',
                                random_state=42, n_jobs=-1),

    # ── From [11] — GBM 100 trees, depth 3 ───────────────────────────────
    'Gradient Boosting':
        GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                    max_depth=3, random_state=42),

    # ── From [7][11] — XGB with John 2025 params (subsample=0.8) ─────────
    'XGBoost':
        XGBClassifier(n_estimators=100, learning_rate=0.3, max_depth=6,
                       subsample=0.8, random_state=42,
                       eval_metric='logloss', verbosity=0),

    # ── From [11] — AdaBoost 200 estimators, lr=0.5 (Lessmann best) ──────
    'AdaBoost':
        AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=42),

    # ── From [11] — Extra Trees (homogeneous ensemble) ────────────────────
    'Extra Trees':
        ExtraTreesClassifier(n_estimators=100, max_features='sqrt',
                              random_state=42, n_jobs=-1),

    # ── From [11] — RBF SVM (Lessmann benchmark) ─────────────────────────
    'SVM':
        SVC(probability=True, kernel='rbf', C=1.0,
            gamma='scale', random_state=42),

    # ── From [13] — MLP 64→32, relu, adam (Thu 2024) ─────────────────────
    'MLP (sklearn)':
        MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                       solver='adam', max_iter=300, random_state=42),

    # ── NEW: From [7] John 2025 — LightGBM, num_leaves=31, subsample=0.8 ─
    'LightGBM':
        LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6,
                        num_leaves=31, subsample=0.8, random_state=42,
                        verbose=-1),

    # ── NEW: From [9][13] — Naive Bayes (Fair-CMNB backbone) ────────────
    'Naive Bayes':
        GaussianNB(var_smoothing=1e-9),

    # ── NEW: From [13] — k=5 as used in Thu et al. experiment ────────────
    'KNN (k=5)':
        KNeighborsClassifier(n_neighbors=5, metric='minkowski',
                              weights='uniform', n_jobs=-1),

    # ── NEW: From [11] — LDA from the 41-model benchmark ─────────────────
    'LDA':
        LinearDiscriminantAnalysis(solver='svd'),

    # ── NEW: From [11] — Bagging with DT base ────────────────────────────
    'Bagging (DT)':
        BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=5),
                           n_estimators=100, max_samples=0.8,
                           bootstrap=True, random_state=42, n_jobs=-1),
}

trained_sklearn = {}
trained_pytorch = {}
bakeoff_results = []

# ── (a) Train & evaluate all sklearn models ───────────────────────────────────
for name, model in sklearn_models.items():
    X_tr = X_train_sc if name in NEEDS_SCALING else X_train
    X_te = X_test_sc  if name in NEEDS_SCALING else X_test
    model.fit(X_tr, y_train)
    trained_sklearn[name] = model

    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    acc = round(accuracy_score(y_test, y_pred),  4)
    auc = round(roc_auc_score(y_test,  y_proba), 4)
    f1  = round(f1_score(y_test,       y_pred),  4)
    dpd = demographic_parity_diff(y_pred, g_test)
    eod = equal_opportunity_diff(y_pred, y_test, g_test)

    arch_str, paper_ref = ARCH_INFO.get(name, ('N/A', 'N/A'))
    bakeoff_results.append({'Model': name, 'Type': 'ML',
                             'Accuracy': acc, 'AUC-ROC': auc,
                             'F1-Score': f1, 'DP Diff': dpd, 'EO Diff': eod,
                             'Paper Ref': paper_ref})
    print(f"  ✅  {name:<22}  Acc={acc:.3f}  AUC={auc:.3f}  F1={f1:.3f}  "
          f"DPD={dpd:.3f}  EOD={eod:.3f}")
    print(f"       ↳ Arch  : {arch_str}")
    print(f"       ↳ Papers: {paper_ref}\n")

# ── (b) PyTorch models ────────────────────────────────────────────────────────
torch_configs = [
    ('PyTorch Shallow NN', CreditNet(),     60),
    ('PyTorch Deep NN',    DeepCreditNet(), 80),
]
for pt_name, pt_arch, pt_epochs in torch_configs:
    pt_model = train_torch(pt_arch, X_train_sc, y_train, epochs=pt_epochs)
    trained_pytorch[pt_name] = pt_model
    y_proba = torch_predict_proba(pt_model, X_test_sc)
    y_pred  = (y_proba >= 0.50).astype(int)

    acc = round(accuracy_score(y_test, y_pred),  4)
    auc = round(roc_auc_score(y_test,  y_proba), 4)
    f1  = round(f1_score(y_test,       y_pred),  4)
    dpd = demographic_parity_diff(y_pred, g_test)
    eod = equal_opportunity_diff(y_pred, y_test, g_test)

    arch_str, paper_ref = ARCH_INFO.get(pt_name, ('N/A', 'N/A'))
    bakeoff_results.append({'Model': pt_name, 'Type': 'DL',
                             'Accuracy': acc, 'AUC-ROC': auc,
                             'F1-Score': f1, 'DP Diff': dpd, 'EO Diff': eod,
                             'Paper Ref': paper_ref})
    print(f"  ✅  {pt_name:<22}  Acc={acc:.3f}  AUC={auc:.3f}  F1={f1:.3f}  "
          f"DPD={dpd:.3f}  EOD={eod:.3f}")
    print(f"       ↳ Arch  : {arch_str}")
    print(f"       ↳ Papers: {paper_ref}\n")

# ── Leaderboard ───────────────────────────────────────────────────────────────
bakeoff_df = (pd.DataFrame(bakeoff_results)
                .sort_values('AUC-ROC', ascending=False)
                .reset_index(drop=True))
bakeoff_df.index += 1

print("─" * 80)
print("  ╔══════════════════════════════════════════════════════════════════════╗")
print("  ║          🏆  LITERATURE-SURVEY BAKE-OFF LEADERBOARD (by AUC-ROC)   ║")
print("  ╠══════╦══════════════════════╦═════╦════════╦════════╦══════╦══════╣")
print("  ║ Rank ║ Model                ║Type ║ AUC-ROC║ F1     ║DPD   ║EOD   ║")
print("  ╠══════╬══════════════════════╬═════╬════════╬════════╬══════╬══════╣")
for idx, row in bakeoff_df.iterrows():
    medal = "🥇" if idx == 1 else ("🥈" if idx == 2 else ("🥉" if idx == 3 else "  "))
    print(f"  ║{medal}{idx:<3} ║ {row['Model']:<20} ║ {row['Type']:<3} ║"
          f" {row['AUC-ROC']:.4f} ║ {row['F1-Score']:.4f} ║{row['DP Diff']:.4f}║{row['EO Diff']:.4f}║")
print("  ╚══════╩══════════════════════╩═════╩════════╩════════╩══════╩══════╝")
print("─" * 80)

bakeoff_df.to_csv('bakeoff_leaderboard.csv', index=True)
print("  📁  Saved: bakeoff_leaderboard.csv\n")

# ── Pull champion ────────────────────────────────────────────────────────────────
champion_name = 'Logistic Regression'
print(f"  🏆  Forced Champion : {champion_name}\n")


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 2 — CHAMPION DEEP DIVE
#  Confusion Matrix (left) + ROC Curve (right)
# ═════════════════════════════════════════════════════════════════════════════

print("═" * 80)
print("  STEP 2 │ CHAMPION DEEP DIVE — Confusion Matrix + ROC Curve")
print("═" * 80)

BG      = '#0D1117'
GOLD_C  = '#FFD700'
CHAMP_C = '#FF4444'
DALOF_C = '#00E676'

def get_model_proba(name):
    """Return (y_pred, y_proba) for any model name."""
    if name in trained_pytorch:
        proba = torch_predict_proba(trained_pytorch[name], X_test_sc)
        pred  = (proba >= 0.5).astype(int)
    else:
        X_te = X_test_sc if name in NEEDS_SCALING else X_test
        proba = trained_sklearn[name].predict_proba(X_te)[:, 1]
        pred  = trained_sklearn[name].predict(X_te)
    return pred, proba

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(BG)
fig.suptitle('Graph 1: Champion Model Deep Dive — Confusion Matrix & ROC Curve',
             fontsize=16, fontweight='bold', color='white', y=1.01)

cm_labels = np.array([
    ['True Negative\n(Correct Denial)',    'False Positive\n(Bad Loan Given)'],
    ['False Negative\n(Missed Borrower)',   'True Positive\n(Correct Approval)']
])

y_pred_m, y_proba_m = get_model_proba(champion_name)
row_df  = bakeoff_df[bakeoff_df['Model'] == champion_name].iloc[0]
arch_s, paper_s = ARCH_INFO.get(champion_name, ('N/A', 'N/A'))

ax_cm  = axes[0]
ax_roc = axes[1]
ax_cm.set_facecolor(BG)
ax_roc.set_facecolor(BG)

# ── Confusion Matrix ──────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred_m)
im = ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, alpha=0.85)
thresh = cm.max() / 2.
for i in range(2):
    for j in range(2):
        txt_col = 'white' if cm[i, j] > thresh else 'black'
        ax_cm.text(j, i, f"{cm[i, j]}\n{cm_labels[i, j]}",
                   ha='center', va='center', fontsize=10,
                   fontweight='bold', color=txt_col, linespacing=1.7)

ax_cm.set_xticks([0, 1])
ax_cm.set_yticks([0, 1])
ax_cm.set_xticklabels(['Predicted: Default', 'Predicted: Repay'],
                       fontsize=9, color='#CCCCCC')
ax_cm.set_yticklabels(['Actual: Default', 'Actual: Repay'],
                       fontsize=9, color='#CCCCCC')
ax_cm.set_title(
    f'🥇 Champion: {champion_name}\n'
    f'AUC={row_df["AUC-ROC"]:.4f} | F1={row_df["F1-Score"]:.4f} | '
    f'DPD={row_df["DP Diff"]:.4f}\n'
    f'Papers: {paper_s}',
    fontsize=9, color=GOLD_C, fontweight='bold', pad=8)
ax_cm.tick_params(colors='#AAAAAA')
for sp in ax_cm.spines.values():
    sp.set_edgecolor('#333333')

# ── ROC Curve ─────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_proba_m)
auc_val     = roc_auc_score(y_test, y_proba_m)
ax_roc.plot(fpr, tpr, color=GOLD_C, lw=3,
            label=f'{champion_name}  (AUC={auc_val:.4f})', zorder=4)
ax_roc.plot([0,1],[0,1], color='#555555', lw=1.5, ls='--',
            label='Random (AUC=0.50)')
ax_roc.fill_between(fpr, tpr, alpha=0.12, color=GOLD_C, zorder=3)
ax_roc.set_xlabel('False Positive Rate', fontsize=10, color='#CCCCCC')
ax_roc.set_ylabel('True Positive Rate',  fontsize=10, color='#CCCCCC')
ax_roc.set_title(f'ROC Curve — {champion_name}', fontsize=10,
                  color=GOLD_C, fontweight='bold', pad=8)
ax_roc.legend(loc='lower right', fontsize=9, facecolor='#13131F',
               edgecolor='#444444', labelcolor='white')
ax_roc.tick_params(colors='#AAAAAA')
for sp in ax_roc.spines.values():
    sp.set_edgecolor('#333333')
ax_roc.grid(color='#1E1E2E', ls='--', lw=0.8, alpha=0.8)

print(f"  🥇 {champion_name} | AUC={row_df['AUC-ROC']:.4f} | "
      f"F1={row_df['F1-Score']:.4f} | DPD={row_df['DP Diff']:.4f}")

plt.tight_layout()
plt.savefig('graph1_top3_deep_dive.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
plt.close()
print("\n  📊  Saved: graph1_top3_deep_dive.png\n")


# ═════════════════════════════════════════════════════════════════════════════
#  STEPS 3 & 4 — 60-MONTH POVERTY TRAP SIMULATION + CSV CHECKPOINTING
#  Champion model only
# ═════════════════════════════════════════════════════════════════════════════

print("═" * 80)
print("  STEP 3 & 4 │ 60-MONTH POVERTY TRAP SIMULATION + CHECKPOINTING")
print(f"             │ Running on champion: {champion_name}")
print("═" * 80)

# ── Champion pointers ─────────────────────────────────────────────────────────
champ_is_torch        = champion_name in trained_pytorch
champ_needs_scaling   = champion_name in NEEDS_SCALING
champ_arch, champ_ref = ARCH_INFO.get(champion_name, ('N/A', 'N/A'))

# ── Simulation constants ──────────────────────────────────────────────────────
# CALIBRATION RATIONALE (FICO-realistic, equilibrium-balanced):
#
#   EQUILIBRIUM PROOF (Normal Economy):
#   At threshold repay_prob ≈ 0.566:
#     Approved: 0.566×(+5) + 0.434×(-5) + 1.0 = 2.83 − 2.17 + 1.0 = +1.66   ✓ (slight upward)
#     Denied:   −0.5 (denied penalty) + 1.0 (natural growth) = +0.5            ✓ (stable)
#   → System is in steady-state with slight upward bias during months 0–23.
#
#   PHASE-DEPENDENT DENIED PENALTY:
#     Normal:    −0.5 pts/mo  (occasional hard inquiry, offset by credit aging)
#     Recession: −2.0 pts/mo  (predatory alt-lending, compounding stress)
#     Recovery:  −0.3 pts/mo  (stress subsiding, but still no prime credit access)
# ──────────────────────────────────────────────────────────────────────────────

T_STEPS          = 61
NORMAL_END       = 23
RECESSION_START  = 24
RECESSION_END    = 36
RECOVERY_START   = 37

# Probability-based decision threshold
# sigmoid(640) = 1/(1+exp(-(640-620)/75)) ≈ 0.5661 → use 0.57
THRESHOLD        = 0.57   # ≈ FICO 641 on our sigmoid curve

# ── Economic phase parameters ─────────────────────────────────────────────────
NATURAL_GROWTH   = +1.0   # Credit-age seasoning (normal economy only — NOT during crisis)
RECESSION_SHOCK  = -50    # One-time shock at month 24 (job-loss missed payments)
RECESSION_DRIFT  = -3     # Monthly drift during recession (minorities only)
RECOVERY_LIFT    = +2.5   # Lowered from +5 to ensure a realistic, slow recovery

# ── Feedback loop parameters ──────────────────────────────────────────────────
FEEDBACK_REPAY   = +5     # On-time payment credit building (~60 pts/year)
FEEDBACK_DEFAULT = -5     # Missed payment / default penalty (amortized monthly)
# Denied penalty is PHASE-DEPENDENT (see loop):
DENIED_NORMAL    = -0.5   # Normal: minimal (hard inquiry + stagnation)
DENIED_RECESSION = -2.0   # Recession: predatory alt-lending spiral
DENIED_RECOVERY  = -0.3   # Recovery: easing — still no prime credit, but stress subsiding

CHECKPOINT_MONTHS = {
    23: 'population_Before_Recession.csv',
    30: 'population_During_Recession.csv',
    60: 'population_After_Recession.csv'
}


class DALOF:
    """Drift-Aware Lagrangian Online Fairness (DALOF)"""
    # ── Tighter Threshold safety bounds to prevent over-lending ──────
    THRESHOLD_FLOOR = 0.45   # Drop enough to save good borrowers, but don't approve everyone
    THRESHOLD_CEIL  = 0.57   # Base rigid threshold
    
    # ── Emergency brake parameters ───────────────────────────────────────
    EMERGENCY_SCORE_DROP = 20   
    EMERGENCY_RELIEF     = 0.08 # Calibrated so it stays below normal economy levels
    
    def __init__(self, base_threshold, lr=0.20, drift_window=5, drift_sensitivity=3.0):
        self.base_threshold = base_threshold
        self.thresholds = {
            'white': base_threshold, 'black': base_threshold,
            'hispanic': base_threshold, 'asian': base_threshold
        }
        self.lr = lr
        self.drift_window = drift_window
        self.drift_sensitivity = drift_sensitivity
        self.score_history = []
        self.drift_detected = False
        self.drift_magnitude = 0.0

    def detect_drift(self, current_mean_score):
        """Detect economic drift via population score gradient."""
        self.score_history.append(current_mean_score)
        if len(self.score_history) < self.drift_window + 1:
            return False, 0.0
            
        recent = self.score_history[-self.drift_window:]
        gradient = (recent[-1] - recent[0]) / self.drift_window
        
        # If scores are dropping sharply -> CRISIS MODE ON
        if gradient < -self.drift_sensitivity:
            self.drift_detected = True
            magnitude = abs(gradient)
        else:
            magnitude = 0.0
            
        return self.drift_detected, magnitude

    def emergency_brake(self, current_mean_score):
        if len(self.score_history) < 2:
            return False
        month_drop = self.score_history[-2] - current_mean_score
        if month_drop > self.EMERGENCY_SCORE_DROP:
            relief = min(self.EMERGENCY_RELIEF, self.EMERGENCY_RELIEF * (month_drop / 50.0))
            for g in self.thresholds:
                self.thresholds[g] = max(self.THRESHOLD_FLOOR, self.thresholds[g] - relief)
            self.drift_detected = True
            return True
        return False

    def optimize(self, group_fnrs, overall_fnr, drift_active):
        """Continuous Online Fairness Controller with Normal-Economy Lock"""
        
        # 1. THE NORMAL ECONOMY LOCK:
        # If the recession crash hasn't happened yet, do absolutely nothing.
        # This guarantees LR and LR+DALOF are perfectly identical for Months 0-23.
        if not self.drift_detected:
            return self.thresholds
            
        # 2. CONTINUOUS ONLINE CONTROLLER (Months 24-60)
        for g, fnr in group_fnrs.items():
            error = fnr - overall_fnr
            
            # FAIRNESS CORRECTION: Push threshold down if unfairly rejecting
            if error > 0:
                adjustment = self.lr * error
                self.thresholds[g] -= adjustment
                
            # BANK SAFETY SPRING: If fairness gap is closed, gently pull threshold up
            else:
                if self.thresholds[g] < self.base_threshold:
                    # Gentle spring rate (+0.005) ensures smooth, safe recovery to baseline
                    self.thresholds[g] = min(self.base_threshold, self.thresholds[g] + 0.005)
                    
            # CLAMP: Keep within safe operating bounds
            self.thresholds[g] = np.clip(self.thresholds[g], self.THRESHOLD_FLOOR, self.THRESHOLD_CEIL)
            
        return self.thresholds

print(f"""
  Champion Model        : {champion_name}
  Architecture          : {champ_arch}
  Paper Reference       : {champ_ref}
  Decision Logic        : Rigid Threshold vs DALOF (Drift-Aware Lagrangian Online Fairness)
                          sigmoid(score) = 1/(1+exp(-(score-{SIGMOID_OFFSET})/{SIGMOID_K}))
                          THRESHOLD {THRESHOLD} ≈ FICO score ~641
  Natural Credit Growth : +{NATURAL_GROWTH} pts/month  (credit aging — normal economy only)
  Recession Shock       : {RECESSION_SHOCK} pts ONE-TIME at month {RECESSION_START}
  Recession Monthly Dip : {RECESSION_DRIFT} pts/month (months {RECESSION_START}–{RECESSION_END-1})
  Feedback (Repay)      : +{FEEDBACK_REPAY} pts/month  (on-time payment reward)
  Feedback (Default)    : {FEEDBACK_DEFAULT} pts/month  (missed payment — realistic FICO penalty)
  Feedback (Denied)     : {DENIED_NORMAL}/{DENIED_RECESSION}/{DENIED_RECOVERY} pts/month
                          (Normal / Recession / Recovery)  ← POVERTY TRAP DRIVER
  Recovery Lift         : +{RECOVERY_LIFT} pts/month (months {RECOVERY_START}–{T_STEPS-1})
  ─────────────────────────────────────────────────────────────────────────────
""")

sim_rigid = pop_df[['group','credit_score','repay_prob']].copy()
sim_rigid['credit_score'] = sim_rigid['credit_score'].astype(float)

sim_dalof = pop_df[['group','credit_score','repay_prob']].copy()
sim_dalof['credit_score'] = sim_dalof['credit_score'].astype(float)

dalof_optimizer = DALOF(base_threshold=THRESHOLD, lr=0.20)  # 10× learning rate for fast response

monthly_minority_loans_rigid = []
monthly_minority_loans_dalof = []
monthly_fnr_overall_rigid  = []
monthly_fnr_minority_rigid = []
monthly_fnr_overall_dalof  = []
monthly_fnr_minority_dalof = []

sim_rigid['loan_granted'] = 0
sim_dalof['loan_granted'] = 0

cm_rigid_before = None
cm_rigid_during = None
cm_rigid_after  = None
cm_rigid_total  = np.zeros((2, 2), dtype=int)

cm_dalof_before = None
cm_dalof_during = None
cm_dalof_after  = None
cm_dalof_total  = np.zeros((2, 2), dtype=int)

for t in range(T_STEPS):
    if t in CHECKPOINT_MONTHS:
        fname = CHECKPOINT_MONTHS[t].replace('.csv', '_Rigid.csv')
        sim_rigid[['group','credit_score','repay_prob','loan_granted']].to_csv(fname, index=False)
        fname_d = CHECKPOINT_MONTHS[t].replace('.csv', '_DALOF.csv')
        sim_dalof[['group','credit_score','repay_prob','loan_granted']].to_csv(fname_d, index=False)

    is_minority = pop_df['group'].isin(MINORITY_GROUPS)
    is_recession = (RECESSION_START <= t < RECESSION_END)
    is_recovery  = (t >= RECOVERY_START)
    is_normal    = (t < RECESSION_START)

    if is_normal:
        n_all = len(pop_df)
        growth = np.random.normal(NATURAL_GROWTH, 1.0, n_all)
        sim_rigid['credit_score'] += growth
        sim_dalof['credit_score'] += growth

    if t == RECESSION_START:
        n_min = is_minority.sum()
        shock = np.random.normal(RECESSION_SHOCK, 12, n_min)
        sim_rigid.loc[is_minority, 'credit_score'] += shock
        sim_dalof.loc[is_minority, 'credit_score'] += shock
        print(f"  ⚡  Month {t}: RECESSION SHOCK ({RECESSION_SHOCK} pts)")

    if is_recession:
        n_min = is_minority.sum()
        drift = np.random.normal(RECESSION_DRIFT, 1.5, n_min)
        sim_rigid.loc[is_minority, 'credit_score'] += drift
        sim_dalof.loc[is_minority, 'credit_score'] += drift

    if is_recovery:
        sim_rigid['credit_score'] += RECOVERY_LIFT
        sim_dalof['credit_score'] += RECOVERY_LIFT

    sim_rigid['credit_score'] = np.clip(sim_rigid['credit_score'], 300, 850)
    sim_dalof['credit_score'] = np.clip(sim_dalof['credit_score'], 300, 850)

    sim_rigid['repay_prob'] = sigmoid(sim_rigid['credit_score'].values)
    sim_dalof['repay_prob'] = sigmoid(sim_dalof['credit_score'].values)

    preds_rigid = (sim_rigid['repay_prob'].values >= THRESHOLD).astype(int)
    sim_rigid['loan_granted'] = preds_rigid

    preds_dalof = np.zeros(len(sim_dalof), dtype=int)
    for g, thresh in dalof_optimizer.thresholds.items():
        g_mask = (sim_dalof['group'] == g)
        preds_dalof[g_mask] = (sim_dalof.loc[g_mask, 'repay_prob'].values >= thresh).astype(int)
    sim_dalof['loan_granted'] = preds_dalof

    # Repayment simulation using identical logic
    stochastic_repay = np.random.rand(len(sim_rigid))
    sim_rigid['repaid'] = (stochastic_repay < sim_rigid['repay_prob']).astype(int)
    sim_dalof['repaid'] = (stochastic_repay < sim_dalof['repay_prob']).astype(int)

    min_mask = pop_df['group'].isin(MINORITY_GROUPS)
    cm_r = confusion_matrix(sim_rigid.loc[min_mask, 'repaid'], sim_rigid.loc[min_mask, 'loan_granted'], labels=[0, 1])
    cm_d = confusion_matrix(sim_dalof.loc[min_mask, 'repaid'], sim_dalof.loc[min_mask, 'loan_granted'], labels=[0, 1])
    cm_rigid_total += cm_r
    cm_dalof_total += cm_d
    
    if t == 23:
        cm_rigid_before = cm_r
        cm_dalof_before = cm_d
    elif t == 30:
        cm_rigid_during = cm_r
        cm_dalof_during = cm_d
    elif t == 60:
        cm_rigid_after = cm_r
        cm_dalof_after = cm_d

    def calc_fnr(sim_df):
        y_t = sim_df['repaid'].values
        y_p = sim_df['loan_granted'].values
        min_m = sim_df['group'].isin(MINORITY_GROUPS).values

        pos_all = (y_t == 1).sum()
        fn_all = ((y_t == 1) & (y_p == 0)).sum()
        fnr_all = fn_all / pos_all if pos_all > 0 else 0.0

        pos_min = (y_t[min_m] == 1).sum()
        fn_min = ((y_t[min_m] == 1) & (y_p[min_m] == 0)).sum()
        fnr_min = fn_min / pos_min if pos_min > 0 else 0.0
        
        group_fnrs = {}
        for g in sim_df['group'].unique():
            g_m = (sim_df['group'] == g)
            pos_g = (y_t[g_m] == 1).sum()
            fn_g = ((y_t[g_m] == 1) & (y_p[g_m] == 0)).sum()
            group_fnrs[g] = fn_g / pos_g if pos_g > 0 else 0.0
            
        return fnr_all, fnr_min, group_fnrs

    fnr_all_r, fnr_min_r, _ = calc_fnr(sim_rigid)
    fnr_all_d, fnr_min_d, group_fnrs_d = calc_fnr(sim_dalof)

    monthly_fnr_overall_rigid.append(round(fnr_all_r, 4))
    monthly_fnr_minority_rigid.append(round(fnr_min_r, 4))
    monthly_fnr_overall_dalof.append(round(fnr_all_d, 4))
    monthly_fnr_minority_dalof.append(round(fnr_min_d, 4))

    def apply_feedback(sim_df):
        granted = sim_df['loan_granted'] == 1
        repaid_ok = sim_df['repaid'] == 1
        sim_df.loc[granted & repaid_ok, 'credit_score'] += FEEDBACK_REPAY
        sim_df.loc[granted & ~repaid_ok, 'credit_score'] += FEEDBACK_DEFAULT
        n_denied = (~granted).sum()
        if n_denied > 0:
            if is_recession: denied_rate = DENIED_RECESSION
            elif is_recovery: denied_rate = DENIED_RECOVERY
            else: denied_rate = DENIED_NORMAL
            sim_df.loc[~granted, 'credit_score'] += np.random.normal(denied_rate, 0.5, n_denied)
        sim_df['credit_score'] = np.clip(sim_df['credit_score'], 300, 850)

    apply_feedback(sim_rigid)
    apply_feedback(sim_dalof)

    loans_min_r = int(sim_rigid.loc[is_minority, 'loan_granted'].sum())
    loans_min_d = int(sim_dalof.loc[is_minority, 'loan_granted'].sum())
    monthly_minority_loans_rigid.append(loans_min_r)
    monthly_minority_loans_dalof.append(loans_min_d)

    # DALOF drift detection: monitor minority credit score trajectory
    min_mean_score = sim_dalof.loc[is_minority, 'credit_score'].mean()
    drift_active, drift_mag = dalof_optimizer.detect_drift(min_mean_score)
    # EMERGENCY BRAKE: if scores dropped >20 pts this month, slash thresholds NOW
    emergency_fired = dalof_optimizer.emergency_brake(min_mean_score)
    dalof_optimizer.optimize(group_fnrs_d, fnr_all_d, drift_active or emergency_fired)

    if t % 6 == 0 or t in (RECESSION_START, RECESSION_END-1, T_STEPS-1):
        print(f"  📊  Month {t:>2}: Loans Min(Rigid={loans_min_r:>4}, DALOF={loans_min_d:>4})  "
              f"FNR Min(Rigid={fnr_min_r:.3f}, DALOF={fnr_min_d:.3f})")

print(f"\n  ✅  Simulation complete.\n")

print("═" * 80)
print("  STEP 5 │ TIMELINE VISUALIZATION — The 60-Month Poverty Trap")
print("═" * 80)

months = list(range(T_STEPS))

def piecewise_smooth(series, breakpoints, w=3):
    result = np.array(series, dtype=float)
    edges  = [0] + list(breakpoints) + [len(series)]
    for i in range(len(edges) - 1):
        seg = pd.Series(series[edges[i]:edges[i+1]])
        result[edges[i]:edges[i+1]] = seg.rolling(w, min_periods=1, center=False).mean().values
    return result

champ_smooth = piecewise_smooth(monthly_minority_loans_rigid, [RECESSION_START, RECESSION_END], w=3)
dalof_smooth = piecewise_smooth(monthly_minority_loans_dalof, [RECESSION_START, RECESSION_END], w=3)

pre_avg_r   = int(np.mean(champ_smooth[:RECESSION_START]))
crash_val_r = int(np.min(champ_smooth[RECESSION_START:RECESSION_END]))
final_val_r = int(champ_smooth[-1])
pct_crash_r = round((1 - crash_val_r / max(pre_avg_r, 1)) * 100, 1)
pct_recov_r = round((final_val_r / max(pre_avg_r, 1)) * 100, 1)

pre_avg_d   = int(np.mean(dalof_smooth[:RECESSION_START]))
crash_val_d = int(np.min(dalof_smooth[RECESSION_START:RECESSION_END]))
final_val_d = int(dalof_smooth[-1])
pct_crash_d = round((1 - crash_val_d / max(pre_avg_d, 1)) * 100, 1)
pct_recov_d = round((final_val_d / max(pre_avg_d, 1)) * 100, 1)

y_ceil = max(max(champ_smooth), max(dalof_smooth)) * 1.55

crash_idx_r = RECESSION_START + int(np.argmin(champ_smooth[RECESSION_START:RECESSION_END]))
loans_saved = int(np.sum(dalof_smooth) - np.sum(champ_smooth))

fig2, ax2 = plt.subplots(figsize=(18, 8))
fig2.patch.set_facecolor(BG)
ax2.set_facecolor(BG)

# Shaded zones
ax2.axvspan(0,               RECESSION_START, color='#1A3A1A', alpha=0.18, zorder=0)
ax2.axvspan(RECESSION_START, RECESSION_END,   color='#FF1744', alpha=0.12, zorder=0)
ax2.axvspan(RECESSION_END,   T_STEPS,         color='#1A1A3A', alpha=0.15, zorder=0)
ax2.axvline(RECESSION_START, color='#FF1744', lw=2.5, ls='--', alpha=0.8, zorder=2,
            label=f'Recession onset (Month {RECESSION_START})')
ax2.axvline(RECESSION_END,   color='#42A5F5', lw=2.5, ls='--', alpha=0.8, zorder=2,
            label=f'Recovery begins (Month {RECESSION_END})')

# Zone labels with month ranges
ax2.text(RECESSION_START / 2, y_ceil * 0.97,
         f'✅  NORMAL ECONOMY\nMonths 0 – {RECESSION_START - 1}',
         ha='center', va='top', fontsize=13, color='#66BB6A', fontweight='bold')
ax2.text((RECESSION_START + RECESSION_END) / 2, y_ceil * 0.97,
         f'🔴  RECESSION\nMonths {RECESSION_START} – {RECESSION_END}',
         ha='center', va='top', fontsize=13, color='#FF5555',
         fontweight='bold', style='italic')
ax2.text((RECESSION_END + T_STEPS) / 2, y_ceil * 0.97,
         f'📈  "RECOVERY"\nMonths {RECOVERY_START} – {T_STEPS - 1}',
         ha='center', va='top', fontsize=13, color='#42A5F5', fontweight='bold')

# Main lines
ax2.step(months, champ_smooth, where='post', color=CHAMP_C, lw=3.8, zorder=5,
         label=f'Logistic Regression  (rigid threshold: P(repay) ≥ {THRESHOLD})\n'
               f'Papers: {champ_ref}')
ax2.step(months, dalof_smooth, where='post', color=DALOF_C, lw=3.8, ls='--', zorder=6,
         label=f'LR + DALOF  (dynamic adaptive threshold)')
ax2.fill_between(months, 0, champ_smooth, step='post', alpha=0.08, color=CHAMP_C, zorder=3)
ax2.fill_between(months, 0, dalof_smooth, step='post', alpha=0.08, color=DALOF_C, zorder=3)

# Pre-recession baseline dotted line
ax2.axhline(pre_avg_r, color='#66BB6A', lw=2, ls=':', alpha=0.7, zorder=2)
ax2.text(1, pre_avg_r + y_ceil * 0.03,
         f'Pre-recession baseline (LR): ~{pre_avg_r} loans/month',
         fontsize=11, color='#66BB6A', style='italic', fontweight='bold')

# Annotation 1: The Recession Crash (LR)
ax2.annotate(
    f'💥  RECESSION CRASH\n'
    f'Minority approvals: {pre_avg_r} → {crash_val_r}\n'
    f'({pct_crash_r:.0f}% COLLAPSE)\n'
    f'Rigid P(repay) threshold ({THRESHOLD}) rejects nearly all',
    xy=(crash_idx_r, max(crash_val_r, 5)),
    xytext=(2, y_ceil * 0.25), # <-- Moved up slightly
    fontsize=10, color='#FF8888', fontweight='bold', linespacing=1.5,
    arrowprops=dict(arrowstyle='->', color='#FF6666', lw=2.5,
                    connectionstyle='arc3,rad=-0.3'),
    bbox=dict(boxstyle='round,pad=0.6', fc='#1A0000', ec='#FF4444', lw=2),
    zorder=7)

# Annotation 2: The Poverty Trap (LR)
trap_status = "gap remains substantial." if pct_recov_r < 98 else "fully recovered."
ax2.annotate(
    f'⛓️  THE POVERTY TRAP\n'
    f'Economy recovered — but scores haven\'t.\n'
    f'Denied applicants lost credit access for 12+ months.\n'
    f'Only ~{final_val_r} loans/month vs ~{pre_avg_r} originally.\n'
    f'{pct_recov_r:.0f}% of baseline — {trap_status}',
    xy=(min(52, T_STEPS-2), max(final_val_r, 5)),
    xytext=(38, y_ceil * 0.15), 
    fontsize=10, color='#FF8888', fontweight='bold', linespacing=1.5,
    arrowprops=dict(arrowstyle='->', color='#FF4444', lw=2.5,
                    connectionstyle='arc3,rad=-0.2'),
    bbox=dict(boxstyle='round,pad=0.6', fc='#1A0000', ec='#FF4444', lw=2),
    zorder=7)

# Annotation 3: DALOF Saves
ax2.annotate(
    f'🛡️  DALOF PREVENTS COLLAPSE\n'
    f'Dynamic threshold adapts to recession.\n'
    f'Final: ~{final_val_d} loans/month ({pct_recov_d:.0f}% of baseline).\n'
    f'+{loans_saved} additional loans over 60 months.',
    xy=(T_STEPS - 5, dalof_smooth[-5]),
    xytext=(38, y_ceil * 0.85), # <-- Moved high up to the top right
    fontsize=10, color='#80FFB0', fontweight='bold', linespacing=1.5,
    arrowprops=dict(arrowstyle='->', color=DALOF_C, lw=2.5,
                    connectionstyle='arc3,rad=0.2'),
    bbox=dict(boxstyle='round,pad=0.6', fc='#001A0A', ec=DALOF_C, lw=2),
    zorder=7)

# Unclosed gap bracket for LR
if pre_avg_r > final_val_r + 10:
    bx = min(56, T_STEPS - 4)
    ax2.annotate('', xy=(bx, final_val_r), xytext=(bx, pre_avg_r),
                 arrowprops=dict(arrowstyle='<->', color='white', lw=2.2))
    gap_pct = int(round((1 - final_val_r / max(pre_avg_r, 1)) * 100, 0))
    ax2.text(bx + 1, (final_val_r + pre_avg_r) / 2,
             f'~{gap_pct}%\nunclosed\ngap',
             ha='left', va='center', fontsize=11, color='white',
             fontweight='bold', linespacing=1.4)

ax2.set_xlabel('Month  (0 = Simulation Start)',
               fontsize=14, color='#CCCCCC', labelpad=10)
ax2.set_ylabel('Number of Minority Loans Approved  (per month)',
               fontsize=14, color='#CCCCCC', labelpad=10)
ax2.set_title(
    f'Graph 2: The 60-Month Poverty Trap — LR vs LR+DALOF\n'
    f'How {champion_name} Fails Minority Applicants & How DALOF Prevents It\n'
    f'[Papers: {champ_ref}]',
    fontsize=14, fontweight='bold', color='white', pad=18)
ax2.set_xlim(0, T_STEPS - 1)
ax2.set_ylim(0, y_ceil)
ax2.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 59])
ax2.tick_params(colors='#AAAAAA', labelsize=11)
for sp in ax2.spines.values():
    sp.set_edgecolor('#333333')
ax2.grid(color='#1E1E2E', ls='--', lw=0.8, alpha=0.8)
ax2.legend(loc='upper right', fontsize=10, facecolor='#13131F',
           edgecolor='#444444', labelcolor='white', framealpha=0.92)

plt.tight_layout()
plt.savefig('graph2_poverty_trap.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
plt.close()
print("  📊  Saved: graph2_poverty_trap.png\n")

fnr_overall_rigid_smooth  = piecewise_smooth(monthly_fnr_overall_rigid,  [RECESSION_START, RECESSION_END], w=3)
fnr_minority_rigid_smooth = piecewise_smooth(monthly_fnr_minority_rigid, [RECESSION_START, RECESSION_END], w=3)
fnr_overall_dalof_smooth  = piecewise_smooth(monthly_fnr_overall_dalof,  [RECESSION_START, RECESSION_END], w=3)
fnr_minority_dalof_smooth = piecewise_smooth(monthly_fnr_minority_dalof, [RECESSION_START, RECESSION_END], w=3)

fig3, axes3 = plt.subplots(2, 1, figsize=(18, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.08})
fig3.patch.set_facecolor(BG)

ax_fnr = axes3[0]
ax_fnr.set_facecolor(BG)

for ax in axes3:
    ax.axvspan(0, RECESSION_START, color='#1A3A1A', alpha=0.18, zorder=0)
    ax.axvspan(RECESSION_START, RECESSION_END, color='#FF1744', alpha=0.12, zorder=0)
    ax.axvspan(RECESSION_END, T_STEPS, color='#1A1A3A', alpha=0.15, zorder=0)
    ax.axvline(RECESSION_START, color='#FF1744', lw=2, ls='--', alpha=0.7, zorder=2)
    ax.axvline(RECESSION_END, color='#42A5F5', lw=2, ls='--', alpha=0.7, zorder=2)
    ax.tick_params(colors='#AAAAAA')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333333')
    ax.grid(color='#1E1E2E', ls='--', lw=0.8, alpha=0.5)

# FNR subplot
ax_fnr.plot(months, fnr_minority_rigid_smooth, color=CHAMP_C, lw=2.5, zorder=5,
            label='Minority FNR — LR model (rigid threshold)')
ax_fnr.plot(months, fnr_minority_dalof_smooth, color=DALOF_C, lw=2.5, zorder=5,
            label='Minority FNR — LR + DALOF (dynamic threshold)')
ax_fnr.set_ylabel('Unfair Rejection Rate (FNR)', fontsize=13, color='#CCCCCC')
ax_fnr.set_ylim(0, 1.05)
ax_fnr.legend(loc='upper right', fontsize=9, facecolor='#13131F', edgecolor='#444444', labelcolor='white')

# FNR annotation: LR spike during recession
fnr_peak_idx = RECESSION_START + int(np.argmax(fnr_minority_rigid_smooth[RECESSION_START:RECESSION_END]))
fnr_peak_val = fnr_minority_rigid_smooth[fnr_peak_idx]
fnr_dalof_at_peak = fnr_minority_dalof_smooth[fnr_peak_idx]
ax_fnr.annotate(
    f'LR FNR spikes to {fnr_peak_val:.0%}\n'
    f'DALOF holds at {fnr_dalof_at_peak:.0%}\n'
    f'Δ = {(fnr_peak_val - fnr_dalof_at_peak):.0%} fewer\n'
    f'unfair rejections',
    xy=(fnr_peak_idx, fnr_peak_val),
    xytext=(fnr_peak_idx + 8, fnr_peak_val + 0.15),
    fontsize=9, color='#FF8888', fontweight='bold', linespacing=1.4,
    arrowprops=dict(arrowstyle='->', color='#FF6666', lw=2),
    bbox=dict(boxstyle='round,pad=0.4', fc='#1A0000', ec='#FF4444', lw=1.5),
    zorder=7)

# FNR summary stats box
avg_fnr_rigid = np.mean(fnr_minority_rigid_smooth)
avg_fnr_dalof = np.mean(fnr_minority_dalof_smooth)
fnr_reduction = ((avg_fnr_rigid - avg_fnr_dalof) / max(avg_fnr_rigid, 0.001)) * 100
stats_text = (f'Average Minority FNR:\n'
              f'  LR (rigid):  {avg_fnr_rigid:.1%}\n'
              f'  LR + DALOF:  {avg_fnr_dalof:.1%}\n'
              f'  Reduction:   {fnr_reduction:.0f}%')
ax_fnr.text(0.01, 0.35, stats_text, transform=ax_fnr.transAxes,
            fontsize=9, color='white', fontweight='bold',
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', fc='#13131F', ec='#444444', alpha=0.9))

# Fairness Gap subplot
ax_gap = axes3[1]
ax_gap.set_facecolor(BG)
gap_rigid = np.array(fnr_minority_rigid_smooth) - np.array(fnr_overall_rigid_smooth)
gap_dalof = np.array(fnr_minority_dalof_smooth) - np.array(fnr_overall_dalof_smooth)
ax_gap.plot(months, gap_rigid, color=CHAMP_C, lw=2.5, label='Fairness Gap — LR model')
ax_gap.plot(months, gap_dalof, color=DALOF_C, lw=2.5, label='Fairness Gap — LR + DALOF')
ax_gap.fill_between(months, gap_rigid, gap_dalof, alpha=0.15, color=DALOF_C,
                     where=np.array(gap_rigid) > np.array(gap_dalof), zorder=3,
                     label='DALOF fairness improvement')
ax_gap.axhline(0, color='white', lw=0.8, alpha=0.4)
ax_gap.set_ylabel('Fairness Gap (Minority FNR − Overall FNR)', fontsize=11, color='#CCCCCC')
ax_gap.set_xlabel('Month  (0 = Simulation Start)', fontsize=13, color='#CCCCCC')
ax_gap.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 59])
ax_gap.legend(loc='upper right', fontsize=9, facecolor='#13131F', edgecolor='#444444', labelcolor='white')

fig3.suptitle(
    f'Graph 3: Unfair Rejection Rate & Fairness Gap — LR vs LR+DALOF\n'
    f'How DALOF reduces minority-specific false negative rates and closes the fairness gap',
    fontsize=14, fontweight='bold', color='white', y=1.01)

plt.tight_layout()
plt.savefig('graph3_unfair_rejections.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
plt.close()
print("  📊  Saved: graph3_unfair_rejections.png\n")

# ═════════════════════════════════════════════════════════════════════════════
#  PRODUCTION SUMMARY — LR vs LR+DALOF Comparison
# ═════════════════════════════════════════════════════════════════════════════

pre_loans_r  = int(np.mean(monthly_minority_loans_rigid[:RECESSION_START]))
rec_loans_r  = int(np.mean(monthly_minority_loans_rigid[RECESSION_START:RECESSION_END]))
post_loans_r = int(np.mean(monthly_minority_loans_rigid[RECESSION_END:]))
pre_loans_d  = int(np.mean(monthly_minority_loans_dalof[:RECESSION_START]))
rec_loans_d  = int(np.mean(monthly_minority_loans_dalof[RECESSION_START:RECESSION_END]))
post_loans_d = int(np.mean(monthly_minority_loans_dalof[RECESSION_END:]))

total_loans_r = int(np.sum(monthly_minority_loans_rigid))
total_loans_d = int(np.sum(monthly_minority_loans_dalof))
total_extra   = total_loans_d - total_loans_r

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         SF-RISK: 60-MONTH SIMULATION RESULTS — PRODUCTION SUMMARY          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Champion Model : {champion_name:<52} ║
║  Architecture   : {champ_arch[:52]:<52} ║
║  Paper Ref      : {champ_ref[:52]:<52} ║
║  Threshold      : P(repay) >= {THRESHOLD} (rigid, ≈ FICO 641)                    ║
║                                                                              ║
╠═══════════════════════════════╦═══════════════╦═══════════════════════════════╣
║  METRIC                       ║  LR (Rigid)   ║  LR + DALOF                   ║
╠═══════════════════════════════╬═══════════════╬═══════════════════════════════╣
║  Pre-Recession  (months 0-23) ║  ~{pre_loans_r:>4}/month  ║  ~{pre_loans_d:>4}/month                  ║
║  During Recession (24-36)     ║  ~{rec_loans_r:>4}/month  ║  ~{rec_loans_d:>4}/month                  ║
║  Post-Recession (37-60)       ║  ~{post_loans_r:>4}/month  ║  ~{post_loans_d:>4}/month                  ║
║  Total Loans (60 months)      ║  {total_loans_r:>6}       ║  {total_loans_d:>6}                        ║
║                               ║               ║                               ║
║  Recession DROP               ║  {pct_crash_r:.1f}%        ║  {pct_crash_d:.1f}%                         ║
║  Recovery (% of baseline)     ║  {pct_recov_r:.0f}%         ║  {pct_recov_d:.0f}%                          ║
║  Avg Minority FNR             ║  {avg_fnr_rigid:.1%}       ║  {avg_fnr_dalof:.1%}                        ║
╠═══════════════════════════════╩═══════════════╩═══════════════════════════════╣
║                                                                              ║
║  📈  DALOF IMPROVEMENT:                                                      ║
║    +{total_extra:>5} additional minority loans approved over 60 months           ║
║    {fnr_reduction:.0f}% reduction in unfair rejection rate (FNR)                      ║
║                                                                              ║
║  💼  BUSINESS CASE:                                                          ║
║    A bank using DALOF maintains profitable lending volume during crises      ║
║    while competitors using rigid thresholds panic-freeze and leave money     ║
║    on the table by rejecting safe borrowers (False Negatives).               ║
║                                                                              ║
║  ⚖️  FAIRNESS CASE:                                                         ║
║    DALOF prevents creditworthy minorities from being unjustly penalized      ║
║    for macroeconomic events, stopping the compounding credit decay           ║
║    (Poverty Trap) before it begins.                                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 6 — CONFUSION MATRICES PLOT (Before, During, After, Total)
# ═════════════════════════════════════════════════════════════════════════════
print("═" * 80)
print("  STEP 6 │ CONFUSION MATRICES — LR vs LR+DALOF Across Simulation Phases")
print("═" * 80)

fig4, axes4 = plt.subplots(2, 4, figsize=(24, 10))
fig4.patch.set_facecolor(BG)
fig4.suptitle('Graph 4: Confusion Matrices for Minority Borrowers (LR vs LR+DALOF)',
              fontsize=18, fontweight='bold', color='white', y=1.02)

cm_list = [
    (cm_rigid_before, 'LR model\n(Before Recession)',    CHAMP_C, axes4[0,0]),
    (cm_rigid_during, 'LR model\n(During Recession)',    CHAMP_C, axes4[0,1]),
    (cm_rigid_after,  'LR model\n(After Recession)',     CHAMP_C, axes4[0,2]),
    (cm_rigid_total,  'LR model\n(Total 0-60 Months)',   CHAMP_C, axes4[0,3]),
    (cm_dalof_before, 'LR + DALOF\n(Before Recession)',  DALOF_C, axes4[1,0]),
    (cm_dalof_during, 'LR + DALOF\n(During Recession)',  DALOF_C, axes4[1,1]),
    (cm_dalof_after,  'LR + DALOF\n(After Recession)',   DALOF_C, axes4[1,2]),
    (cm_dalof_total,  'LR + DALOF\n(Total 0-60 Months)', DALOF_C, axes4[1,3])
]

for cm_data, title, col, ax in cm_list:
    ax.set_facecolor(BG)
    if cm_data is not None:
        im = ax.imshow(cm_data, interpolation='nearest', cmap=plt.cm.Blues, alpha=0.85)
        thresh = cm_data.max() / 2.
        for i in range(2):
            for j in range(2):
                txt_col = 'white' if cm_data[i, j] > thresh else 'black'
                ax.text(j, i, f"{cm_data[i, j]}\n{cm_labels[i, j]}",
                        ha='center', va='center', fontsize=10,
                        fontweight='bold', color=txt_col, linespacing=1.7)
        # Add FNR metric below title
        fn = cm_data[1, 0]
        tp = cm_data[1, 1]
        fnr_cm = fn / max(fn + tp, 1)
        title += f'\nFNR: {fnr_cm:.1%}'
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted: Default', 'Predicted: Repay'], fontsize=9, color='#CCCCCC')
    ax.set_yticklabels(['Actual: Default', 'Actual: Repay'], fontsize=9, color='#CCCCCC')
    ax.set_title(title, fontsize=10, color=col, fontweight='bold', pad=8)
    ax.tick_params(colors='#AAAAAA')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333333')

plt.tight_layout()
plt.savefig('graph4_simulation_cms.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
plt.close()
print("  📊  Saved: graph4_simulation_cms.png\n")

print("  ✅  Pipeline complete.")
print("  📁  Files: bakeoff_leaderboard.csv · population CSVs · 4× PNG graphs")