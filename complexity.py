import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import norm, mannwhitneyu
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION — change these each run
# ══════════════════════════════════════════════════════════════════════════

FILE_NAME  = r"C:\Users\kmala\AppData\Local\Programs\Python\Python313\engr_awards.xlsx"
SHEET_NAME = "Sheet1"
OUTPUT_DIR = r"C:\Users\kmala\AppData\Local\Programs\Python\Python313"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── CL definition ─────────────────────────────────────────────────────────
CL_THRESHOLD = 10_000_000          # PI on grant >= this amount = CL

# ── Purdue fiscal year ────────────────────────────────────────────────────
FY_MONTH_CUTOFF = 8                # month >= 8 → belongs to next FY

# ── Date basis ────────────────────────────────────────────────────────────
# "award"  → Award Date / AwardFY
# "submit" → Proposal Submit Date / SubmitFY
DATE_BASIS = "award"

# ── Minimum total years for non-CL faculty ────────────────────────────────
# CL faculty are always kept regardless
MIN_TOTAL_YEARS_REQUIRED = 1

# ── Training window ───────────────────────────────────────────────────────
# CL faculty    → most recent WINDOW_YEARS pre-center years
# Non-CL faculty→ most recent WINDOW_YEARS overall
WINDOW_YEARS        = 10
REQUIRE_FULL_WINDOW = False        # set True to exclude faculty with < WINDOW_YEARS

# ── Features ──────────────────────────────────────────────────────────────
# Comment out to exclude from model
FEATURE_COLS = [
    "n_piGrants",
    "n_Federalgrants",
    # "n_sponsors",
    # "total_Collaborators",
    "log_Total_Staff_Dollars",
    # "log_Total_Dollars",
    # "Total_Staff_Dollars",
    # "Total_Dollars",
    # "log_max_Award_Per_Year",
    # "n_awards",
]

# ── Aggregation per feature ───────────────────────────────────────────────
# mean / sum / max / min / pergrant / last
# NOTE: must be consistent — the scoring step uses this to z-score correctly
FEATURE_AGG = {
    "n_piGrants":              "mean",
    "n_Federalgrants":         "mean",
    "n_sponsors":              "mean",
    "total_Collaborators":     "mean",
    "log_Total_Staff_Dollars": "mean",
    "log_Total_Dollars":       "mean",
    "Total_Staff_Dollars":     "sum",
    "Total_Dollars":           "sum",
    "log_max_Award_Per_Year":  "max",
    "n_awards":                "mean",
}

# ── Outlier removal ───────────────────────────────────────────────────────
REMOVE_OUTLIERS      = True
IQR_MULTIPLIER       = 3.0         # raise to keep more faculty
MIN_CL_AFTER_OUTLIER = 10          # never drop CL if it leaves fewer than this

# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def fiscal_year_from_date(dt_series, cutoff_month=8):
    return np.where(
        dt_series.dt.month >= cutoff_month,
        dt_series.dt.year + 1,
        dt_series.dt.year
    ).astype(float)

def cl_period(idx):
    if pd.isna(idx):   return 'non-CL'
    if idx < 0:        return 'pre-center'
    if idx == 0:       return 'center'
    return 'post-center'

def get_prefix(method):
    return {'mean':'avg','sum':'sum','max':'max','min':'min',
            'pergrant':'pgrant','last':'last'}.get(method, 'avg')

def aggregate(window, year_col, feature, method):
    if feature not in window.columns: return 0.0
    s = window[feature].dropna()
    if len(s) == 0: return 0.0
    if method == 'mean':     return float(s.mean())
    if method == 'sum':      return float(s.sum())
    if method == 'max':      return float(s.max())
    if method == 'min':      return float(s.min())
    if method == 'last':
        ly = window[year_col].max()
        lv = window[window[year_col]==ly][feature].dropna()
        return float(lv.mean()) if len(lv) > 0 else 0.0
    if method == 'pergrant':
        denom = float(window['n_awards'].sum()) if 'n_awards' in window.columns else 1.0
        return float(s.sum()) / denom if denom > 0 else 0.0
    return float(s.mean())

def sig(p):
    if pd.isna(p): return '—'
    if p < .001:   return '***'
    if p < .01:    return '**'
    if p < .05:    return '*'
    if p < .10:    return '.'
    return 'ns'

# Feature column names with prefix
feat_cols = [f"{get_prefix(FEATURE_AGG.get(f,'mean'))}_{f}" for f in FEATURE_COLS]

# Date basis
if DATE_BASIS == "award":
    BASE_DATE_COL = "Award Date"
    BASE_FY_COL   = "AwardFY"
elif DATE_BASIS == "submit":
    BASE_DATE_COL = "Proposal Submit Date"
    BASE_FY_COL   = "SubmitFY"
else:
    raise ValueError("DATE_BASIS must be 'award' or 'submit'")

# All feature variables — used for scoring even if not in model
ALL_VARS = [
    "n_piGrants", "n_Federalgrants", "n_sponsors", "total_Collaborators",
    "log_Total_Staff_Dollars", "log_Total_Dollars",
    "Total_Staff_Dollars", "Total_Dollars",
    "log_max_Award_Per_Year", "n_awards",
]

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD AND CLEAN RAW DATA
# ══════════════════════════════════════════════════════════════════════════

print("=" * 62)
print("STEP 1 — Loading raw data")
print("=" * 62)

raw = pd.read_excel(FILE_NAME, sheet_name=SHEET_NAME)
raw.columns = raw.columns.str.strip()
raw['Staff Name'] = raw['Staff Name'].astype(str).str.strip()

raw['Award Date']            = pd.to_datetime(raw['Award Date'],            errors='coerce')
raw['Proposal Submit Date']  = pd.to_datetime(raw['Proposal Submit Date'],  errors='coerce')

raw['AwardFY']  = fiscal_year_from_date(raw['Award Date'],           FY_MONTH_CUTOFF)
raw['SubmitFY'] = fiscal_year_from_date(raw['Proposal Submit Date'], FY_MONTH_CUTOFF)

raw['BaseDate'] = raw[BASE_DATE_COL]
raw['BaseFY']   = raw[BASE_FY_COL]

# IsFederal
gt = raw['Grant Type'].fillna('').astype(str).str.upper().str.strip()
raw['IsFederal'] = (
    gt.str.startswith('FED') | gt.str.contains('FEDERAL', na=False)
).astype(int)

# StaffFundingClean
raw['StaffFundingClean'] = np.where(
    (raw['Staff Prop Amount'] == 0) &
    (raw['Cost Share Type'] == 'No Cost Share') &
    (raw['Total Proposal Amount1'] > 0),
    raw['Total Proposal Amount1'],
    raw['Staff Prop Amount']
)

# IsCL per row
raw['row_IsCL'] = (
    (raw['Staff Role'] == 'PI') &
    (raw['Total Proposal Amount1'] >= CL_THRESHOLD)
).astype(int)

# Collaborators per row
print("  Computing collaborators per proposal...")
proposal_nunique   = raw.groupby('Proposal Number')['Staff Name'].transform('nunique')
raw['Collaborators'] = (proposal_nunique - 1).clip(lower=0).fillna(0)

print(f"  DATE_BASIS: {DATE_BASIS}")
print(f"  Raw rows:   {len(raw):,}")
print(f"  Faculty:    {raw['Staff Name'].nunique():,}")
print(f"  FY range:   {int(raw['BaseFY'].min())} – {int(raw['BaseFY'].max())}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — IDENTIFY CL FACULTY AND CENTER DATES
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 62)
print("STEP 2 — Identifying CL faculty")
print("=" * 62)

cl_grants = raw[raw['row_IsCL'] == 1].copy()
cl_center = (
    cl_grants.groupby('Staff Name')['BaseDate']
    .min().reset_index()
    .rename(columns={'BaseDate': 'CenterDate'})
)
cl_center['CenterFY'] = fiscal_year_from_date(cl_center['CenterDate'], FY_MONTH_CUTOFF)
cl_names = set(cl_center['Staff Name'])

print(f"  CL faculty: {len(cl_names)}")
for _, r in cl_center.sort_values('CenterDate').iterrows():
    print(f"    {r['Staff Name']:<44} center={r['CenterDate'].date()}  "
          f"CenterFY={int(r['CenterFY'])}")

raw = raw.merge(cl_center[['Staff Name','CenterDate','CenterFY']],
                on='Staff Name', how='left')
raw['IsCL']      = raw['Staff Name'].isin(cl_names).astype(int)
raw['CL_Index']  = np.where(raw['CenterFY'].notna(),
                             raw['BaseFY'] - raw['CenterFY'], np.nan)
raw['CL_Period'] = raw['CL_Index'].apply(cl_period)

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — BUILD FACULTY-YEAR AGGREGATES (ALL YEARS, ALL FACULTY)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 62)
print("STEP 3 — Building Faculty_Year_Aggregates (all years)")
print("=" * 62)

def build_year_aggs(df):
    rows = []
    for (name, fy), g in df.groupby(['Staff Name', 'BaseFY']):
        proposals  = g['Proposal Number'].dropna().unique()
        pi_props   = g[g['Staff Role']=='PI']['Proposal Number'].dropna().unique()
        fed_props  = g[g['IsFederal']==1]['Proposal Number'].dropna().unique()
        collab     = float(g.groupby('Proposal Number')['Collaborators'].max().sum()) \
                     if len(proposals) > 0 else 0.0
        staff_d    = float(g['StaffFundingClean'].sum())
        total_d    = float(g['Total Proposal Amount1'].sum())
        max_award  = float(g['Total Proposal Amount1'].max()) if len(g) > 0 else 0.0
        cl_idx     = g['CL_Index'].iloc[0] if g['CL_Index'].notna().any() else np.nan
        rows.append({
            'Staff Name':              name,
            'FY':                      int(fy),
            'IsCL':                    int(g['IsCL'].max()),
            'CenterDate':              g['CenterDate'].iloc[0] if g['CenterDate'].notna().any() else pd.NaT,
            'CenterFY':                float(g['CenterFY'].iloc[0]) if g['CenterFY'].notna().any() else np.nan,
            'CL_Index':                float(cl_idx) if pd.notna(cl_idx) else np.nan,
            'CL_Period':               cl_period(cl_idx),
            'n_piGrants':              int(len(pi_props)),
            'n_Federalgrants':         int(len(fed_props)),
            'n_sponsors':              int(g['COEUS Sponsor Name'].nunique()),
            'total_Collaborators':     collab,
            'Total_Staff_Dollars':     staff_d,
            'log_Total_Staff_Dollars': float(np.log1p(staff_d)),
            'Total_Dollars':           total_d,
            'log_Total_Dollars':       float(np.log1p(total_d)),
            'max_Award_Per_Year':      max_award,
            'log_max_Award_Per_Year':  float(np.log1p(max_award)),
            'n_awards':                int(len(proposals)),
        })
    return pd.DataFrame(rows)

yearly = build_year_aggs(raw)
print(f"  Faculty-year rows: {len(yearly):,}")
print(f"  Unique faculty:    {yearly['Staff Name'].nunique():,}")
print(f"  FY range:          {int(yearly['FY'].min())} to {int(yearly['FY'].max())}")

# ══════════════════════════════════════════════════════════════════════════
# FILTER 1 — MINIMUM TOTAL YEARS UNLESS CL
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 62)
print("FILTER 1 — Minimum total years unless CL")
print("=" * 62)

fac_total = yearly.groupby('Staff Name')['FY'].nunique().reset_index(name='total_yrs')
fac_iscl  = yearly.groupby('Staff Name')['IsCL'].max().reset_index()
fac_check = fac_total.merge(fac_iscl, on='Staff Name', how='left')

eligible_names = set(fac_check[
    (fac_check['IsCL']==1) |
    (fac_check['total_yrs'] >= MIN_TOTAL_YEARS_REQUIRED)
]['Staff Name'])

excluded_total = fac_check[
    (fac_check['IsCL']==0) &
    (fac_check['total_yrs'] < MIN_TOTAL_YEARS_REQUIRED)
].copy()

print(f"  Kept:    {len(eligible_names)} faculty")
print(f"  Dropped: {len(excluded_total)} non-CL faculty with < {MIN_TOTAL_YEARS_REQUIRED} years")

yearly = yearly[yearly['Staff Name'].isin(eligible_names)].copy().reset_index(drop=True)

yearly.to_csv(os.path.join(OUTPUT_DIR, "Faculty_Year_Aggregates.csv"), index=False)
print("  Saved: Faculty_Year_Aggregates.csv")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — BUILD MODEL TRAINING ROWS (STRICT WINDOW)
#
# CL faculty    → most recent WINDOW_YEARS pre-center years
# Non-CL faculty→ most recent WINDOW_YEARS overall
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 62)
print("STEP 4 — Building model training dataset")
print("=" * 62)

YEAR_COL = 'FY'
T_COL    = 'CL_Index'

print("  Aggregation methods:")
for f in FEATURE_COLS:
    m = FEATURE_AGG.get(f, 'mean')
    print(f"    {f:<36} → {m}  ({get_prefix(m)}_{f})")

rows_model    = []
excluded_win  = []
all_cl        = set(yearly[yearly['IsCL']==1]['Staff Name'].unique())

for staff, g in yearly.groupby('Staff Name'):
    is_cl    = int(g['IsCL'].max())
    g_sorted = g.sort_values(YEAR_COL, ascending=False).copy()
    avail    = int(g_sorted[YEAR_COL].nunique())

    if is_cl == 1:
        pre    = g_sorted[g_sorted[T_COL] < 0].copy()
        window = pre.head(WINDOW_YEARS).copy()
        if window.empty:
            excluded_win.append((staff, "CL — no pre-center rows"))
            continue
        n_yrs = int(window[YEAR_COL].nunique())
        if REQUIRE_FULL_WINDOW and n_yrs < WINDOW_YEARS:
            excluded_win.append((staff, f"CL — only {n_yrs} pre-center yr(s), need {WINDOW_YEARS}"))
            continue
    else:
        window = g_sorted.head(WINDOW_YEARS).copy()
        if window.empty:
            excluded_win.append((staff, "non-CL — no rows"))
            continue
        n_yrs = int(window[YEAR_COL].nunique())
        if REQUIRE_FULL_WINDOW and n_yrs < WINDOW_YEARS:
            excluded_win.append((staff, f"non-CL — only {n_yrs} yr(s), need {WINDOW_YEARS}"))
            continue

    row = {
        'Staff Name':      staff,
        'IsCL':            is_cl,
        'n_yrs_used':      int(window[YEAR_COL].nunique()),
        'n_yrs_available': avail,
        'window_start_fy': int(window[YEAR_COL].min()),
        'window_end_fy':   int(window[YEAR_COL].max()),
        'CL_Index_min':    int(window[T_COL].min()) if is_cl and window[T_COL].notna().any() else None,
        'CL_Index_max':    int(window[T_COL].max()) if is_cl and window[T_COL].notna().any() else None,
        'n_grants_window': float(window['n_awards'].sum()) if 'n_awards' in window.columns else None,
    }
    for f in FEATURE_COLS:
        method = FEATURE_AGG.get(f, 'mean')
        row[f"{get_prefix(method)}_{f}"] = aggregate(window, YEAR_COL, f, method)
    rows_model.append(row)

model_df = pd.DataFrame(rows_model).reset_index(drop=True)
model_df = model_df.dropna(subset=feat_cols+['IsCL']).copy()

n_cl    = int(model_df['IsCL'].sum())
n_noncl = int((model_df['IsCL']==0).sum())
excl_cl = [(s,r) for s,r in excluded_win if s in all_cl]

print(f"\n  Faculty in model:   {len(model_df)}  (CL={n_cl}  Non-CL={n_noncl})")
print(f"  Strict window size: {WINDOW_YEARS} year(s)")
print(f"  Excluded:           {len(excluded_win)}")
for s, r in excluded_win:
    print(f"    {s:<44} {r}")

print(f"\n  CL faculty in model:")
for _, r in model_df[model_df['IsCL']==1]\
        .sort_values(['window_end_fy','n_yrs_used'], ascending=[False,False]).iterrows():
    idx = (f"CL_Index=[{int(r['CL_Index_min'])},{int(r['CL_Index_max'])}]"
           if r['CL_Index_min'] is not None else "")
    print(f"    {r['Staff Name']:<44} {r['n_yrs_used']} yrs  "
          f"FY[{r['window_start_fy']},{r['window_end_fy']}]  {idx}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — UNIVARIATE TESTS
# ══════════════════════════════════════════════════════════════════════════

cl_s = model_df[model_df['IsCL']==1]
nc_s = model_df[model_df['IsCL']==0]

print(f"\n  Univariate tests:")
print(f"  {'Feature':<42} {'CL':>8}  {'Non-CL':>8}  {'ratio':>6}  {'p(MW)':>8}  sig")
print(f"  {'-'*78}")
for fc, f in zip(feat_cols, FEATURE_COLS):
    cv = cl_s[fc].dropna().values
    nv = nc_s[fc].dropna().values
    cm, nm = cv.mean(), nv.mean()
    ratio  = cm/nm if nm != 0 else float('inf')
    try:
        _, p = mannwhitneyu(cv, nv, alternative='two-sided')
        s = '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else '.' if p<.1 else 'ns'
    except:
        p, s = float('nan'), '—'
    print(f"  {f:<42} {cm:>8.3f}  {nm:>8.3f}  {ratio:>6.2f}x  {p:>8.4f}  {s}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — OUTLIER CHECK
# ══════════════════════════════════════════════════════════════════════════

print(f"\n  Outlier check (IQR × {IQR_MULTIPLIER}):")
outlier_flags = pd.Series(False, index=model_df.index)

for group_label, mask in [("CL", model_df['IsCL']==1), ("Non-CL", model_df['IsCL']==0)]:
    group       = model_df[mask]
    group_flags = pd.Series(False, index=group.index)
    for fc in feat_cols:
        Q1, Q3 = group[fc].quantile(0.25), group[fc].quantile(0.75)
        IQR    = Q3 - Q1
        lo, hi = Q1 - IQR_MULTIPLIER*IQR, Q3 + IQR_MULTIPLIER*IQR
        flagged = (group[fc] < lo) | (group[fc] > hi)
        if flagged.any():
            for name in group[flagged]['Staff Name'].tolist():
                val = group[group['Staff Name']==name][fc].values[0]
                print(f"    [{group_label}] {name:<42} {fc}: {val:.2f}"
                      f"  (outside [{lo:.2f},{hi:.2f}])")
        group_flags |= flagged
    outlier_flags |= group_flags
    print(f"    {group_label}: {group_flags.sum()} flagged / {len(group)}")

if REMOVE_OUTLIERS and outlier_flags.sum() > 0:
    cl_flagged = outlier_flags & (model_df['IsCL']==1)
    if (model_df['IsCL'].sum() - cl_flagged.sum()) < MIN_CL_AFTER_OUTLIER:
        print(f"  WARNING: would leave < {MIN_CL_AFTER_OUTLIER} CL — "
              f"keeping all CL, removing non-CL outliers only")
        outlier_flags &= (model_df['IsCL']==0)
    removed = model_df[outlier_flags]['Staff Name'].tolist()
    model_df = model_df[~outlier_flags].copy().reset_index(drop=True)
    print(f"  Removed {len(removed)}: {removed}")
    n_cl    = int(model_df['IsCL'].sum())
    n_noncl = int((model_df['IsCL']==0).sum())

# ══════════════════════════════════════════════════════════════════════════
# STEP 7 — VIF
# ══════════════════════════════════════════════════════════════════════════

print(f"\n  Variance Inflation Factors:")
for i, fc in enumerate(feat_cols):
    try:
        vif  = variance_inflation_factor(model_df[feat_cols].values, i)
        flag = "  *** HIGH" if vif>5 else ("  . moderate" if vif>2.5 else "")
        print(f"    {fc:<44} VIF = {vif:.2f}{flag}")
    except Exception as e:
        print(f"    {fc:<44} VIF = ERROR ({e})")

# ══════════════════════════════════════════════════════════════════════════
# STEP 8 — FIT MODEL
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 62)
print("STEP 8 — Fitting logistic regression")
print("=" * 62)

X_raw = model_df[feat_cols].values
y     = model_df['IsCL'].astype(int).values

sc  = StandardScaler()
Xs  = sc.fit_transform(X_raw)
clf = LogisticRegression(C=1.0, max_iter=2000,
                          class_weight='balanced', random_state=42)
clf.fit(Xs, y)

yp     = clf.predict_proba(Xs)[:,1]
auc    = roc_auc_score(y, yp)
coefs  = clf.coef_[0]
intcpt = clf.intercept_[0]

# P-values
n, W_d = len(y), yp*(1-yp)
Xi     = np.column_stack([np.ones(n), Xs])
try:
    H     = Xi.T @ (W_d[:,None]*Xi)
    se    = np.sqrt(np.diag(np.linalg.inv(H)))
    all_c = np.r_[intcpt, coefs]
    z_st  = all_c / se
    p_vs  = 2*(1-norm.cdf(np.abs(z_st)))
except np.linalg.LinAlgError:
    se = z_st = p_vs = np.full(len(coefs)+1, np.nan)
    all_c = np.r_[intcpt, coefs]
    print("WARNING: Hessian singular — p-values unavailable")

# McFadden R²
eps      = 1e-10
ll_m     = float(np.sum(y*np.log(np.clip(yp,eps,1))+(1-y)*np.log(np.clip(1-yp,eps,1))))
ll_n_bal = float(np.sum(y*np.log(0.5)+(1-y)*np.log(0.5)))
ll_n_std = float(np.sum(y*np.log(y.mean()+eps)+(1-y)*np.log(1-y.mean()+eps)))
mcf_bal  = round(1-ll_m/ll_n_bal, 4)
mcf_std  = round(1-ll_m/ll_n_std, 4)

z_names = [f"z_{fc}" for fc in feat_cols]

print(f"  N={n}  CL={y.sum()}  Non-CL={(y==0).sum()}")
print(f"  AUC:                    {auc:.4f}")
print(f"  McFadden R2 (balanced): {mcf_bal}  ← use this")
print(f"  McFadden R2 (standard): {mcf_std}")
print()
print(f"  {'Variable':<44} {'Coef':>8}  {'OR':>8}  {'SE':>7}  {'z':>7}  {'p':>9}  sig")
print(f"  {'-'*92}")
print(f"  {'Intercept':<44} {intcpt:>8.4f}  {'—':>8}  "
      f"{se[0]:>7.4f}  {z_st[0]:>7.3f}  {p_vs[0]:>9.4f}  {sig(p_vs[0])}")
for fc, zn, c, s, zv, pv in zip(feat_cols, z_names, coefs, se[1:], z_st[1:], p_vs[1:]):
    print(f"  {fc:<44} {c:>8.4f}  {np.exp(c):>8.4f}  "
          f"{s:>7.4f}  {zv:>7.3f}  {pv:>9.4f}  {sig(pv)}")

print(f"\n  Population stats (training population):")
print(f"  {'Feature':<44} {'mean':>10}  {'sd':>10}")
print(f"  {'-'*66}")
for fc, m, s in zip(feat_cols, sc.mean_, sc.scale_):
    print(f"  {fc:<44} {m:>10.4f}  {s:>10.4f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 9 — SCORE EVERY FACULTY-YEAR ROW
#
# For each row in yearly:
#   - z-score model features using TRAINING population mean/SD
#   - z-score non-model variables using full-population mean/SD
#   - compute CompositeZ and Probability using model coefficients
#   - compute complexity component scores (0-1 vs p95)
#   - assign Tier
#
# KEY: z-score keys match the actual feature column names (with prefix)
#      so avg_ and sum_ features are scored correctly
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 62)
print("STEP 9 — Scoring all faculty-year rows")
print("=" * 62)

# Build population stats lookup keyed by feature column name
# Model features → use exact training mean/SD
pop_mean = {}
pop_sd   = {}
for fc, m, s in zip(feat_cols, sc.mean_, sc.scale_):
    pop_mean[fc] = float(m)
    pop_sd[fc]   = float(s)

# Non-model variables → compute from full population using mean aggregation
# stored under avg_{varname} for display z-scores only
for v in ALL_VARS:
    fc = f"avg_{v}"
    if fc not in pop_mean and v in yearly.columns:
        vals = yearly.groupby('Staff Name')[v].mean().dropna()
        pop_mean[fc] = float(vals.mean()) if len(vals) > 0 else 0.0
        pop_sd[fc]   = float(vals.std())  if len(vals) > 1 and vals.std() > 0 else 1.0

# p95 per raw variable for complexity component normalization
p95 = {}
for v in ALL_VARS:
    if v in yearly.columns:
        q = float(yearly[v].quantile(0.95))
        p95[v] = q if q > 0 else 1.0

# Build a mapping: base variable name → feature column name used in model
# e.g. "n_Federalgrants" → "avg_n_Federalgrants"
# Used so scoring loop finds the right z-score for CompositeZ
var_to_featcol = {}
for fc in feat_cols:
    # strip prefix to get base variable name
    prefix = get_prefix(FEATURE_AGG.get(
        next((f for f in FEATURE_COLS if fc.endswith(f)), fc), 'mean'))
    base = fc[len(prefix)+1:]  # remove prefix_ from front
    var_to_featcol[base] = fc

print(f"  Scoring {len(yearly):,} faculty-year rows...")

scored_rows = []
for _, row in yearly.iterrows():
    r = dict(row)

    # ── z-scores for all variables (display only for non-model vars) ──────
    # Model features: z-scored using training population stats
    for fc in feat_cols:
        prefix = get_prefix(FEATURE_AGG.get(
            next((f for f in FEATURE_COLS if fc.endswith(f)), fc), 'mean'))
        base = fc[len(prefix)+1:]
        val  = float(r.get(base, 0) or 0)
        m    = pop_mean.get(fc, 0.0)
        s    = pop_sd.get(fc, 1.0)
        r[f"z_{fc}"] = round((val-m)/s, 4) if s > 0 else 0.0

    # Non-model variables: z-scored using full-population stats
    for v in ALL_VARS:
        fc = f"avg_{v}"
        if fc not in feat_cols:
            val = float(r.get(v, 0) or 0)
            m   = pop_mean.get(fc, 0.0)
            s   = pop_sd.get(fc, 1.0)
            r[f"z_{v}"] = round((val-m)/s, 4) if s > 0 else 0.0

    # ── CompositeZ — model variables only ────────────────────────────────
    compositeZ = float(intcpt)
    for fc, c in zip(feat_cols, coefs):
        zv      = float(r.get(f"z_{fc}", 0.0))
        contrib = float(c) * zv
        compositeZ += contrib
        r[f"contrib_{fc}"] = round(contrib, 4)

    r['CompositeZ'] = round(compositeZ, 4)

    # ── Probability ───────────────────────────────────────────────────────
    z_clamped    = max(-500.0, min(500.0, compositeZ))
    r['Probability'] = round(
        float(np.exp(z_clamped) / (1.0 + np.exp(z_clamped))), 4)

    # ── Complexity component scores (0-1 vs p95) ─────────────────────────
    comp_scores = []
    for v in ALL_VARS:
        val   = float(r.get(v, 0) or 0)
        ceil  = p95.get(v, 1.0)
        score = round(min(val/ceil, 1.0), 4) if ceil > 0 else 0.0
        r[f"complexity_{v}"] = score
        comp_scores.append(score)
    r['ComplexityScore'] = round(float(np.mean(comp_scores)), 4)

    # ── Tier ──────────────────────────────────────────────────────────────
    p = r['Probability']
    if   int(r.get('IsCL', 0)) == 1: r['Tier'] = 'Confirmed CL'
    elif p >= 0.75:                   r['Tier'] = 'High Readiness'
    elif p >= 0.50:                   r['Tier'] = 'Emerging'
    elif p >= 0.30:                   r['Tier'] = 'Developing'
    else:                             r['Tier'] = 'Early Stage'

    scored_rows.append(r)

scored_df = pd.DataFrame(scored_rows)
print(f"  Done. Scored {len(scored_df):,} rows.")
print(f"  CL_Index range: {scored_df['CL_Index'].min():.0f} to {scored_df['CL_Index'].max():.0f}")
print(f"  CL_Period counts:\n" + scored_df['CL_Period'].value_counts().to_string())

# ══════════════════════════════════════════════════════════════════════════
# STEP 10 — SAVE ALL OUTPUTS
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 62)
print("STEP 10 — Saving outputs")
print("=" * 62)

# Add sort column for PBI axis sorting
for df in [yearly, scored_df]:
    df['CL_Index']      = pd.to_numeric(df['CL_Index'], errors='coerce')
    df['CL_Index_Sort'] = df['CL_Index'].fillna(999).astype(int)

yearly.to_csv(os.path.join(OUTPUT_DIR, "Faculty_Year_Aggregates.csv"), index=False)
scored_df.to_csv(os.path.join(OUTPUT_DIR, "faculty_year_scored.csv"), index=False)

model_df['Pred_Prob_CL'] = yp
model_df['CompositeZ']   = Xs@coefs + intcpt
for i, zn in enumerate(z_names):
    model_df[zn] = Xs[:,i]
model_df.to_csv(os.path.join(OUTPUT_DIR, "annualized_model_output.csv"), index=False)

pd.DataFrame({
    'Variable': ['Intercept']+z_names,
    'Coef':     np.round(all_c, 6),
    'OR':       ['—']+list(np.round(np.exp(coefs), 6)),
    'SE':       np.round(se, 6),
    'z':        np.round(z_st, 4),
    'p_value':  np.round(p_vs, 6),
    'sig':      [sig(p) for p in p_vs],
}).to_csv(os.path.join(OUTPUT_DIR, "annualized_model_coefficients.csv"), index=False)

pd.DataFrame({
    'FeatureCol': feat_cols,
    'Feature':    FEATURE_COLS,
    'Agg':        [FEATURE_AGG.get(f,'mean') for f in FEATURE_COLS],
    'Mean':       np.round(sc.mean_, 6),
    'SD':         np.round(sc.scale_, 6),
}).to_csv(os.path.join(OUTPUT_DIR, "annualized_model_zscore_stats.csv"), index=False)

(model_df[['Staff Name','IsCL','n_yrs_used','n_yrs_available',
           'window_start_fy','window_end_fy','Pred_Prob_CL','CompositeZ']]
 .sort_values('Pred_Prob_CL', ascending=False)
 .reset_index(drop=True)
 .to_csv(os.path.join(OUTPUT_DIR, "annualized_model_predictions.csv"), index=False))

pd.DataFrame(excluded_win, columns=['Staff Name','Reason'])\
  .to_csv(os.path.join(OUTPUT_DIR, "annualized_model_excluded_window.csv"), index=False)

excluded_total.rename(columns={'total_yrs':'YearsAvailable'})\
  .to_csv(os.path.join(OUTPUT_DIR, "annualized_model_excluded_total_years.csv"), index=False)

# ── Sigmoid chart data for Excel/PowerPoint ───────────────────────────────
z_vals     = [round(-6+i*0.25, 2) for i in range(49)]
sigmoid_df = pd.DataFrame({
    'z':           z_vals,
    'Probability': [round(1/(1+np.exp(-z)), 4) for z in z_vals]
})

def get_rep_score(g):
    if g['IsCL'].max() == 1:
        pre = g[g['CL_Period']=='pre-center']
        if len(pre) > 0:
            return pd.Series({'CompositeZ': round(pre['CompositeZ'].mean(),4),
                              'Probability': round(pre['Probability'].mean(),4),
                              'IsCL': 1})
    return pd.Series({'CompositeZ': round(g['CompositeZ'].mean(),4),
                      'Probability': round(g['Probability'].mean(),4),
                      'IsCL': int(g['IsCL'].max())})

fac_scores = scored_df.groupby('Staff Name').apply(get_rep_score).reset_index()
noncl_pts  = fac_scores[fac_scores['IsCL']==0][['CompositeZ','Probability']]
cl_pts     = fac_scores[fac_scores['IsCL']==1][['CompositeZ','Probability']]

chart_path = os.path.join(OUTPUT_DIR, "sigmoid_chart_data.xlsx")
with pd.ExcelWriter(chart_path) as writer:
    sigmoid_df.to_excel(writer, sheet_name='Sigmoid Curve',  index=False)
    noncl_pts.to_excel( writer, sheet_name='Non-CL Faculty', index=False)
    cl_pts.to_excel(    writer, sheet_name='CL Faculty',     index=False)

# ── DAX VAR block ─────────────────────────────────────────────────────────
ALL_DAX = [
    ("piGrants",        "n_piGrants"),
    ("Federalgrants",   "n_Federalgrants"),
    ("sponsors",        "n_sponsors"),
    ("Collaborators",   "total_Collaborators"),
    ("LogStaffDollars", "log_Total_Staff_Dollars"),
    ("LogTotalDollars", "log_Total_Dollars"),
    ("StaffDollars",    "Total_Staff_Dollars"),
    ("TotalDollars",    "Total_Dollars"),
    ("LogMaxAward",     "log_max_Award_Per_Year"),
    ("awards",          "n_awards"),
]
coef_lkp = dict(zip(feat_cols, coefs))
mean_lkp = dict(zip(feat_cols, sc.mean_))
sd_lkp   = dict(zip(feat_cols, sc.scale_))

print(f"\n{'='*62}")
print("DAX VAR BLOCK — paste into FacultyYearScored")
print(f"{'='*62}\n")
print(f"VAR coeff_Intercept          = {intcpt:.4f}")
for dv, feat in ALL_DAX:
    matched = next((fc for fc in feat_cols if fc.endswith(feat)), None)
    val     = coef_lkp[matched] if matched else 0.0
    note    = "" if matched else "  -- not in this model, set to 0"
    print(f"VAR coeff_{dv:<20} = {val:>8.4f}{note}")
print()
for dv, feat in ALL_DAX:
    matched = next((fc for fc in feat_cols if fc.endswith(feat)), None)
    val     = mean_lkp[matched] if matched else pop_mean.get(f"avg_{feat}", 0.0)
    print(f"VAR mean_{dv:<21} = {val:>8.4f}")
print()
for dv, feat in ALL_DAX:
    matched = next((fc for fc in feat_cols if fc.endswith(feat)), None)
    val     = sd_lkp[matched] if matched else pop_sd.get(f"avg_{feat}", 1.0)
    print(f"VAR sd_{dv:<23} = {val:>8.4f}")

print(f"\n{'='*62}")
print("OUTPUT FILES")
print(f"{'='*62}")
print("  Faculty_Year_Aggregates.csv             one row per faculty per year")
print("  faculty_year_scored.csv                 all years + z-scores + probability + tier")
print("  annualized_model_output.csv             one row per faculty (training set)")
print("  annualized_model_coefficients.csv       coefficients + p-values")
print("  annualized_model_zscore_stats.csv       training population means and SDs")
print("  annualized_model_predictions.csv        all faculty sorted by P(CL)")
print("  annualized_model_excluded_window.csv    excluded in window step")
print("  annualized_model_excluded_total_years.csv excluded for too few years")
print("  sigmoid_chart_data.xlsx                 curve + faculty dots for PowerPoint")
print(f"\nAll saved to: {OUTPUT_DIR}")
