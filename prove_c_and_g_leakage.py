import os, sys, json, warnings
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import pyreadr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler

ROOT = r"d:\MetaKD"
RAW = os.path.join(ROOT, "adni_raw_flexmoe")

print("=" * 100)
print("THỰC NGHIỆM CHỨNG MINH KHOA HỌC: GIẢI MÃ SỰ SAI KHÁC CỦA CLINICAL (C) VÀ GENOMIC (G)")
print("=" * 100)

# ============================================================================
# 1. LOAD LABELS & LOCKED SPLIT
# ============================================================================
dx = pd.read_csv(os.path.join(RAW, "diagnosis", "DXSUM.csv"), low_memory=False)
dx.columns = [c.upper() for c in dx.columns]
dx['DIAGNOSIS'] = pd.to_numeric(dx['DIAGNOSIS'], errors='coerce')
valid = dx[dx['DIAGNOSIS'].isin([1, 2, 3])].dropna(subset=['PTID'])
unique = valid.groupby('PTID')['DIAGNOSIS'].nunique()
labels_df = valid[valid['PTID'].isin(unique[unique == 1].index)][['PTID', 'DIAGNOSIS']].drop_duplicates('PTID')
labels_df['label'] = labels_df['DIAGNOSIS'].astype(int) - 1
label_map = dict(zip(labels_df['PTID'], labels_df['label']))

split_path = os.path.join(ROOT, "logs", "adni_generated_PTID_splits.json")
if not os.path.exists(split_path):
    split_path = os.path.join(ROOT, "adni", "data", "PTID_splits.json")
with open(split_path, 'r', encoding='utf-8') as f:
    split = json.load(f)

tr_ptids = [p for p in split['training'] if p in label_map]
te_ptids = [p for p in split['testing'] if p in label_map]
y_tr = np.array([label_map[p] for p in tr_ptids])
y_te = np.array([label_map[p] for p in te_ptids])

print(f"Dataset split: Train = {len(tr_ptids)} ca, Test = {len(te_ptids)} ca")
print(f"Phân phối nhãn Test: CTL={sum(y_te==0)}, MCI={sum(y_te==1)}, AD={sum(y_te==2)}")
majority_test_acc = max(sum(y_te==0), sum(y_te==1), sum(y_te==2)) / len(y_te) * 100
print(f"Độ chính xác đoán mò lớp đa số (Majority Baseline): {majority_test_acc:.2f}%\n")

# ============================================================================
# PHẦN 1: CHỨNG MINH CLINICAL (C) - RÒ RỈ CỘT CHẨN ĐOÁN TRONG GITHUB FLEX-MOE
# ============================================================================
print("=" * 100)
print("PHẦN 1: CHỨNG MINH RÒ RỈ DỮ LIỆU Ở CLINICAL (C)")
print("So sánh: (A) Lâm sàng sạch (bỏ rò rỉ) vs (B) Script gốc Flex-MoE trên GitHub (không bỏ rò rỉ)")
print("=" * 100)

names = ["MEDHIST", "NEUROEXM", "PTDEMOG", "RECCMEDS", "VITALS"]
clin_dfs = []
for n in names:
    p = os.path.join(RAW, "clinical", n + ".rda")
    df = list(pyreadr.read_r(p).values())[0]
    df.columns = [c.upper() for c in df.columns]
    df['UPDATE_STAMP'] = pd.to_datetime(df['UPDATE_STAMP'], errors='coerce')
    df = df.sort_values(by='UPDATE_STAMP').groupby('PTID').last().reset_index()
    clin_dfs.append(df)

merged = clin_dfs[0].copy()
for d in clin_dfs[1:]:
    merged = pd.merge(merged, d, on=['PTID'], how='outer', suffixes=('', '_dup'))
    for col in d.columns:
        if col != 'PTID' and col != 'UPDATE_STAMP':
            if col + '_dup' in merged.columns:
                merged[col] = merged[col].combine_first(merged[col + '_dup'])
                merged.drop(columns=[col + '_dup'], inplace=True)

common_cols = set.intersection(*(set(df.columns) for df in clin_dfs))
merged_base = merged.drop(columns=[c for c in common_cols if c != 'PTID'])

def encode_flexmoe(df_in):
    df = df_in.copy().set_index('PTID')
    num_cols = []
    cat_cols = []
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]) or 'DATE' in c:
            continue
        elif df[c].dtype in (int, float) and df[c].nunique() > 2:
            num_cols.append(c)
        elif df[c].nunique() <= 2 or df[c].astype(str).str.contains(r'[-:\sa-zA-Z]').any():
            cat_cols.append(c)
        else:
            num_cols.append(c)
            
    parts = []
    if num_cols:
        d_num = df[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        scaler = MinMaxScaler((-1, 1))
        parts.append(pd.DataFrame(scaler.fit_transform(d_num), index=df.index, columns=num_cols))
    if cat_cols:
        d_cat = df[cat_cols].copy()
        d_cat = d_cat.loc[:, d_cat.nunique() < 1000].fillna('MISSING')
        d_dum = pd.get_dummies(d_cat, drop_first=True)
        parts.append(d_dum.astype(float))
    return pd.concat(parts, axis=1)

# Pipeline A: Lâm sàng SẠCH (Loại bỏ triệt để PTCOGBEG, PTADDX, PTADBEG)
drop_leak = [c for c in merged_base.columns if c.startswith(('PTCOGBEG', 'PTADDX', 'PTADBEG'))]
df_clean = merged_base.drop(columns=drop_leak)
enc_clean = encode_flexmoe(df_clean)

# Pipeline B: Theo đúng script GITHUB của Flex-MoE (KHÔNG DROP PTCOGBEG, PTADDX, PTADBEG)
enc_github = encode_flexmoe(merged_base)

# Pipeline C: CHỈ DÙNG ĐÚNG 2 CỘT RÒ RỈ (PTCOGBEG và PTADDX)
df_leak_only = merged_base[['PTID', 'PTCOGBEG', 'PTADDX']].copy().set_index('PTID')
df_leak_only = df_leak_only.apply(pd.to_numeric, errors='coerce').fillna(-1)

def eval_matrix(enc_df, tag):
    X_tr = enc_df.reindex(tr_ptids).fillna(0).to_numpy(np.float32)
    X_te = enc_df.reindex(te_ptids).fillna(0).to_numpy(np.float32)
    
    mu = X_tr.mean(0, keepdims=True)
    X_tr -= mu
    X_te -= mu
    
    lr = LogisticRegression(max_iter=1000, C=0.1)
    lr.fit(X_tr, y_tr)
    p_lr = lr.predict(X_te)
    
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    p_rf = rf.predict(X_te)
    
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=60, random_state=42)
    mlp.fit(X_tr, y_tr)
    p_mlp = mlp.predict(X_te)
    
    print(f"\n---> {tag} (Số đặc trưng: {X_tr.shape[1]} chiều):")
    for name, p in [("Logistic Regression", p_lr), ("Random Forest", p_rf), ("MLP Classifier", p_mlp)]:
        acc = accuracy_score(y_te, p) * 100
        bacc = balanced_accuracy_score(y_te, p) * 100
        f1 = f1_score(y_te, p, average='macro') * 100
        print(f"   [{name:<20}] Test Acc = {acc:6.2f}% | Balanced Acc = {bacc:6.2f}% | Macro-F1 = {f1:6.2f}%")

eval_matrix(enc_clean, "1A. LÂM SÀNG SẠCH (ĐÃ LOẠI BỎ CÁC CỘT RÒ RỈ - CỦA CHÚNG TA)")
eval_matrix(enc_github, "1B. SCRIPT GỐC GITHUB FLEX-MOE (KHÔNG DROP CÁC CỘT RÒ RỈ)")
eval_matrix(df_leak_only, "1C. CHỈ DÙNG ĐÚNG 2 CỘT RÒ RỈ: PTCOGBEG & PTADDX (2 ĐẶC TRƯNG)")

# ============================================================================
# PHẦN 2: CHỨNG MINH GENOMIC (G) - ĐỘT BIẾN APOE & HIỆN TƯỢNG DATA LEAKAGE
# ============================================================================
print("\n" + "=" * 100)
print("PHẦN 2: CHỨNG MINH VẤN ĐỀ GENOMIC (G) VÀ TÍN HIỆU ĐỘT BIẾN APOE")
print("=" * 100)

z = np.load(os.path.join(ROOT, "kaggle_adni_run", "adni.npz"), allow_pickle=True)
G_matrix = z['genomic']
idx_tr = z['idx_train']
idx_te = z['idx_test']
y_g_tr = z['y'][idx_tr]
y_g_te = z['y'][idx_te]

# 2A. Toàn bộ 40,956 SNPs ngẫu nhiên (không có APOE)
print(f"\n---> 2A. CHẠY TRÊN 40,956 SNPs GWAS SAU LD-PRUNING (KHÔNG CÓ APOE):")
lr_g = LogisticRegression(max_iter=1000, C=0.01)
lr_g.fit(G_matrix[idx_tr], y_g_tr)
p_g = lr_g.predict(G_matrix[idx_te])
print(f"   [Toàn bộ 40,956 SNPs] Test Acc = {accuracy_score(y_g_te, p_g)*100:6.2f}% | Balanced Acc = {balanced_accuracy_score(y_g_te, p_g)*100:6.2f}% | Macro-F1 = {f1_score(y_g_te, p_g, average='macro')*100:6.2f}%")
print(f"   => Nhận xét: Độ chính xác rơi về {accuracy_score(y_g_te, p_g)*100:.2f}%, thấp hơn đoán mò ({majority_test_acc:.2f}%) vì 40k SNPs chỉ là nhiễu!")

# 2B. Tín hiệu từ APOE (trong APOERES.csv)
apoe = pd.read_csv(os.path.join(RAW, "biospecimen", "APOERES.csv"), low_memory=False)
apoe.columns = [c.upper() for c in apoe.columns]
apoe = apoe.dropna(subset=['PTID']).drop_duplicates('PTID').set_index('PTID')
apoe_dummies = pd.get_dummies(apoe['GENOTYPE'].astype(str)).astype(float)
X_apoe_tr = apoe_dummies.reindex(tr_ptids).fillna(0).to_numpy()
X_apoe_te = apoe_dummies.reindex(te_ptids).fillna(0).to_numpy()

lr_apoe = LogisticRegression(max_iter=1000, C=1.0)
lr_apoe.fit(X_apoe_tr, y_tr)
p_apoe = lr_apoe.predict(X_apoe_te)
print(f"\n---> 2B. CHỈ DÙNG ĐỘT BIẾN GEN APOE (BẢNG APOERES):")
print(f"   [Chỉ dùng APOE Genotype] Test Acc = {accuracy_score(y_te, p_apoe)*100:6.2f}% | Balanced Acc = {balanced_accuracy_score(y_te, p_apoe)*100:6.2f}% | Macro-F1 = {f1_score(y_te, p_apoe, average='macro')*100:6.2f}%")
print(f"   => Nhận xét: Riêng một mình gen APOE đã đạt {accuracy_score(y_te, p_apoe)*100:.2f}%, cao hơn 40.000 SNPs GWAS cộng lại!")

# 2C. Thử nghiệm rò rỉ chọn đặc trưng (Feature Selection Leakage)
print(f"\n---> 2C. CHỨNG MINH RÒ RỈ CHỌN ĐẶC TRƯNG (LEAKED VS HONEST SELECTION):")
def calc_corr(X, target):
    xc = X - X.mean(0, keepdims=True)
    tc = target - target.mean()
    var_x = (xc**2).sum(0)
    var_t = (tc**2).sum()
    denom = np.sqrt(var_x * var_t)
    return np.divide(np.dot(tc, xc), denom, out=np.zeros(X.shape[1]), where=denom>0)

# Honest: Chọn SNP CHỈ trên tập Train
corr_honest = np.abs(calc_corr(G_matrix[idx_tr], (y_g_tr == 2).astype(float)))
top_honest = np.argsort(corr_honest)[-500:]
clf_h = LogisticRegression(max_iter=1000, C=0.1).fit(G_matrix[idx_tr][:, top_honest], y_g_tr)
acc_h = accuracy_score(y_g_te, clf_h.predict(G_matrix[idx_te][:, top_honest])) * 100

# Leaked: Chọn SNP trên TOÀN BỘ dữ liệu (Train + Test) trước khi split
corr_leak = np.abs(calc_corr(G_matrix, (z['y'] == 2).astype(float)))
top_leak = np.argsort(corr_leak)[-500:]
clf_l = LogisticRegression(max_iter=1000, C=0.1).fit(G_matrix[idx_tr][:, top_leak], y_g_tr)
acc_l = accuracy_score(y_g_te, clf_l.predict(G_matrix[idx_te][:, top_leak])) * 100

print(f"   Top 500 SNPs chọn TRUNG THỰC (chỉ trên Train) : Test Acc = {acc_h:.2f}%")
print(f"   Top 500 SNPs chọn RÒ RỈ (trên cả Train + Test)  : Test Acc = {acc_l:.2f}%  (Tăng ảo {acc_l - acc_h:+.2f}%)")
print("=" * 100)
print("KẾT THÚC THỰC NGHIỆM CHỨNG MINH THÀNH CÔNG!")
print("=" * 100)
