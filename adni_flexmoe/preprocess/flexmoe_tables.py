"""Official UNITES-Lab/Flex-MoE tabular and FreeSurfer loaders.

Ported from:
  data/adni/clinical/clinical_preprocess.py
  data/adni/biospecimen/biospecimen_preprocess.py
  data.py (preprocessed image branch, commit 2168807)
"""
import glob
import os

import numpy as np
import pandas as pd
import pyreadr
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, StandardScaler


def find_file(data_root, subfolder, pattern):
    direct = os.path.join(data_root, subfolder, pattern)
    if os.path.exists(direct):
        return direct
    matches = glob.glob(os.path.join(data_root, "**", pattern), recursive=True)
    return matches[0] if matches else None


def read_table(path):
    if path.endswith(".rda"):
        frame = list(pyreadr.read_r(path).values())[0]
    else:
        frame = pd.read_csv(path, low_memory=False)
    frame.columns = [str(c).upper() for c in frame.columns]
    return frame


def load_labels(data_root):
    """Reproduce label.csv: keep subjects whose DIAGNOSIS is unique and in {1,2,3}."""
    path = find_file(data_root, "diagnosis", "DXSUM.csv")
    if path is None:
        path = find_file(data_root, "diagnosis", "DXSUM*.csv")
    if path is None:
        raise FileNotFoundError("DXSUM was not found under diagnosis/")
    label = read_table(path)
    label["DIAGNOSIS"] = pd.to_numeric(label["DIAGNOSIS"], errors="coerce")
    valid = label[label["DIAGNOSIS"].isin((1, 2, 3))].dropna(subset=["PTID"])
    unique = valid.groupby("PTID")["DIAGNOSIS"].nunique()
    keep = unique[unique == 1].index
    out = valid[valid["PTID"].isin(keep)][["PTID", "DIAGNOSIS"]].drop_duplicates("PTID")
    out["label"] = out["DIAGNOSIS"].astype(np.int64) - 1
    return out[["PTID", "label"]].reset_index(drop=True)


def latest_per_subject(df):
    """Official: sort by UPDATE_STAMP, then ffill/bfill and take the last row."""
    df = df[df["PTID"].notna()].copy()
    if "UPDATE_STAMP" in df.columns:
        df["UPDATE_STAMP"] = pd.to_datetime(df["UPDATE_STAMP"], errors="coerce")
        df = df.sort_values(["PTID", "UPDATE_STAMP"], kind="stable")
    else:
        df = df.sort_values("PTID", kind="stable")
    cols = [c for c in df.columns if c != "PTID"]
    df[cols] = df.groupby("PTID", sort=False)[cols].ffill()
    df[cols] = df.groupby("PTID", sort=False)[cols].bfill()
    return df.groupby("PTID", sort=False).tail(1).reset_index(drop=True)


def merge_tables(frames):
    latest = [latest_per_subject(df) for df in frames]
    common = set.intersection(*(set(df.columns) for df in latest))
    result = latest[0]
    for df in latest[1:]:
        result = pd.merge(result, df, on="PTID", how="outer", suffixes=("", "_DUP"))
        for col in df.columns:
            dup = col + "_DUP"
            if col not in ("PTID", "UPDATE_STAMP") and dup in result.columns:
                result[col] = result[col].combine_first(result[dup])
                result.drop(columns=dup, inplace=True)
    result = result.drop(columns=[c for c in common if c != "PTID" and c in result.columns])
    result = result.loc[:, ~result.columns.duplicated()]
    return result.set_index("PTID")


def process_num_cat_cols(df, initial_filling="mean"):
    """Exact Flex-MoE numeric / categorical / date encoding."""
    categorical_col_idx, numerical_col_idx, date_col_idx = [], [], []
    pattern = r"[-:\sa-zA-Z]"
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or "DATE" in col:
            date_col_idx.append(col)
        elif (df[col].dtypes == int) or (df[col].dtypes == float):
            (categorical_col_idx if df[col].nunique() <= 2 else numerical_col_idx).append(col)
        elif df[col].astype(str).str.contains(pattern).any():
            categorical_col_idx.append(col)
        else:
            numerical_col_idx.append(col)

    parts = []
    if numerical_col_idx:
        df_num = df[numerical_col_idx].copy()
        for column in df_num.columns:
            df_num[column] = (
                df_num[column].astype(str).str.replace("<", "", regex=False)
                .str.replace(">", "", regex=False)
            )
            df_num[column] = pd.to_numeric(df_num[column], errors="coerce")
        scaled = MinMaxScaler(feature_range=(-1, 1)).fit_transform(df_num)
        df_num = pd.DataFrame(scaled, columns=numerical_col_idx, index=df.index)
        if df_num.isnull().sum().sum() > 0 and initial_filling == "mean":
            df_num = df_num.fillna(df_num.mean().iloc[0])
        elif df_num.isnull().sum().sum() > 0 and initial_filling == "median":
            df_num = df_num.fillna(df_num.median().iloc[0])
        parts.append(df_num)

    if categorical_col_idx:
        df_cat = df[categorical_col_idx].copy()
        df_cat = df_cat.loc[:, df_cat.nunique() < 1000]
        if len(df_cat.columns):
            if df_cat.isnull().sum().sum() > 0 and initial_filling != "None":
                df_cat = df_cat.fillna(df_cat.mode().iloc[0])
            df_cat = pd.get_dummies(df_cat, columns=list(df_cat.columns))
            df_cat = df_cat.replace({True: 1, False: -1})
            parts.append(df_cat)

    if date_col_idx:
        df_date = df.loc[:, date_col_idx]
        df_date = df_date.apply(pd.to_datetime, errors="coerce")
        if df_date.isnull().sum().sum() > 0 and initial_filling != "None":
            df_date = df_date.fillna(df_date.mode().iloc[0])
        kbin = KBinsDiscretizer(n_bins=5, encode="onehot-dense", strategy="uniform")
        date_bins = []
        for col in list(dict.fromkeys(date_col_idx)):
            df_temp = df_date.loc[:, [col]].dropna()
            if df_temp.shape[1] != 1 or df_temp.empty:
                continue
            seconds = (df_temp.iloc[:, 0].astype("int64") // 10**9).to_numpy().reshape(-1, 1)
            n_bins = 5
            if np.unique(seconds).size < 2:
                binned = np.ones((len(seconds), 1), dtype=np.float32)
            else:
                n_bins = min(5, int(np.unique(seconds).size))
                binner = KBinsDiscretizer(n_bins=n_bins, encode="onehot-dense", strategy="uniform")
                binned = binner.fit_transform(seconds)
            date_bins.append(pd.DataFrame(
                binned, columns=[f"{col}_BIN{i}" for i in range(binned.shape[1])], index=df_temp.index
            ))
        if date_bins:
            parts.append(pd.concat(date_bins, axis=1).reindex(df.index))

    if not parts:
        raise ValueError("No usable Flex-MoE features remained")
    out = pd.concat(parts, axis=1)
    return out.replace({True: 1, False: -1}).astype(np.float32)


def load_clinical(data_root, ptids):
    names = ("MEDHIST", "NEUROEXM", "PTDEMOG", "RECCMEDS", "VITALS")
    frames = []
    for name in names:
        path = find_file(data_root, "clinical", f"{name}.rda") or find_file(data_root, "clinical", f"{name}*.csv")
        if path is None:
            raise FileNotFoundError(f"Missing Flex-MoE clinical table {name}")
        frames.append(read_table(path))
    merged = merge_tables(frames)
    merged = merged.drop(columns=[c for c in merged.columns if c.startswith(("PTCOGBEG", "PTADDX", "PTADBEG"))])
    encoded = process_num_cat_cols(merged, initial_filling="mean")
    observed = ptids.isin(encoded.index).to_numpy()
    return encoded.reindex(ptids).fillna(0).to_numpy(np.float32), observed


def load_biospecimen(data_root, ptids):
    apoe_path = find_file(data_root, "biospecimen", "APOERES.csv") or find_file(data_root, "biospecimen", "APOERES*.csv")
    roche_path = (find_file(data_root, "biospecimen", "UPENNBIOMK_ROCHE_ELECSYS.rda")
                  or find_file(data_root, "biospecimen", "UPENNBIOMK_ROCHE_ELECSYS*.csv"))
    if apoe_path is None or roche_path is None:
        raise FileNotFoundError("Missing APOERES or UPENNBIOMK_ROCHE_ELECSYS")
    apoe, roche = read_table(apoe_path), read_table(roche_path)
    need_map = {"PHASE", "RID"}.issubset(roche.columns)
    if need_map and ("PTID" not in roche.columns or roche["PTID"].isna().all()):
        id_map = {}
        for _, row in apoe.dropna(subset=["PHASE", "RID", "PTID"]).iterrows():
            id_map[f"{str(row['PHASE'])[-1]}_{int(row['RID'])}"] = row["PTID"]
        roche["PTID"] = [
            id_map.get(f"{str(p)[-1]}_{int(r)}") if pd.notna(p) and pd.notna(r) else np.nan
            for p, r in zip(roche["PHASE"], roche["RID"])
        ]
    encoded = process_num_cat_cols(merge_tables([apoe, roche]), initial_filling="mean")
    observed = ptids.isin(encoded.index).to_numpy()
    return encoded.reindex(ptids).fillna(0).to_numpy(np.float32), observed


def load_image(data_root, ptids):
    """Official preprocessed image path: UCSFFSX7 ST* {CV,TA,SV}, latest UPDATE_STAMP, StandardScaler."""
    path = find_file(data_root, "image", "UCSFFSX7*.csv")
    if path is None:
        raise FileNotFoundError("Missing UCSF Cross-Sectional FreeSurfer 7.x CSV")
    df = read_table(path)
    stamp = "UPDATE_STAMP"
    df[stamp] = pd.to_datetime(df[stamp], errors="coerce")
    df = df.loc[df.groupby("PTID")[stamp].idxmax()].set_index("PTID")
    cols = [c for c in df.columns if c.startswith("ST") and c.endswith(("CV", "TA", "SV"))]
    values = df[cols].apply(pd.to_numeric, errors="coerce")
    if values.isnull().sum().sum() > 0:
        values = values.apply(lambda col: col.fillna(col.mode().iloc[0] if not col.mode().empty else 0))
    values = pd.DataFrame(StandardScaler().fit_transform(values.fillna(0)), index=values.index, columns=cols)
    observed = ptids.isin(values.index).to_numpy()
    return values.reindex(ptids).fillna(0).to_numpy(np.float32), observed
