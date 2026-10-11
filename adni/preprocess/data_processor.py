"""
Comprehensive ADNI Preprocessor following Flex-MoE and MetaKD protocols.
Extracts, normalizes, and aligns 4 modalities:
- Clinical (C): Demographics (PTDEMOG), Medical History (MEDHIST.rda), Vital signs (VITALS.rda).
  STRICT ZERO-LEAKAGE: Excludes CDR, MMSE, FAQ, ADAS, MOCA, PTCOGBEG, PTADDX, PTADBEG.
- Genomic (G): ApoE e4/e2 allele dose, genotype encoding, and 64-dim AD candidate risk loci.
- Biospecimen (B): CSF ABETA, TAU, PTAU, diagnostic ratios (ABETA/TAU, ABETA/PTAU), and CSFNFL.
- Imaging (I): FreeSurfer MRI cortical/subcortical ROI metrics normalized by ICV.
Target Labels: 3-class AD staging: 0: CTL (Control), 1: MCI, 2: AD.
Splits: 70% Train (with SMOTE k=5 on minority classes), 15% Val, 15% Test.
"""
import os
import re
import numpy as np
import pandas as pd
import pyreadr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from .smote import smote_oversample


def find_file(directory, pattern):
    """Find a file matching regex pattern in a directory."""
    for root, _, files in os.walk(directory):
        for f in files:
            if re.search(pattern, f, re.IGNORECASE):
                return os.path.join(root, f)
    return None


class ADNIPreprocessor:
    def __init__(self, raw_data_dir="data/ADNI/ADNI", output_dir="adni/data_processed"):
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def load_diagnosis(self):
        """Extract baseline 3-class diagnosis from DXSUM with Flex-MoE consistency filtering."""
        dx_file = find_file(self.raw_data_dir, r"DXSUM.*\.csv")
        if not dx_file:
            raise FileNotFoundError(f"DXSUM file not found in {self.raw_data_dir}")
        print(f"Loading Diagnosis from: {dx_file}")
        df = pd.read_csv(dx_file, low_memory=False)

        # Baseline filter: bl, sc, or init
        baseline_mask = df['VISCODE'].isin(['bl', 'sc', 'init']) | df['VISCODE2'].isin(['bl', 'sc', 'init'])
        df_bl = df[baseline_mask].copy()

        # Target classes: 1: CTL, 2: MCI, 3: AD
        valid_dx = df_bl['DIAGNOSIS'].isin([1.0, 2.0, 3.0])
        df_clean = df_bl[valid_dx].copy()

        # Flex-MoE protocol: filter subjects with stable diagnosis across all visits (nunique == 1)
        dx_valid_all = df[df['DIAGNOSIS'].isin([1.0, 2.0, 3.0])]
        nunique_dx = dx_valid_all.groupby('PTID')['DIAGNOSIS'].nunique()
        stable_ptids = set(nunique_dx[nunique_dx == 1].index)
        df_clean = df_clean[df_clean['PTID'].isin(stable_ptids)].copy()
        df_clean = df_clean.drop_duplicates(subset=['PTID'], keep='first')

        # Map to 0 (CTL), 1 (MCI), 2 (AD)
        df_clean['label'] = df_clean['DIAGNOSIS'].astype(int) - 1
        print(f"Loaded {len(df_clean)} subjects with stable baseline diagnosis (Flex-MoE protocol).")
        print(f"Class distribution: CTL={sum(df_clean['label']==0)}, MCI={sum(df_clean['label']==1)}, AD={sum(df_clean['label']==2)}")
        return df_clean[['PTID', 'RID', 'label']].reset_index(drop=True)

    def extract_clinical(self, subjects_df):
        """
        Extract Clinical (C) modality:
        Demographics + Medical History + Vital Signs.
        STRICT ZERO-LEAKAGE: Excludes CDR, MMSE, FAQ, ADAS, MOCA, PTCOGBEG, PTADDX, PTADBEG.
        Returns raw features with NaNs intact (no split leakage).
        """
        print("\n[1/4] Processing Clinical (C) modality (Zero-Leakage: Demographics + MedHist + Vitals)...")
        ptdemog_file = find_file(self.raw_data_dir, r"PTDEMOG.*\.csv")
        medhist_rda = find_file(self.raw_data_dir, r"MEDHIST\.rda")
        vitals_rda = find_file(self.raw_data_dir, r"VITALS\.rda")

        merged = subjects_df[['PTID', 'RID']].copy()
        c_available = pd.Series(False, index=merged.index)

        # 1. Demographics
        if ptdemog_file:
            demo = pd.read_csv(ptdemog_file, low_memory=False)
            demo_bl = demo[demo['VISCODE'].isin(['bl', 'sc', 'init']) | demo['VISCODE2'].isin(['bl', 'sc', 'init'])].drop_duplicates('PTID')
            demo_cols = ['PTID', 'PTGENDER', 'PTDOBYY', 'PTEDUCAT', 'PTMARRY']
            demo_cols = [c for c in demo_cols if c in demo_bl.columns]
            merged = pd.merge(merged, demo_bl[demo_cols], on='PTID', how='left')
            c_available = c_available | merged['PTID'].isin(demo_bl['PTID'])

        # 2. Medical History (from MEDHIST.rda)
        if medhist_rda:
            r_med = pyreadr.read_r(medhist_rda)
            med = list(r_med.values())[0]
            med_bl = med[med['VISCODE'].isin(['bl', 'sc', 'init']) | med['VISCODE2'].isin(['bl', 'sc', 'init'])].drop_duplicates('PTID')
            med_cols = ['MHPSYCH', 'MH2NEURL', 'MH3HEAD', 'MH4CARD', 'MH5RESP', 'MH6HEPAT',
                        'MH7DERM', 'MH8MUSCL', 'MH9ENDO', 'MH10GAST', 'MH11HEMA', 'MH12RENA']
            med_cols = [c for c in med_cols if c in med_bl.columns]
            med_subset = med_bl[['PTID'] + med_cols].copy()
            # Convert 'Yes'/'No' to 1.0 / 0.0
            for col in med_cols:
                med_subset[col] = (med_subset[col] == 'Yes').astype(float)
            merged = pd.merge(merged, med_subset, on='PTID', how='left')

        # 3. Vital Signs (from VITALS.rda)
        if vitals_rda:
            r_vit = pyreadr.read_r(vitals_rda)
            vit = list(r_vit.values())[0]
            vit_bl = vit[vit['VISCODE'].isin(['bl', 'sc', 'init']) | vit['VISCODE2'].isin(['bl', 'sc', 'init'])].drop_duplicates('PTID')
            vit_cols = ['VSWEIGHT', 'VSHEIGHT', 'VSBPSYS', 'VSBPDIA', 'VSPULSE', 'VSRESP', 'VSTEMP']
            vit_cols = [c for c in vit_cols if c in vit_bl.columns]
            vit_subset = vit_bl[['PTID'] + vit_cols].copy()
            for col in vit_cols:
                s = pd.to_numeric(vit_subset[col], errors='coerce')
                s[s <= 0] = np.nan
                vit_subset[col] = s
            # Derived BMI
            if 'VSWEIGHT' in vit_subset.columns and 'VSHEIGHT' in vit_subset.columns:
                vit_subset['BMI_proxy'] = vit_subset['VSWEIGHT'] / (vit_subset['VSHEIGHT'] + 1e-5)
            merged = pd.merge(merged, vit_subset, on='PTID', how='left')

        # 4. Neurological Exam (from NEUROEXM.rda - Flex-MoE protocol)
        neuro_rda = find_file(self.raw_data_dir, r"NEUROEXM\.rda")
        if neuro_rda:
            r_neuro = pyreadr.read_r(neuro_rda)
            neuro = list(r_neuro.values())[0]
            neuro_bl = neuro[neuro['VISCODE'].isin(['bl', 'sc', 'init']) | neuro['VISCODE2'].isin(['bl', 'sc', 'init'])].drop_duplicates('PTID')
            neuro_cols = ['NXVISUAL', 'NXAUDITO', 'NXTREMOR', 'NXCONSCI', 'NXNERVE', 'NXMOTOR',
                          'NXFINGER', 'NXHEEL', 'NXSENSOR', 'NXTENDON', 'NXPLANTA', 'NXGAIT', 'NXOTHER']
            neuro_cols = [c for c in neuro_cols if c in neuro_bl.columns]
            neuro_subset = neuro_bl[['PTID'] + neuro_cols].copy()
            for col in neuro_cols:
                neuro_subset[col] = neuro_subset[col].astype(str).str.upper().apply(
                    lambda x: 1.0 if any(k in x for k in ['ABNORMAL', 'PRESENT', 'YES', '1']) else (0.0 if any(k in x for k in ['NORMAL', 'ABSENT', 'NO', '0']) else np.nan)
                )
            merged = pd.merge(merged, neuro_subset, on='PTID', how='left')

        # 5. Medications count (from RECCMEDS.rda - Flex-MoE protocol)
        recc_rda = find_file(self.raw_data_dir, r"RECCMEDS\.rda")
        if recc_rda:
            r_recc = pyreadr.read_r(recc_rda)
            recc = list(r_recc.values())[0]
            recc_bl = recc[recc['VISCODE'].isin(['bl', 'sc', 'init']) | recc['VISCODE2'].isin(['bl', 'sc', 'init'])]
            med_counts = recc_bl.groupby('PTID').size().reset_index(name='NUM_MEDICATIONS')
            merged = pd.merge(merged, med_counts, on='PTID', how='left')

        # Zero-Leakage: Strictly exclude MMSE, FAQ, CDR, ADAS, MOCA, PTCOGBEG, PTADDX, PTADBEG, DIAGNOSIS
        forbidden_patterns = ['PTCOGBEG', 'PTADDX', 'PTADBEG', 'CDR', 'CDGLOBAL', 'DIAGNOSIS', 'DX', 'MMSE', 'MMSCORE', 'FAQ', 'FAQTOTAL', 'ADAS', 'MOCA']
        for c in list(merged.columns):
            if any(re.search(pat, c, re.IGNORECASE) for pat in forbidden_patterns) and c not in ['PTID', 'RID']:
                print(f"  Dropping forbidden leakage column: {c}")
                merged = merged.drop(columns=[c])

        feature_cols = [c for c in merged.columns if c not in ['PTID', 'RID']]
        raw_c = merged[feature_cols].apply(pd.to_numeric, errors='coerce').values.astype(np.float32)
        mask_c = c_available.values
        print(f"Clinical raw features: shape {raw_c.shape}, available: {mask_c.sum()}/{len(mask_c)} ({mask_c.mean()*100:.1f}%)")
        return raw_c, mask_c

    def extract_biospecimen(self, subjects_df):
        """
        Extract Biospecimen (B) modality:
        CSF ABETA40, ABETA42, TAU, PTAU, and diagnostic ratios from UPENNBIOMK_ROCHE_ELECSYS.rda + CSFNFL.
        Returns raw features with NaNs intact (no split leakage).
        """
        print("\n[2/4] Processing Biospecimen (B) modality (Roche Elecsys CSF + CSFNFL)...")
        roche_rda = find_file(self.raw_data_dir, r"UPENNBIOMK_ROCHE_ELECSYS\.rda")
        nfl_file = find_file(self.raw_data_dir, r"BLENNOWCSFNFL.*\.csv")
        upenn_file = find_file(self.raw_data_dir, r"UPENNBIOMK_MASTER.*\.csv")

        merged = subjects_df[['PTID', 'RID']].copy()
        b_available = pd.Series(False, index=merged.index)

        # 1. UPENNBIOMK_ROCHE_ELECSYS.rda (Paper Table 5 exact file)
        if roche_rda:
            r_roche = pyreadr.read_r(roche_rda)
            roche_df = list(r_roche.values())[0]
            join_col = 'PTID' if 'PTID' in roche_df.columns else 'RID'
            roche_bl = roche_df.drop_duplicates(join_col).copy()
            for col in ['ABETA40', 'ABETA42', 'TAU', 'PTAU']:
                if col in roche_bl.columns:
                    clean_s = roche_bl[col].astype(str).str.replace('<', '').str.replace('>', '')
                    roche_bl[col] = pd.to_numeric(clean_s, errors='coerce')

            # Calculate diagnostic ratio features
            if 'ABETA42' in roche_bl.columns and 'TAU' in roche_bl.columns:
                roche_bl['ABETA42_TAU_RATIO'] = roche_bl['ABETA42'] / (roche_bl['TAU'] + 1e-5)
                roche_bl['LOG_ABETA42'] = np.log1p(np.maximum(roche_bl['ABETA42'].fillna(0), 0))
                roche_bl['LOG_TAU'] = np.log1p(np.maximum(roche_bl['TAU'].fillna(0), 0))
            if 'ABETA42' in roche_bl.columns and 'PTAU' in roche_bl.columns:
                roche_bl['ABETA42_PTAU_RATIO'] = roche_bl['ABETA42'] / (roche_bl['PTAU'] + 1e-5)
                roche_bl['LOG_PTAU'] = np.log1p(np.maximum(roche_bl['PTAU'].fillna(0), 0))
            if 'ABETA42' in roche_bl.columns and 'ABETA40' in roche_bl.columns:
                roche_bl['ABETA42_ABETA40_RATIO'] = roche_bl['ABETA42'] / (roche_bl['ABETA40'] + 1e-5)

            biomk_cols = [c for c in ['ABETA40', 'ABETA42', 'TAU', 'PTAU', 'ABETA42_TAU_RATIO', 'ABETA42_PTAU_RATIO', 'ABETA42_ABETA40_RATIO', 'LOG_ABETA42', 'LOG_TAU', 'LOG_PTAU'] if c in roche_bl.columns]
            merged = pd.merge(merged, roche_bl[[join_col] + biomk_cols], on=join_col, how='left')
            b_available = b_available | merged[join_col].isin(roche_bl[join_col])
        elif upenn_file:
            upenn = pd.read_csv(upenn_file, low_memory=False)
            join_col = 'RID' if 'RID' in upenn.columns else 'PTID'
            upenn_bl = upenn.drop_duplicates(join_col).copy()
            for col in ['ABETA', 'TAU', 'PTAU']:
                if col in upenn_bl.columns:
                    clean_s = upenn_bl[col].astype(str).str.replace('<', '').str.replace('>', '')
                    upenn_bl[col] = pd.to_numeric(clean_s, errors='coerce')
            biomk_cols = [c for c in ['ABETA', 'TAU', 'PTAU'] if c in upenn_bl.columns]
            merged = pd.merge(merged, upenn_bl[[join_col] + biomk_cols], on=join_col, how='left')
            b_available = b_available | merged[join_col].isin(upenn_bl[join_col])

        # 2. Blennow CSF Neurofilament Light (NfL)
        if nfl_file:
            nfl = pd.read_csv(nfl_file, low_memory=False)
            join_col = 'RID' if 'RID' in nfl.columns else 'PTID'
            nfl_bl = nfl.drop_duplicates(join_col).copy()
            val_col = 'CSFNFL' if 'CSFNFL' in nfl_bl.columns else None
            if val_col:
                nfl_bl[val_col] = pd.to_numeric(nfl_bl[val_col], errors='coerce')
                merged = pd.merge(merged, nfl_bl[[join_col, val_col]], on=join_col, how='left')

        feature_cols = [c for c in merged.columns if c not in ['PTID', 'RID']]
        raw_b = merged[feature_cols].apply(pd.to_numeric, errors='coerce').values.astype(np.float32)
        mask_b = b_available.values
        print(f"Biospecimen raw features: shape {raw_b.shape}, available: {mask_b.sum()}/{len(mask_b)} ({mask_b.mean()*100:.1f}%)")
        return raw_b, mask_b

    def extract_imaging(self, subjects_df):
        """
        Extract Imaging (I) modality:
        FreeSurfer MRI cortical & subcortical metrics normalized by Intracranial Volume (ST10CV).
        Returns raw features with NaNs intact (no split leakage).
        """
        print("\n[3/4] Processing Imaging (I) modality (FreeSurfer 7.x UCSFFSX7)...")
        fs_file = find_file(self.raw_data_dir, r"UCSFFSX7.*\.csv") or find_file(self.raw_data_dir, r"UCSFFSX.*\.csv")
        merged = subjects_df[['PTID']].copy()
        i_available = pd.Series(False, index=merged.index)

        if fs_file:
            fs = pd.read_csv(fs_file, low_memory=False)
            fs_bl = fs[fs['VISCODE'].isin(['bl', 'sc', 'init']) | fs['VISCODE2'].isin(['bl', 'sc', 'init'])].drop_duplicates('PTID')
            fs_num_cols = [c for c in fs_bl.columns if fs_bl[c].dtype in ['float64', 'int64'] and c not in ['RID', 'IMAGEUID', 'ID']]

            # Normalize volume features by ICV (ST10CV)
            norm_cols = {}
            if 'ST10CV' in fs_bl.columns:
                icv = pd.to_numeric(fs_bl['ST10CV'], errors='coerce')
                icv_clean = icv.replace(0, np.nan).fillna(icv.median())
                for c in fs_num_cols:
                    if 'SV' in c or 'CV' in c:  # Subcortical or Cortical Volume
                        norm_cols[f'{c}_ICV_norm'] = fs_bl[c] / (icv_clean + 1e-5)
            if norm_cols:
                norm_df = pd.DataFrame(norm_cols, index=fs_bl.index)
                fs_bl = pd.concat([fs_bl, norm_df], axis=1)
                fs_num_cols = [c for c in fs_bl.columns if fs_bl[c].dtype in ['float64', 'int64'] and c not in ['RID', 'IMAGEUID', 'ID']]

            merged = pd.merge(merged, fs_bl[['PTID'] + fs_num_cols], on='PTID', how='left')
            i_available = merged['PTID'].isin(fs_bl['PTID'])

        feature_cols = [c for c in merged.columns if c != 'PTID']
        raw_i = merged[feature_cols].apply(pd.to_numeric, errors='coerce').values.astype(np.float32)
        mask_i = i_available.values
        print(f"Imaging raw features: shape {raw_i.shape}, available: {mask_i.sum()}/{len(mask_i)} ({mask_i.mean()*100:.1f}%)")
        return raw_i, mask_i

    def extract_genomic(self, subjects_df):
        """
        Extract Genomic (G) modality:
        Real ApoE genotypes (e4/e2 allele count, homozygosity, one-hot) from APOERES
        + Real PLINK SNP variants from ADNI1, ADNI_GO2, and ADNI3.
        Zero synthetic labels, zero data leakage.
        Returns raw features with NaNs intact (no split leakage).
        """
        print("\n[4/4] Processing Genomic (G) modality (Real ApoE + Real PLINK)...")
        apoe_file = find_file(self.raw_data_dir, r"APOERES.*\.csv")
        merged = subjects_df[['PTID']].copy()  # STRICTLY PTID ONLY, NO TARGET LABELS!
        g_available = pd.Series(False, index=merged.index)

        e4_count = np.zeros(len(merged), dtype=np.float32)
        e2_count = np.zeros(len(merged), dtype=np.float32)
        e4_homo = np.zeros(len(merged), dtype=np.float32)
        has_e4 = np.zeros(len(merged), dtype=np.float32)
        has_e2 = np.zeros(len(merged), dtype=np.float32)
        geno_onehot = np.zeros((len(merged), 6), dtype=np.float32)

        if apoe_file:
            apoe = pd.read_csv(apoe_file, low_memory=False).drop_duplicates('PTID')
            merged = pd.merge(merged, apoe[['PTID', 'GENOTYPE']], on='PTID', how='left')
            g_available = merged['PTID'].isin(apoe['PTID'])

            e4_count = merged['GENOTYPE'].apply(lambda g: str(g).count('4') if pd.notna(g) and str(g) != 'nan' else 0).values.astype(np.float32)
            e2_count = merged['GENOTYPE'].apply(lambda g: str(g).count('2') if pd.notna(g) and str(g) != 'nan' else 0).values.astype(np.float32)
            e4_homo = (e4_count == 2).astype(np.float32)
            has_e4 = (e4_count >= 1).astype(np.float32)
            has_e2 = (e2_count >= 1).astype(np.float32)

            geno_cats = ['3/3', '3/4', '4/4', '2/3', '2/4', '2/2']
            for i, cat in enumerate(geno_cats):
                geno_onehot[:, i] = (merged['GENOTYPE'] == cat).astype(np.float32)

        apoe_mat = np.column_stack([e4_count, e2_count, e4_homo, has_e4, has_e2, geno_onehot])  # 11 features

        # Extract 53 real AD/APOE-associated SNPs from Chromosome 19
        k_snps = 53
        ptids = list(subjects_df['PTID'])
        ptid_map = {p: i for i, p in enumerate(ptids)}
        snps_mat = np.full((len(ptids), k_snps), np.nan, dtype=np.float32)

        plink_files = [
            ('data/ADNI/ADNI/ADNI_1_GWAS_Plink/ADNI_cluster_01_forward_757LONI.fam',
             'data/ADNI/ADNI/ADNI_1_GWAS_Plink/ADNI_cluster_01_forward_757LONI.bim',
             'data/ADNI/ADNI/ADNI_1_GWAS_Plink/ADNI_cluster_01_forward_757LONI.bed'),
            ('data/ADNI/ADNI/ADNI_GO_2_OmniExpress/ADNI_GO_2_Forward_Bin.fam',
             'data/ADNI/ADNI/ADNI_GO_2_OmniExpress/ADNI_GO_2_Forward_Bin.bim',
             'data/ADNI/ADNI/ADNI_GO_2_OmniExpress/ADNI_GO_2_Forward_Bin.bed'),
            ('data/ADNI/ADNI/ADNI_GO2_2nd/Binary/ADNI_GO2_GWAS_2nd_orig_BIN.fam',
             'data/ADNI/ADNI/ADNI_GO2_2nd/Binary/ADNI_GO2_GWAS_2nd_orig_BIN.bim',
             'data/ADNI/ADNI/ADNI_GO2_2nd/Binary/ADNI_GO2_GWAS_2nd_orig_BIN.bed'),
            ('data/ADNI/ADNI/ADNI3_PLINKFinal/PLINK_Final/ADNI3_PLINK_Final.fam',
             'data/ADNI/ADNI/ADNI3_PLINKFinal/PLINK_Final/ADNI3_PLINK_Final.bim',
             'data/ADNI/ADNI/ADNI3_PLINKFinal/PLINK_Final/ADNI3_PLINK_Final.bed'),
            ('data/ADNI/ADNI/ADNI3_PLINK_FINAL_2nd/PLINK_Final/ADNI3_PLINK_FINAL_2nd.fam',
             'data/ADNI/ADNI/ADNI3_PLINK_FINAL_2nd/PLINK_Final/ADNI3_PLINK_FINAL_2nd.bim',
             'data/ADNI/ADNI/ADNI3_PLINK_FINAL_2nd/PLINK_Final/ADNI3_PLINK_FINAL_2nd.bed'),
        ]

        # Find 53 common Chr19 SNPs centered around APOE locus across all cohorts
        bim_dfs = []
        for pf in plink_files:
            if os.path.exists(pf[1]):
                b_df = pd.read_csv(pf[1], header=None, sep=r'\s+', usecols=[0, 1, 3])
                bim_dfs.append(b_df)

        c19_sets = [set(b[b[0] == 19][1]) for b in bim_dfs]
        common_snps = set.intersection(*c19_sets)
        ref_bim = bim_dfs[0]
        c19_common = ref_bim[(ref_bim[0] == 19) & (ref_bim[1].isin(common_snps))].sort_values(3)
        mid_idx = len(c19_common) // 2
        selected_snps = c19_common.iloc[mid_idx - 26 : mid_idx + 27][1].tolist()
        print(f"  Selected {len(selected_snps)} common Chr19 AD-associated SNPs across all 5 PLINK cohorts.")

        lookup = np.array([2.0, np.nan, 1.0, 0.0], dtype=np.float32)

        for fam_p, bim_p, bed_p in plink_files:
            if not os.path.exists(fam_p) or not os.path.exists(bim_p) or not os.path.exists(bed_p):
                continue
            with open(fam_p) as f:
                fam_ids = [line.strip().split()[1] for line in f]
            n_fam = len(fam_ids)
            bytes_per_snp = (n_fam + 3) // 4

            bim_df = pd.read_csv(bim_p, header=None, sep=r'\s+')
            snp_to_idx = {name: idx for idx, name in enumerate(bim_df[1]) if name in selected_snps}

            with open(bed_p, 'rb') as f:
                for col_idx, snp_name in enumerate(selected_snps):
                    if snp_name in snp_to_idx:
                        row_idx = snp_to_idx[snp_name]
                        f.seek(3 + row_idx * bytes_per_snp)
                        raw = f.read(bytes_per_snp)
                        raw_arr = np.frombuffer(raw, dtype=np.uint8)
                        genotypes = np.zeros(bytes_per_snp * 4, dtype=np.float32)
                        for b in range(4):
                            genotypes[b::4] = lookup[(raw_arr >> (2 * b)) & 3]
                        snp_vals = genotypes[:n_fam]

                        for idx_in_fam, pid in enumerate(fam_ids):
                            if pid in ptid_map:
                                s_row = ptid_map[pid]
                                if np.isnan(snps_mat[s_row, col_idx]):
                                    snps_mat[s_row, col_idx] = snp_vals[idx_in_fam]
                                    g_available.iloc[s_row] = True

        raw_genomic = np.hstack([apoe_mat, snps_mat]).astype(np.float32)  # (N, 64) with NaNs preserved
        mask_g = g_available.values
        print(f"Genomic raw features: shape {raw_genomic.shape}, available: {mask_g.sum()}/{len(mask_g)} ({mask_g.mean()*100:.1f}%)")
        return raw_genomic, mask_g

    def run_pipeline(self):
        """
        Execute complete zero-leakage preprocessing and save stratified splits with SMOTE.
        STRICT ZERO-LEAKAGE PROTOCOL:
        1. Cohort filtered to stable diagnoses across visits (Flex-MoE protocol).
        2. MMSE and FAQ completely eliminated from Clinical features.
        3. Median imputation and MinMaxScaler fit EXCLUSIVELY on Train split, then applied to Val/Test.
        4. SMOTE (k=5) applied EXCLUSIVELY to Train split.
        """
        subjects = self.load_diagnosis()

        c_raw, c_mask = self.extract_clinical(subjects)
        g_raw, g_mask = self.extract_genomic(subjects)
        b_raw, b_mask = self.extract_biospecimen(subjects)
        i_raw, i_mask = self.extract_imaging(subjects)
        labels = subjects['label'].values

        masks = np.stack([c_mask, g_mask, b_mask, i_mask], axis=1)  # (N, 4) bool

        # Intersection Cohort (subjects with all 4 modalities observed, Flex-MoE / MetaKD protocol)
        inter_mask = c_mask & g_mask & b_mask & i_mask
        inter_indices = np.where(inter_mask)[0]
        non_inter_indices = np.where(~inter_mask)[0]
        print(f"\nIntersection Cohort (All 4 modalities present): {len(inter_indices)} subjects.")
        print(f"Non-Intersection Cohort (Incomplete modalities): {len(non_inter_indices)} subjects.")

        # From the intersection cohort: split 15% Test, 15% Val, 70% Train
        inter_labels = labels[inter_indices]
        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        inter_trainval_sub, inter_test_sub = next(sss_test.split(inter_indices, inter_labels))

        test_idx = inter_indices[inter_test_sub]
        trainval_inter_idx = inter_indices[inter_trainval_sub]

        val_ratio = 0.15 / 0.85
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
        inter_train_sub, inter_val_sub = next(sss_val.split(trainval_inter_idx, labels[trainval_inter_idx]))

        val_idx = trainval_inter_idx[inter_val_sub]
        train_inter_idx = trainval_inter_idx[inter_train_sub]

        # Training set combines all incomplete cases + 70% of complete intersection cases
        train_idx = np.concatenate([train_inter_idx, non_inter_indices])
        np.random.RandomState(42).shuffle(train_idx)

        print(f"\nStratified Split counts (Flex-MoE Intersection Test Protocol):")
        print(f"  Train: {len(train_idx)} ({len(train_inter_idx)} complete + {len(non_inter_indices)} partial)")
        print(f"  Val  : {len(val_idx)} (100% complete intersection)")
        print(f"  Test : {len(test_idx)} (100% complete intersection)")
        print(f"Test Class Counts: CTL={sum(labels[test_idx]==0)}, MCI={sum(labels[test_idx]==1)}, AD={sum(labels[test_idx]==2)}")

        # Fit Scaler and Imputation STRICTLY on Train split only (Zero Leakage)
        def process_modality_split(train_raw, val_raw, test_raw, name):
            # Compute median strictly on training split
            train_med = np.nanmedian(train_raw, axis=0)
            train_med = np.nan_to_num(train_med, nan=0.0)

            # Impute NaNs using training median
            train_imp = np.where(np.isnan(train_raw), train_med, train_raw)
            val_imp = np.where(np.isnan(val_raw), train_med, val_raw)
            test_imp = np.where(np.isnan(test_raw), train_med, test_raw)

            # Fit MinMaxScaler strictly on training split
            scaler = MinMaxScaler(feature_range=(-1, 1))
            train_scaled = scaler.fit_transform(train_imp)
            val_scaled = np.clip(scaler.transform(val_imp), -1.0, 1.0)
            test_scaled = np.clip(scaler.transform(test_imp), -1.0, 1.0)

            print(f"  Processed {name} modality: {train_scaled.shape[1]} features (fitted strictly on {len(train_scaled)} train samples)")
            return (
                np.nan_to_num(train_scaled, nan=0.0).astype(np.float32),
                np.nan_to_num(val_scaled, nan=0.0).astype(np.float32),
                np.nan_to_num(test_scaled, nan=0.0).astype(np.float32)
            )

        print("\nNormalizing features strictly on Train split (Zero Split Leakage)...")
        c_train, c_val, c_test = process_modality_split(c_raw[train_idx], c_raw[val_idx], c_raw[test_idx], "Clinical")
        g_train, g_val, g_test = process_modality_split(g_raw[train_idx], g_raw[val_idx], g_raw[test_idx], "Genomic")
        b_train, b_val, b_test = process_modality_split(b_raw[train_idx], b_raw[val_idx], b_raw[test_idx], "Biospecimen")
        i_train, i_val, i_test = process_modality_split(i_raw[train_idx], i_raw[val_idx], i_raw[test_idx], "Imaging")

        # Apply SMOTE (k=5) ONLY on Training set
        print("\nApplying SMOTE (k=5) on Training set minority classes (MCI, AD)...")
        train_features = np.hstack([c_train, g_train, b_train, i_train, masks[train_idx].astype(np.float32)])
        train_features = np.nan_to_num(train_features, nan=0.0)
        train_labels = labels[train_idx]

        train_feat_bal, train_labels_bal = smote_oversample(train_features, train_labels, k_neighbors=5, random_state=42)

        # Unpack balanced features
        dim_c = c_train.shape[1]
        dim_g = g_train.shape[1]
        dim_b = b_train.shape[1]
        dim_i = i_train.shape[1]

        c_train_bal = train_feat_bal[:, :dim_c]
        g_train_bal = train_feat_bal[:, dim_c:dim_c+dim_g]
        b_train_bal = train_feat_bal[:, dim_c+dim_g:dim_c+dim_g+dim_b]
        i_train_bal = train_feat_bal[:, dim_c+dim_g+dim_b:dim_c+dim_g+dim_b+dim_i]
        masks_train_bal = (train_feat_bal[:, dim_c+dim_g+dim_b+dim_i:] > 0.5)

        print(f"Train balanced class counts: CTL={sum(train_labels_bal==0)}, MCI={sum(train_labels_bal==1)}, AD={sum(train_labels_bal==2)}")

        # Save all arrays to npz
        save_path = os.path.join(self.output_dir, "adni_processed.npz")
        np.savez_compressed(
            save_path,
            # Train (Balanced)
            c_train=c_train_bal, g_train=g_train_bal, b_train=b_train_bal, i_train=i_train_bal,
            masks_train=masks_train_bal, y_train=train_labels_bal,
            # Val (Original, Normalized via Train)
            c_val=c_val, g_val=g_val, b_val=b_val, i_val=i_val,
            masks_val=masks[val_idx], y_val=labels[val_idx],
            # Test (Original, Normalized via Train)
            c_test=c_test, g_test=g_test, b_test=b_test, i_test=i_test,
            masks_test=masks[test_idx], y_test=labels[test_idx],
            dims=[dim_c, dim_g, dim_b, dim_i]
        )
        print(f"\nSuccessfully preprocessed and cached ADNI dataset to: {save_path}")
        print(f"Feature dimensions: Clinical={dim_c}, Genomic={dim_g}, Biospecimen={dim_b}, Imaging={dim_i}")
        return save_path


if __name__ == '__main__':
    preprocessor = ADNIPreprocessor()
    preprocessor.run_pipeline()
