# 📋 Checklist Triển Khai Pipeline ADNI 100% Theo Paper Flex-MoE & MetaKD Trên Kaggle

> **Mục tiêu:** Tái lập chính xác và nguyên bản 100% thực nghiệm phân loại giai đoạn Alzheimer (ADNI) với đầy đủ 4 phương thức (Clinical, Genomic, Biospecimen, Image) theo đúng giao thức của bài báo **Flex-MoE (NeurIPS 2024 / UNITES-Lab)** và bảng đối chiếu **Table 2 của Paper MetaKD (arXiv:2405.07155v4)**.

---

## 📌 Tổng Quan Kiến Trúc & Dữ Liệu Chuẩn

| Phương thức | Nguồn dữ liệu gốc | Cách tiền xử lý chuẩn Paper | Kiến trúc Encoder |
| :--- | :--- | :--- | :--- |
| **Clinical (C)** | `ADNIMERGE2` (5 bảng: `MEDHIST`, `NEUROEXM`, `PTDEMOG`, `RECCMEDS`, `VITALS`) | Loại bỏ cột rò rỉ chẩn đoán (`PTCOGBEG`, `PTADDX`, `PTADBEG`), MinMax `[-1, 1]`, One-hot biến phân loại | **MLP Encoder** ($d = 128$) |
| **Genomic (G)** | 5 cohort PLINK thô (`ADNI1`, `ADNI_GO2_1`, `ADNI_GO2_2`, `ADNI3_1`, `ADNI3_2`) gồm 567,386 SNPs | Lọc Allele chuẩn $\rightarrow$ `liftOver` đồng bộ tọa độ sang `hg38` $\rightarrow$ PLINK LD Pruning `50 5 0.1` thu được ~20,000 – 30,000 SNPs | **1D ResNet-34** ($d = 128$) |
| **Biospecimen (B)** | `UPENNBIOMK_ROCHE_ELECSYS` + `APOERES` | Các chỉ số dịch não tủy chuẩn ($A\beta40, A\beta42, \text{Tau}, \text{pTau}$) + kiểu gen $APOE$ | **MLP Encoder** ($d = 128$) |
| **Imaging (I)** | `UCSFFSX7` (FreeSurfer 7.x) | 348 đặc trưng thể tích chất xám/não thất, chuẩn hóa theo thể tích nội sọ (`ST10CV`) | **MLP Encoder** ($d = 128$) |

---

## 🚀 Checklist Chi Tiết Từng Giai Đoạn

### Phase 1: Chuẩn bị Dữ liệu Local & Upload lên Kaggle
- [x] **1.1. Gom các file dữ liệu ADNI cần thiết vào 1 thư mục staging:**
  * **Clinical:** Thư mục `ADNIMERGE2` (gồm 5 file `.rda`: `MEDHIST`, `NEUROEXM`, `PTDEMOG`, `RECCMEDS`, `VITALS`) + nhãn chẩn đoán `DXSUM_PDXCONV_*.csv`.
  * **Biospecimen:** `UPENNBIOMK_ROCHE_ELECSYS.rda` và `APOERES.csv`.
  * **Imaging:** `UCSFFSX7_*.csv` (FreeSurfer 7.x).
  * **Genomic:** Cả 5 cohort PLINK (`.bed`, `.bim`, `.fam`):
    1. `ADNI_1_GWAS_Plink` (757 ca)
    2. `ADNI_GO_2_OmniExpress` (432 ca)
    3. `ADNI_GO2_2nd` (361 ca)
    4. `ADNI3_PLINKFinal` (327 ca)
    5. `ADNI3_PLINK_FINAL_2nd` (328 ca)
- [x] **1.2. Nén thành file zip:** `adni_raw_flexmoe.zip` (~213.7 MB) đã hoàn tất tại `d:\MetaKD\adni_raw_flexmoe.zip`.
- [ ] **1.3. Tạo Kaggle Dataset:**
  * Dataset slug: `salala1706/adni-raw-flexmoe`.
  * ⚠️ **BẮT BUỘC:** Chế độ **`Private`** (bảo mật nghiêm ngặt theo ADNI Data Use Agreement).

---

### Phase 2: Cấu hình Kaggle Notebook
- [ ] **2.1. Tạo Notebook mới:** Đặt tên ví dụ `metakd-adni-flexmoe`.
- [ ] **2.2. Gắn Dataset vào Notebook:**
  * Bấm `Add Input` $\rightarrow$ Thêm `adni-raw-flexmoe`.
- [ ] **2.3. Cài đặt môi trường:**
  * **Accelerator:** Chọn **`GPU T4 x1`** (hoặc `GPU P100`).
  * **Internet:** Bật **`ON`** (bắt buộc để tải `liftOver`, `plink` và các file chain).

---

### Phase 3: Cài đặt công cụ Tin sinh học (Cell 1 Notebook)
- [ ] **3.1. Cài đặt Python packages:**
  ```python
  !pip install pyreadr pandas-plink anndata gdown -q
  ```
- [ ] **3.2. Tải UCSC `liftOver` & PLINK 1.9 (Linux 64-bit):**
  ```bash
  %%bash
  mkdir -p /kaggle/working/tools && cd /kaggle/working/tools
  wget -q http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver && chmod +x liftOver
  wget -q https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_latest.zip && unzip -q plink_linux_x86_64_latest.zip
  wget -q http://hgdownload.soe.ucsc.edu/goldenPath/hg18/liftOver/hg18ToHg38.over.chain.gz
  wget -q http://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz
  ```

---

### Phase 4: Xử lý Genomic đúng 100% repo Flex-MoE (`genomic_preprocess.ipynb`)
- [ ] **4.1. Lọc Allele chuẩn (A, C, G, T) trong 5 file `.bim`:**
  * Lọc sạch biến thể không chuẩn trên 5 file `.bim`.
- [ ] **4.2. Chuẩn bị định dạng BED cho Liftover:**
  * Tạo file ánh xạ tọa độ gồm 4 cột: `chr{id}`, `pos`, `pos`, `snp_id`.
- [ ] **4.3. Chạy `liftOver` sang chuẩn `hg38`:**
  * ADNI1 dùng chain `hg18ToHg38`.
  * ADNI GO/2 và ADNI3 dùng chain `hg19ToHg38`.
- [ ] **4.4. Hợp nhất 5 cohort:**
  * Tìm tập SNPs giao nhau trên `hg38` chung cho cả 5 cohort.
- [ ] **4.5. Chạy PLINK LD Pruning:**
  ```bash
  ./tools/plink --bfile merged_adni --indep-pairwise 50 5 0.1 --out pruned_snps
  ./tools/plink --bfile merged_adni --extract pruned_snps.prune.in --make-bed --out final_genomic
  ```
  * Thu được vector SNP độc lập trên toàn bộ hệ gen (~20,000 – 30,000 SNPs).

---

### Phase 5: Xử lý Clinical, Biospecimen & Image (Zero-leakage Split)
- [ ] **5.1. Clinical (C):** Ghép 5 bảng `ADNIMERGE2`, loại bỏ `PTCOGBEG`, `PTADDX`, `PTADBEG`, chuẩn hóa MinMax `[-1, 1]`.
- [ ] **5.2. Biospecimen (B):** Nạp `UPENNBIOMK_ROCHE_ELECSYS` ($A\beta40, A\beta42, \text{Tau}, \text{pTau}$) + `APOERES`.
- [ ] **5.3. Imaging (I):** Nạp `UCSFFSX7` (FreeSurfer 7.x) và chuẩn hóa ROI theo Intracranial Volume (`ST10CV`).
- [ ] **5.4. Chia tập dữ liệu chuẩn Paper:**
  * 70% Train, 15% Val, 15% Test cố định (hoặc 5-fold cross validation).
  * Scaler chỉ fit trên tập Train (Zero data leakage).

---

### Phase 6: Xây dựng Kiến trúc MetaKD + 1D ResNet-34 cho Modality G (Đã tạo trong `adni_flexmoe/`)
- [x] **6.1. Xây dựng 1D ResNet-34 Backbone:**
  * File: [adni_flexmoe/models/resnet34_1d.py](file:///d:/MetaKD/adni_flexmoe/models/resnet34_1d.py).
  * Đầu vào: tensor `[Batch, 1, num_snps]` (~30k chiều).
  * 4 residual stages với 1D Conv, BatchNorm, ReLU, AdaptiveAvgPool1d, Linear $\rightarrow$ $d = 128$. Đã kiểm thử thành công.
- [x] **6.2. Encoders cho C, B, I:** MLP Encoders đưa về latent dimension $d = 128$.
- [x] **6.3. MetaKD Bi-level Optimization:**
  * File: [adni_flexmoe/models/metakd_model.py](file:///d:/MetaKD/adni_flexmoe/models/metakd_model.py).
  * Vector trọng số meta-learning $w = [w_c, w_g, w_b, w_i]^\top$.
  * Pairwise Knowledge Distillation Loss $\mathcal{L}_{kd} = \sum_{j \neq i} \frac{w_i}{w_j} \mathcal{D}_{KL}(p_i, p_j)$.
  * Modality dropout ngẫu nhiên để tăng khả năng chống chịu dữ liệu thiếu sót.

---

### Phase 7: Huấn luyện & Benchmark Toàn Diện 15 Tổ Hợp Modality (Đã tạo trong `adni_flexmoe/`)
- [x] **7.1. Huấn luyện mô hình:** Script [adni_flexmoe/train.py](file:///d:/MetaKD/adni_flexmoe/train.py) với bi-level optimization.
- [x] **7.2. Benchmark kiểm thử trên 15 tổ hợp modality:** Script [adni_flexmoe/eval.py](file:///d:/MetaKD/adni_flexmoe/eval.py) đối chiếu trực tiếp Table 2 Paper MetaKD.
- [x] **7.3. Notebook Kaggle trọn gói:** File [adni_flexmoe/notebooks/kaggle_adni_full_pipeline.ipynb](file:///d:/MetaKD/adni_flexmoe/notebooks/kaggle_adni_full_pipeline.ipynb).
