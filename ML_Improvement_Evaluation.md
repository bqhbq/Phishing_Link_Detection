# Báo cáo Đánh giá Kết quả Cải tiến: Phát hiện Phishing URL bằng Machine Learning truyền thống

> **Đối tượng đánh giá:** [notebooks/06_Feature_Engineering_and_Advanced_ML.ipynb](notebooks/06_Feature_Engineering_and_Advanced_ML.ipynb)
> **Baseline đối chứng:** pipeline của [notebooks/05_Model_Comparison_and_Tuning.ipynb](notebooks/05_Model_Comparison_and_Tuning.ipynb) (20 đặc trưng, Random Forest / XGBoost tuned), được **tái lập trên cùng tập dữ liệu và cùng tập test** với mô hình mới.
> **Tài liệu liên quan:** [ML_Pipeline_Analysis.md](ML_Pipeline_Analysis.md), bản phân tích gốc đã xác định các vấn đề mà notebook 06 giải quyết.
> **Phạm vi phương pháp:** chỉ dùng **Machine Learning truyền thống**: Logistic Regression, Random Forest, Gradient Boosting (HistGradientBoosting, XGBoost), Stacking, Soft Voting.
> **Nguồn số liệu:** mọi con số trong báo cáo được trích **trực tiếp từ output của notebook 06** (`RANDOM_STATE = 42`). Notebook đã chạy hai lần độc lập và cho kết quả trùng khớp. Số liệu nào được tính thêm để phục vụ báo cáo đều được ghi chú rõ.

---

## Tóm tắt điều hành

| | Baseline tốt nhất (XGBoost tuned, NB05) | **Mô hình tốt nhất (Stacking TF-IDF LR → XGBoost)** | Thay đổi |
|---|:-:|:-:|:-:|
| F1-Score | 0.7747 | **0.9284** | **+15.4 điểm (+19.8%)** |
| Recall (tỉ lệ bắt được phishing) | 0.7034 | **0.9028** | +19.9 điểm (+28.3%) |
| Precision (độ chính xác của cảnh báo) | 0.8621 | **0.9555** | +9.3 điểm (+10.8%) |
| ROC-AUC | 0.9479 | **0.9939** | +4.6 điểm (phần diện tích còn thiếu giảm 88%) |
| Phishing bị bỏ sót (FN) | 6,447 | **2,112** | **−67.2%** |
| URL sạch bị chặn nhầm (FP) | 2,445 | **914** | **−62.6%** |

**Kết luận chính:**
- Với cùng tập test gồm 100,307 URL và **không có domain nào trùng với tập train**, mô hình Stacking cải thiện **đồng thời** cả hai loại lỗi: số phishing bị bỏ sót giảm 3 lần, số cảnh báo sai giảm 2.7 lần.
- Đòn bẩy quyết định là **biểu diễn đặc trưng** (TF-IDF char n-gram kết hợp đặc trưng thủ công có cơ sở bảo mật), không phải độ phức tạp của thuật toán. Điều này khẳng định hướng tiếp cận ML truyền thống là đủ mạnh cho bài toán.

---

## 1. Tổng quan các cải tiến đã thực hiện

### 1.1. Cải tiến về dữ liệu và giao thức đánh giá

Đây là nền móng để kết quả đánh giá **đáng tin cậy**, trước khi bàn tới đặc trưng hay mô hình:

| Cải tiến | Chi tiết | Kết quả |
|---|---|---|
| **Chuẩn hóa URL** (`normalize_url`) | Bỏ scheme, bỏ `www.`, viết thường host, bỏ `/` cuối. Áp dụng **giống hệt** cho train và inference | Loại shortcut "không có `/` ⇒ phishing" (tỉ lệ phishing của URL không có `/` trong dữ liệu thô là 98.6%, sau chuẩn hóa domain trần chỉ còn 16.6%, gần tỉ lệ nền 21.7%)¹ |
| **Khử trùng lặp sau chuẩn hóa** | Loại các bản gần trùng và 2 URL chuẩn hóa mâu thuẫn nhãn (6 dòng) | 501,550 URL (ít hơn notebook 01 là 5,645 dòng) |
| **Chia dữ liệu theo domain** | `StratifiedGroupKFold` theo *registered domain*: Train 70% / Validation 10% / Test 20% | 106,056 / 15,156 / 30,323 domain, **không giao nhau**. Tỉ lệ phishing 21.67% ở cả ba tập |
| **Chọn ngưỡng trên validation** | Ngưỡng tối ưu F1 được chọn trên val, sau đó áp nguyên lên test | Tránh selection bias trên tập test |

¹ *Số liệu tính bổ sung khi viết báo cáo, nhằm kiểm chứng hiệu quả của bước chuẩn hóa.*

### 1.2. Đặc trưng mới (60 đặc trưng số + 1 đặc trưng phân loại)

Bộ 20 đặc trưng cũ được thay bằng **61 đặc trưng** chia thành 8 nhóm. Mỗi nhóm nhắm tới một kỹ thuật tấn công cụ thể:

| Nhóm | Đặc trưng | Kỹ thuật tấn công mục tiêu |
|---|---|---|
| Cấu trúc URL | `url_len`, `path_len`, `query_len`, `path_depth`, `n_params`, `n_dots`, `n_hyphens`, `n_slashes`, `n_at`, `n_tilde`, `n_percent`, `n_equals`, `n_underscores` | Hình thái tổng quát |
| Thành phần ký tự | `digit_ratio`, `letter_ratio`, `special_ratio`, `upper_ratio`, `n_tokens`, `longest_token`, `avg_token_len` | Chuỗi hash, payload mã hóa |
| Tính ngẫu nhiên | `host_entropy`, `sld_entropy`, `url_entropy`, `path_entropy`, `sld_vowel_ratio`, `sld_max_consonant_run`, `sld_digit_ratio` | Domain sinh tự động (DGA) |
| Thương hiệu | `brand_out_of_domain`, `brand_in_subdomain`, `brand_in_path`, `brand_combo_sld`, `sld_is_brand` | Brand impersonation, combosquatting |
| Typosquatting | `typosquat_dist`, `is_typosquat` (Levenshtein tới 39 thương hiệu) | `paypa1.com`, `gooogle.com` |
| Từ khóa theo vị trí | `sens_in_host`, `sens_in_path` (31 từ khóa, so khớp theo **ranh giới từ**) | Trang đăng nhập giả |
| Né tránh và payload | `host_is_ip` (IPv4/IPv6/hex/số nguyên), `host_has_punycode`, `host_non_ascii`, `has_userinfo`, `has_port`, `is_shortener`, `double_slash_in_path`, `has_embedded_url`, `has_redirect_param`, `ext_executable/script/html/archive/document` | Che giấu đích đến, open redirect, phát tán mã độc |
| TLD | `tld` → **Target Encoding** (cross-fitted, 811 TLD), `tld_len` | TLD có mức rủi ro khác nhau |

Ngoài ra còn có **biểu diễn văn bản thưa**: TF-IDF **char n-gram (3–5)** với 300,000 n-gram, kết hợp TF-IDF **token** (51,710 token). Ma trận train có kích thước 351,090 × 351,710 với 44.3 triệu phần tử khác 0.

### 1.3. Thuật toán đã huấn luyện

| # | Mô hình | Đầu vào | Kỹ thuật chính |
|:-:|---|---|---|
| B1 | Baseline RF (NB05) | 20 đặc trưng cũ | Tái lập nguyên bản, ngưỡng 0.5 |
| B2 | Baseline XGBoost tuned (NB05) | 20 đặc trưng cũ | Tham số tốt nhất của NB05, ngưỡng 0.5 |
| M1 | **Logistic Regression + TF-IDF** | TF-IDF char/token | `liblinear`, `class_weight='balanced'`, chọn `C ∈ {0.5, 2, 8}` theo val PR-AUC → **C = 8** |
| M2 | **HistGradientBoosting** | 61 đặc trưng thủ công | Thuật toán histogram-based GBDT (**cùng họ với LightGBM**, có sẵn trong scikit-learn), `class_weight='balanced'`, early stopping (853 vòng) |
| M3 | **XGBoost + Optuna** | 61 đặc trưng thủ công | Optuna TPE 20 trials (1.6 phút), tối ưu **val PR-AUC**, `scale_pos_weight = 3.615`, early stopping (73 cây) |
| M4 | **Stacking** | Đặc trưng thủ công + `lr_tfidf_score` | Level-0: điểm LR **out-of-fold** (5-fold `StratifiedGroupKFold` theo domain). Level-1: XGBoost meta (227 cây) |
| M5 | **Soft Voting** | Xác suất M1 và M3 | `p = 0.7·p_LR + 0.3·p_XGB` (trọng số chọn trên val) |

> **Ghi chú về LightGBM/CatBoost:** môi trường thực thi chưa cài hai thư viện này. Thay vì cài thêm gói, notebook dùng `HistGradientBoostingClassifier` của scikit-learn. Đây là hiện thực của cùng thuật toán GBDT dựa trên histogram và phát triển cây theo lá (leaf-wise) mà LightGBM sử dụng. CatBoost **chưa** được huấn luyện (xem §5.3).

---

## 2. Phân tích trực quan (Visual Insights)

### 2.1. Phân phối đặc trưng liên tục (§8, biểu đồ 4×4)

Biểu đồ mật độ hai lớp trên 100,000 mẫu train cho thấy:

| Đặc trưng | Quan sát | Diễn giải |
|---|---|---|
| `url_len` | Hai lớp chồng lấn mạnh trong khoảng 20–80 ký tự. Phishing có **đuôi dài** hơn hẳn trong khoảng 100–250 | Độ dài chỉ phân biệt tốt ở vùng cực trị (URL nhồi tham số, subdomain giả) |
| `longest_token` | Phishing có **đỉnh nhọn tại ~32 ký tự** và đuôi tới 65 | 32 ký tự hex là độ dài của **mã băm MD5**: phishing kit hay sinh thư mục ngẫu nhiên kiểu `/a9f610dbd488353106e3e9d02d9c8c8f/` |
| `path_entropy`, `url_entropy` | Phishing dịch sang phải, có mật độ cao hơn ở vùng 4.5–5.0 bit | Path chứa chuỗi ngẫu nhiên hoặc đã mã hóa |
| `special_ratio` | Phishing có phân phối **hai đỉnh**: một đỉnh ở 0.08–0.10 (URL ngắn ít ký tự đặc biệt) và một đuôi > 0.20 (URL nhồi `=`, `&`, `%`) | Đây là một tín hiệu phi tuyến, nên mô hình cây khai thác tốt hơn mô hình tuyến tính |
| `sld_len`, `sld_entropy`, `sld_vowel_ratio` | Phishing có **cột dồn tại giá trị 0** | Tương ứng các host là địa chỉ IP (không có SLD) |
| `path_depth` | Phishing chiếm ưu thế ở độ sâu ≥ 4 | Trang giả thường nằm sâu trong cây thư mục của site bị chiếm quyền (`/wp-content/themes/...`) |
| `subdomain_levels`, `host_entropy` | Chồng lấn gần hoàn toàn | Đơn lẻ thì phân biệt kém, chỉ có giá trị khi kết hợp với đặc trưng khác |

**Nhận định:** đa số đặc trưng liên tục chỉ **phân tách yếu khi xét đơn lẻ**, vì hai phân phối chồng lấn lớn. Đây là đặc điểm bình thường của URL phishing hiện đại: kẻ tấn công cố tình bắt chước hình thái URL hợp lệ. Sức mạnh phân loại đến từ **sự kết hợp phi tuyến** của nhiều tín hiệu yếu (nhiệm vụ của boosting) và từ các **cờ nhị phân có độ đặc hiệu cao** dưới đây.

### 2.2. Đặc trưng nhị phân: tỉ lệ phishing khi cờ bật (§8)

Tỉ lệ phishing nền trên tập train là **21.67%**. Các cờ mới có độ đặc hiệu rất cao:

| Cờ | Số URL có cờ bật (train) | P(Phishing \| cờ = 1) | Mức tăng so với nền |
|---|---:|:-:|:-:|
| `ext_executable` (`.exe`, `.apk`, `.scr`…) | 1,732 | **0.996** | ×4.6 |
| `host_is_ip` | 2,454 | **0.994** | ×4.6 |
| `double_slash_in_path` | 495 | 0.984 | ×4.5 |
| `is_shortener` | 682 | 0.977 | ×4.5 |
| `brand_combo_sld` (`paypal-secure.com`) | 2,246 | 0.969 | ×4.5 |
| `has_embedded_url` | 3,291 | 0.966 | ×4.5 |
| `brand_in_path` | 8,227 | 0.931 | ×4.3 |
| `ext_archive` | 357 | 0.924 | ×4.3 |
| **`brand_out_of_domain`** | **11,582** | **0.911** | ×4.2 |
| `has_port` | 1,134 | 0.890 | ×4.1 |
| `sld_is_brand` (chính chủ, ví dụ `paypal.com`) | 29,531 | **0.021** | ×0.1 (tín hiệu **an toàn**) |

Cặp `brand_out_of_domain` (91.1% phishing) và `sld_is_brand` (2.1% phishing) thể hiện đúng nguyên lý phát hiện **brand impersonation**: *tên thương hiệu nằm ở đâu trong URL* quan trọng hơn *URL có chứa tên thương hiệu hay không*. Đây chính là điểm yếu của đặc trưng `count_sensitive_words` cũ, vốn đếm `paypal` như nhau dù nó nằm trong domain chính chủ hay trong path giả mạo.

Hai cờ còn hạn chế là `is_typosquat` (180 mẫu, 30.6% phishing) và `has_userinfo` (18 mẫu). Chúng hiếm trong dataset nên đóng góp thống kê nhỏ, nhưng vẫn có giá trị phòng thủ trước các kỹ thuật mà dataset hiện tại chưa phản ánh đầy đủ.

### 2.3. Tương quan đơn biến với nhãn (§9, Spearman ρ)

Top đặc trưng tương quan với nhãn phishing: `sens_in_path` (ρ ≈ 0.43), `brand_out_of_domain` (≈ 0.31), `host_n_digits` (≈ 0.29), `brand_in_path` (≈ 0.27), `n_dots` (≈ 0.26), `ext_script` (≈ 0.21). Hai đặc trưng duy nhất có tương quan âm trong top 25 là `sld_is_brand` (≈ −0.14) và `tld_len` (≈ −0.13; TLD dài như `.co.uk` hay `.org.au` thường thuộc site hợp lệ). **Không đặc trưng nào có |ρ| > 0.45**, xác nhận nhận định ở §2.1: không có "đặc trưng vàng" đơn lẻ nào.

### 2.4. Đa cộng tuyến (Multicollinearity) (§9, heatmap 60×60)

Có hiện tượng đa cộng tuyến, nhưng **cục bộ và có thể giải thích được**. 9 cặp có |ρ| ≥ 0.9:

| Cặp đặc trưng | ρ | Nguyên nhân |
|---|:-:|---|
| `n_params` – `query_len` | 0.998 | Cùng đo kích thước query string |
| `subdomain_len` – `subdomain_levels` | 0.988 | Cùng đo subdomain |
| `typosquat_dist` – `sld_is_brand` | −0.969 | Khoảng cách = 0 khi và chỉ khi SLD là brand |
| `n_slashes` – `path_depth` | 0.964 | Mỗi `/` là một cấp thư mục |
| `n_equals` – `n_params` / `query_len` | 0.955 / 0.951 | Mỗi tham số có một dấu `=` |
| `n_tokens` – `url_len` | 0.918 | URL dài thì có nhiều token |
| `avg_token_len` – `special_ratio` | −0.905 | Nhiều ký tự phân tách thì token ngắn hơn |
| `path_entropy` – `url_entropy` | 0.901 | Path chiếm phần lớn URL |

Heatmap còn cho thấy các khối tương quan cao (0.7–0.9) giữa nhóm độ dài (`url_len`, `path_len`, `n_tokens`, `url_entropy`) và nhóm host (`host_len`, `host_entropy`, `sld_len`, `sld_entropy`). Ngoài các khối này, phần lớn ma trận gần 0, nghĩa là các nhóm đặc trưng mới (brand, payload, né tránh) mang **thông tin độc lập** với nhóm cấu trúc.

**Tác động:**
- **Hiệu năng của mô hình cây không bị ảnh hưởng.** Ở mỗi lần chia nút, cây chọn một trong các biến đồng tuyến.
- **Feature importance bị "chia sẻ"** giữa các biến đồng tuyến. Ví dụ, mức quan trọng của khái niệm "độ dài path" bị chia cho `n_slashes`, `path_depth` và `path_len`. Vì vậy ở §3, các đặc trưng được đọc theo **nhóm khái niệm**.
- Với Logistic Regression trên TF-IDF, các n-gram lồng nhau cũng đồng tuyến. Hệ quả là hệ số đơn lẻ có thể mang dấu "kỳ lạ" (xem §3.4).

---

## 3. Phân tích đặc trưng quan trọng (Feature Importance)

Notebook cung cấp **bốn góc nhìn** bổ sung cho nhau. Mỗi phương pháp có thiên lệch riêng, nên kết luận chỉ dựa trên những gì **nhất quán giữa các phương pháp**:

- **Permutation importance** (HistGradientBoosting, trên val): đo mức giảm PR-AUC khi xáo trộn đặc trưng. Đây là cách đo trung thực nhất về đóng góp *thực tế*.
- **XGBoost average gain:** mức cải thiện trung bình *mỗi lần* đặc trưng được dùng để chia. Cách đo này ưu tiên các cờ hiếm nhưng tạo ra những lần chia rất "sạch".
- **XGBoost total gain** (tính bổ sung từ mô hình đã lưu): tổng đóng góp trên toàn bộ các cây.
- **Stacking meta-model gain:** cho biết vai trò của điểm TF-IDF so với đặc trưng thủ công.

### 3.1. Bảng xếp hạng tổng hợp

| Hạng | Đặc trưng | Permutation (HGB) | XGB avg gain | XGB total gain² | Stacking avg gain |
|:-:|---|:-:|:-:|:-:|:-:|
| 1 | **`sens_in_path`** | #2 (ΔPR-AUC ≈ 0.065) | **#1** (≈ 1,830) | #2 (9.7%) | #3 |
| 2 | **`tld_target_enc`** | **#1** (ΔPR-AUC ≈ 0.117) | #4 | **#1 (20.2%)** | #8 |
| 3 | **`brand_out_of_domain`** | #8 | #2 (≈ 630) | — | #2 |
| 4 | **`sld_is_brand`** / `typosquat_dist` | — | #6 / #9 | — | **#1** / #5 |
| 5 | **`path_entropy` / `url_entropy`** | #3 / #5 | — | #7 (`url_entropy`) | — |
| 6 | **`n_dots`** | #4 | #10 | #6 | #7 |
| 7 | **`brand_in_path`** | — | #3 (≈ 430) | — | #10 |
| 8 | **`ext_executable`** | #17 | #5 | — | #6 |
| 9 | **`longest_token`** | #13 | #14 | #3 (5.5%) | — |
| 10 | **`host_n_digits`** | #16 | #7 | — | #11 |

² *Số liệu tính bổ sung khi viết báo cáo, không in trong notebook: tỉ trọng total gain (%) tính từ mô hình đã lưu `models/phishing_url_models_v2.joblib`; tỉ lệ phishing theo mẫu URL/token (§3.4) tính trực tiếp trên dữ liệu thô. Dấu "—" nghĩa là không nằm trong top của biểu đồ tương ứng.*

### 3.2. Giải thích logic bảo mật của Top 10

1. **`sens_in_path`** (số từ khóa nhạy cảm trong path/query: `login`, `verify`, `webscr`, `account`, `billing`…). Trang phishing phải **thuyết phục nạn nhân nhập thông tin xác thực**, nên đường dẫn thường mô phỏng luồng đăng nhập hoặc xác minh. Điểm mấu chốt là từ khóa nằm ở **path**, tức trên hạ tầng của kẻ tấn công. Khi nằm ở **host**, chúng thường thuộc domain hợp lệ (ví dụ `login.microsoftonline.com`). Vì vậy `sens_in_host` chỉ xếp hạng thấp (#21 theo XGB average gain).
2. **`tld_target_enc`** (mức rủi ro lịch sử của TLD). Kẻ tấn công ưu tiên TLD **rẻ, đăng ký dễ, ít kiểm duyệt**. Trong tập train, `.xyz` và `.top` có tỉ lệ phishing 100%, `.cn` 99.2%, `.tk` 97.5%, `.com.br` 97.8%. Target Encoding nén 811 TLD thành một điểm rủi ro liên tục, nên mô hình cây chỉ cần một lần chia thay vì hàng trăm.
3. **`brand_out_of_domain`** (thương hiệu xuất hiện ở subdomain/path nhưng **không** thuộc registered domain). Đây là chữ ký của **brand impersonation**, ví dụ `paypal.com.secure-login.verify-account.xyz`: người dùng đọc thấy "paypal.com" ở đầu URL, nhưng domain thực là `verify-account.xyz`.
4. **`sld_is_brand` / `typosquat_dist`** (SLD trùng chính xác, hoặc gần giống, tên thương hiệu). Đây là **tín hiệu an toàn mạnh nhất** (2.1% phishing), vì domain chính chủ của các thương hiệu lớn hầu như không bị dùng để lừa đảo. Ngược lại, khoảng cách Levenshtein 1–2 (`paypa1`) là dấu hiệu typosquatting. Trong meta-model, `sld_is_brand` xếp #1 theo average gain. Đặc trưng này giúp meta-model **sửa sai cho TF-IDF**: LR chấm `accounts.google.com/signin` ở mức 0.999, còn Stacking hạ xuống 0.142.
5. **`path_entropy` / `url_entropy`** (độ ngẫu nhiên của chuỗi). Phishing kit sinh thư mục ngẫu nhiên để **né blacklist theo URL** (mỗi nạn nhân nhận một URL khác nhau) và thường nhúng token hoặc payload đã mã hóa. Hai đặc trưng này xếp #3 và #5 theo permutation importance.
6. **`n_dots`** (số dấu chấm). Nhiều dấu chấm phản ánh **chuỗi subdomain giả** (`paypal.us.cgi.bin.webscr...lindenleafnannies.com`) hoặc nhiều phần mở rộng file lồng nhau. Đây là đặc trưng ổn định nhất: lọt top 10 ở cả bốn góc nhìn.
7. **`brand_in_path`** (thương hiệu xuất hiện trong path, ví dụ `/paypal.co.uk/cycgi-bin/...`). Đây là kỹ thuật dựng trang giả **trên site hợp lệ bị chiếm quyền**, nơi kẻ tấn công không kiểm soát được domain mà chỉ kiểm soát được thư mục.
8. **`ext_executable`** (`.exe`, `.apk`, `.scr`…). Có 99.6% là độc hại. URL trỏ thẳng tới file thực thi là kênh **phát tán malware** điển hình (dropper), và gần như không xuất hiện trong lưu lượng duyệt web hợp lệ.
9. **`longest_token`** (độ dài token dài nhất). Đặc trưng này nắm bắt **mã băm MD5/SHA**, ID phiên và chuỗi base64 trong path (xem đỉnh 32 ký tự ở §2.1). Xếp #3 theo total gain: được dùng rất thường xuyên dù mỗi lần chia chỉ đóng góp vừa phải.
10. **`host_n_digits`** (số chữ số trong host). Host chứa nhiều chữ số gợi ý **IP, domain sinh tự động hoặc hạ tầng hosting giá rẻ** (`tr7.datahost84.org`), thay vì một thương hiệu đầu tư tên miền có ý nghĩa.

**Về câu hỏi "tại sao độ dài URL lại quan trọng":** kết quả cho thấy `url_len` **không** lọt top 10 ở bất kỳ góc nhìn nào. Độ dài thuần túy là tín hiệu *gián tiếp*: URL phishing dài vì chúng chứa subdomain giả, hash hoặc tham số nhồi nhét. Khi mô hình đã có các đặc trưng đo *trực tiếp* những nguyên nhân đó (`n_dots`, `longest_token`, `path_entropy`, `brand_out_of_domain`), độ dài trở nên dư thừa. Đây là bằng chứng bộ đặc trưng mới đã chuyển từ đo *triệu chứng* sang đo *cơ chế tấn công*.

### 3.3. Vai trò của TF-IDF trong Stacking

Theo **average gain**, `lr_tfidf_score` chỉ xếp **#4** trong meta-model. Tuy nhiên theo **total gain** (tính bổ sung), nó chiếm **61.1%** tổng đóng góp, cao gấp 8.7 lần đặc trưng đứng thứ hai (`tld_target_enc`, 7.0%). Cách đọc đúng như sau:
- Điểm TF-IDF là **xương sống**, được dùng ở hầu hết các nhánh cây.
- Các đặc trưng thủ công (`sld_is_brand`, `brand_out_of_domain`, `typosquat_dist`) đóng vai trò **bộ hiệu chỉnh** cho những trường hợp đặc thù. Chúng được dùng ít lần, nhưng mỗi lần tạo ra cải thiện lớn (average gain cao). Đây chính là các ca TF-IDF dễ sai, ví dụ domain chính chủ có chứa `signin`.

### 3.4. Diễn giải hệ số Logistic Regression TF-IDF (§12)

- **N-gram đẩy về phishing:** `tok:exe`, `tok:js`, `tok:paypal`, `tok:webscr`, `ogin` (từ `login`), `pay`, `tok:includes`, `tok:libraries`, `/face`. Các token `includes` và `libraries` là thư mục đặc trưng của CMS Joomla/WordPress **bị chiếm quyền** (99.6% và 91.8% phishing trong dữ liệu²).
- **N-gram đẩy về hợp lệ:** `htm`, `.htm` (trang tĩnh cổ điển), `.org`, `.ca`, `tok:imdb`, `tok:myspace`, `tok:people`, `id=`.
- **Cặp `.com/` (+14.8) và `.com` (−11.0):** hai n-gram này lồng nhau nên hệ số của chúng **bù trừ cho nhau** (đa cộng tuyến, §2.4). Chúng **không** cho thấy shortcut cũ quay lại: sau chuẩn hóa, domain trần chỉ có 16.6% phishing, thấp hơn tỉ lệ nền².
- ⚠️ **Phát hiện artefact thời gian:** `tok:2011` nghiêng mạnh về hợp lệ (11,554 URL, chỉ 0.8% phishing²), còn `tok:2015` nghiêng về phishing (87%). Năm xuất hiện trong URL đang phản ánh **thời điểm thu thập dữ liệu** của từng nguồn, không phải tính độc hại. Mô hình triển khai thực tế cần được kiểm tra trên dữ liệu mới (xem §5.3).

---

## 4. So sánh hiệu năng: Baseline và mô hình cải tiến

### 4.1. Bảng tổng hợp metrics (tập test: 100,307 URL, gồm 78,572 hợp lệ và 21,735 phishing)

| Mô hình | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Recall@FPR≤1% | Ngưỡng | FP | FN |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|--:|--:|
| *B1. Baseline RF (NB05)* | 0.9063 | 0.8153 | 0.7340 | 0.7725 | 0.9315 | 0.8585 | 0.5630 | 0.500 | 3,615 | 5,782 |
| *B2. Baseline XGBoost tuned (NB05)* | 0.9114 | 0.8621 | 0.7034 | 0.7747 | 0.9479 | 0.8799 | 0.5904 | 0.500 | 2,445 | 6,447 |
| M1. LR + TF-IDF char/token | 0.9648 | 0.9512 | 0.8831 | 0.9159 | 0.9926 | 0.9788 | 0.8648 | 0.643 | 985 | 2,541 |
| M2. HistGradientBoosting | 0.9325 | **0.9740** | 0.7074 | 0.8196 | 0.9673 | 0.9263 | 0.7522 | 0.931 | **411** | 6,359 |
| M3. XGBoost + Optuna | 0.9371 | 0.8954 | 0.8034 | 0.8469 | 0.9622 | 0.9181 | 0.7315 | 0.671 | 2,039 | 4,273 |
| **M4. Stacking (TF-IDF LR → XGBoost)** | **0.9698** | 0.9555 | **0.9028** | **0.9284** | **0.9939** | **0.9825** | **0.8978** | 0.719 | 914 | **2,112** |
| M5. Soft Voting (LR + XGBoost) | 0.9662 | 0.9553 | 0.8856 | 0.9192 | 0.9930 | 0.9798 | 0.8802 | 0.598 | 900 | 2,486 |

### 4.2. Mô hình tốt nhất: Stacking (TF-IDF LR → XGBoost)

Stacking dẫn đầu **6/7 chỉ số chính** (Accuracy, Recall, F1, ROC-AUC, PR-AUC, Recall@FPR≤1%). Nó cũng có **Brier score thấp nhất** (0.0232), tức xác suất đầu ra sát thực tế nhất. Ba điểm đáng lưu ý:

- **Stacking hơn Soft Voting** ở mọi chỉ số xếp hạng (PR-AUC 0.9825 so với 0.9798; Recall@FPR≤1% 0.898 so với 0.880). Lợi thế này đến từ việc meta-model học *có điều kiện*: khi nào nên tin TF-IDF, khi nào nên tin đặc trưng brand/TLD. Phép trung bình cố định với `w = 0.7` không làm được điều đó.
- **HistGradientBoosting có Precision cao nhất** (0.974, chỉ 411 FP) nhưng Recall thấp (0.707). Nguyên nhân là ngưỡng chọn trên val (0.931) quá bảo thủ khi áp lên test. Reliability diagram (§17) cho thấy mô hình này **đánh giá quá cao xác suất phishing** (đường cong nằm dưới đường chéo) do dùng `class_weight='balanced'`. Mô hình này phù hợp với kịch bản "chặn tự động, chỉ chặn khi rất chắc chắn", nếu được hiệu chỉnh xác suất.
- **Đặc trưng thủ công tự thân đã vượt baseline:** cùng thuật toán XGBoost, chỉ thay bộ đặc trưng (M3 so với B2) đã nâng F1 từ 0.775 lên 0.847 (+7.2 điểm) và giảm 33.7% FN.

### 4.3. Mức cải thiện so với baseline

So sánh **Stacking (M4)** với **baseline tốt nhất B2** (XGBoost tuned, PR-AUC cao hơn B1), trên **cùng tập test**:

| Chỉ số | Baseline B2 | Stacking M4 | Δ tuyệt đối | Δ tương đối |
|---|:-:|:-:|:-:|:-:|
| Accuracy | 0.9114 | 0.9698 | +5.84 điểm | +6.4% (tỉ lệ lỗi giảm từ 8.86% xuống 3.02%, tức **−65.9%**) |
| Precision | 0.8621 | 0.9555 | +9.34 điểm | **+10.8%** |
| Recall | 0.7034 | 0.9028 | +19.94 điểm | **+28.3%** |
| F1-Score | 0.7747 | 0.9284 | +15.37 điểm | **+19.8%** |
| ROC-AUC | 0.9479 | 0.9939 | +4.61 điểm | +4.9% (phần diện tích còn thiếu 1−AUC giảm **−88.3%**) |
| PR-AUC | 0.8799 | 0.9825 | +10.25 điểm | +11.7% |
| Recall @ FPR ≤ 1% | 0.5904 | 0.8978 | +30.74 điểm | **+52.1%** |
| FPR | 3.11% | 1.16% | −1.95 điểm | **−62.6%** |
| FNR (tỉ lệ bỏ sót) | 29.66% | 9.72% | −19.94 điểm | **−67.2%** |

**Đối chiếu với số liệu notebook 05 tự báo cáo:** notebook 05 báo cáo RF đạt F1 0.826 và Accuracy 0.9255. Các con số đó được đo trên split **ngẫu nhiên**, có rò rỉ domain, nên lạc quan hơn thực tế: khi tái lập trên split theo domain, RF chỉ còn F1 0.7725 và Accuracy 0.9063. Dù được đánh giá theo giao thức khắt khe hơn, Stacking vẫn vượt con số lạc quan đó **+10.2 điểm F1** và **+4.4 điểm Accuracy**.

### 4.4. Ý nghĩa thực tiễn đối với hệ thống chặn link độc hại

Quy đổi trên tập test (~100,000 URL, mức tương đương lưu lượng một ngày của một tổ chức quy mô vừa):

| Khía cạnh vận hành | Baseline B2 | Stacking M4 | Ý nghĩa |
|---|:-:|:-:|---|
| **Phishing lọt qua** (FN) | 6,447 | 2,112 | **Chặn thêm 4,335 link độc hại.** FN là lỗi nguy hiểm nhất: mỗi link lọt qua là một cơ hội đánh cắp thông tin xác thực hoặc cài malware. Tỉ lệ bỏ sót giảm từ ~3/10 xuống **~1/10** |
| **URL sạch bị chặn nhầm** (FP) | 2,445 | 914 | **Giảm 1,531 lần chặn nhầm.** FP gây gián đoạn công việc, tạo ticket helpdesk và khiến người dùng mất niềm tin, thậm chí tìm cách vượt qua bộ lọc |
| **Tỉ lệ cảnh báo sai** trong số các cảnh báo gửi tới SOC (1 − Precision) | 13.8% | **4.4%** | Mỗi 100 cảnh báo, analyst chỉ phải xử lý **4–5 cảnh báo sai** thay vì 14. **Alert fatigue giảm hơn 3 lần** |
| **Recall khi khống chế FPR ≤ 1%** | 59.0% | **89.8%** | Ở chế độ "chặn tự động" yêu cầu cực ít báo nhầm, baseline chỉ bắt được hơn một nửa số phishing, còn mô hình mới bắt được gần 9/10 |

Điểm quan trọng nhất về mặt kỹ thuật: thông thường **Precision và Recall đánh đổi lẫn nhau** khi dịch ngưỡng. Mô hình mới cải thiện **đồng thời cả hai** (cả FP lẫn FN đều giảm hơn 60%). Như vậy toàn bộ đường cong Precision–Recall đã được đẩy lên (PR-AUC 0.880 → 0.983), chứ không chỉ là di chuyển điểm làm việc trên cùng một đường cong. Biểu đồ ROC phóng to (§17) xác nhận điều này: ở mọi mức FPR từ 0% tới 5%, đường cong Stacking nằm trên đường cong baseline khoảng 20–30 điểm TPR (ví dụ tại FPR = 1%: 0.898 so với 0.590).

**Tính bất biến (§18):** cùng một website viết theo nhiều cách (có/không scheme, `www.`, `/` cuối) nay nhận **cùng một xác suất** (độ lệch 0.000, so với 0.787 ở baseline với `google.com`). Kết quả với một số URL tiêu biểu:

| URL | Baseline B2 | Stacking M4 |
|---|:-:|:-:|
| `google.com` | 0.968 ❌ | 0.028 ✅ |
| `ikmbtfomlqp.deliverdarken.ru/start_page.exe` (malware/DGA) | 0.166 ❌ | 1.000 ✅ |
| `paypa1.com/signin` (typosquat) | 0.882 | 0.995 ✅ |
| `accounts.google.com/signin` (hợp lệ) | 0.667 | 0.142 ✅ |
| `gobankingrates.com/banking/checking-accounts/` (hợp lệ) | 0.996 ❌ | 0.782 ❌ (vẫn là FP) |

---

## 5. Kết luận và hướng đi tiếp theo

### 5.1. Khẳng định hiệu quả của hướng tiếp cận Traditional ML

1. **ML truyền thống đủ mạnh cho bài toán phát hiện phishing URL.** Một pipeline gồm Logistic Regression, XGBoost và Stacking đạt **ROC-AUC 0.994, F1 0.928, Recall 89.8% tại FPR 1%**, trên tập test **không chia sẻ domain** với tập train.
2. **Chất lượng biểu diễn quan trọng hơn độ phức tạp của mô hình.** Chỉ riêng mô hình tuyến tính (LR + TF-IDF) đã đạt F1 0.916, vượt mọi mô hình cây trên đặc trưng cũ. Đặc trưng thủ công có cơ sở bảo mật mang lại thêm khả năng **giải thích** và **sửa lỗi có chủ đích** (brand, TLD, typosquat).
3. **Lợi thế vận hành:** toàn bộ pipeline huấn luyện trong khoảng 10 phút trên CPU 8 nhân. Inference cho mỗi URL chỉ gồm thao tác chuỗi, một phép nhân ma trận thưa và vài trăm cây quyết định, nên đáp ứng yêu cầu thời gian thực **mà không cần GPU**. Mô hình nhỏ gọn (13 MB), mọi quyết định đều truy vết được tới đặc trưng hoặc n-gram cụ thể. Đây là yêu cầu quan trọng khi analyst cần giải thích vì sao một URL bị chặn.

### 5.2. Hai tinh chỉnh ưu tiên trước khi bàn giao cho SOC

#### (1) Tối ưu ngưỡng theo chính sách vận hành hai tầng

Ngưỡng hiện tại (0.719) tối đa hóa F1. Đây là lựa chọn hợp lý cho học thuật nhưng **chưa phải chính sách vận hành**. SOC nên dùng **hai ngưỡng**, chọn trên tập validation theo ràng buộc FPR:

| Vùng | Điều kiện | Hành động | Mục tiêu |
|---|---|---|---|
| 🔴 **Chặn tự động** | `p ≥ T_block` | Chặn ở proxy hoặc email gateway | FPR ≤ 0.3% (giữ helpdesk ổn định) |
| 🟡 **Cảnh báo / Sandbox** | `T_review ≤ p < T_block` | Gửi cảnh báo SIEM, cho detonate trong sandbox, hiện banner cảnh báo cho người dùng | Bắt thêm phần phishing còn lại để analyst xử lý |
| 🟢 **Cho phép** | `p < T_review` | Cho qua, ghi log | Recall tổng ≥ 95% |

```python
from sklearn.metrics import roc_curve

def threshold_for_fpr(y_val, p_val, max_fpr):
    """Ngưỡng nhỏ nhất (Recall lớn nhất) mà vẫn giữ FPR ≤ max_fpr trên validation."""
    fpr, tpr, thr = roc_curve(y_val, p_val)
    return thr[fpr <= max_fpr][-1]

T_block  = threshold_for_fpr(y_val, p_val_stack, max_fpr=0.003)   # vùng đỏ
T_review = threshold_for_fpr(y_val, p_val_stack, max_fpr=0.03)    # vùng vàng
```

Theo đường ROC phóng to (§17), Stacking đạt TPR khoảng 0.87 tại FPR 0.5% và khoảng 0.95 tại FPR 3%. Như vậy phần lớn phishing sẽ bị **chặn tự động**, và analyst chỉ phải xem xét một lượng nhỏ URL ở vùng vàng.

#### (2) Optuna cho meta-model, kèm hiệu chỉnh xác suất

- **Hiện trạng:** Optuna mới chỉ tune XGBoost trên đặc trưng thủ công (20 trials, 1.6 phút). Meta-model của Stacking **dùng lại** bộ tham số đó, dù tập đầu vào đã khác (thêm `lr_tfidf_score` chiếm 61% total gain). Chạy một study Optuna riêng cho meta-model (50–100 trials, tối ưu val PR-AUC, mất khoảng 10 phút) là bước rẻ và nhiều khả năng mang lại thêm cải thiện.
- **Hiệu chỉnh xác suất:** reliability diagram cho thấy các mô hình dùng trọng số lớp bị lệch xác suất. Nên bọc mô hình cuối bằng `CalibratedClassifierCV(method='isotonic')` trên một **tập calibration riêng**, tách thêm từ train theo domain. Như vậy giá trị "độ tin cậy 95%" hiển thị trên dashboard SOC mới phản ánh đúng xác suất thực, và ngưỡng ở mục (1) sẽ ổn định hơn qua các lần retrain.

### 5.3. Hạn chế cần lưu ý khi triển khai

| Hạn chế | Bằng chứng | Biện pháp (vẫn trong ML truyền thống) |
|---|---|---|
| **Dữ liệu cũ, đơn nguồn** | Artefact thời gian `tok:2011` / `tok:2015` (§3.4) | Bổ sung dữ liệu mới từ PhishTank, OpenPhish, URLhaus và danh sách Tranco; đánh giá theo **temporal split** |
| **Nhãn thiên lệch theo từ khóa** | LR TF-IDF chấm `accounts.google.com/signin` = 0.999; `gobankingrates.com/banking/...` vẫn bị chặn nhầm (0.782) | Allowlist domain uy tín làm đặc trưng (`domain_rank` từ Tranco); bổ sung URL đăng nhập hợp lệ vào tập train |
| **Hiệu năng biến động theo domain** | PR-AUC OOF dao động 0.85–0.97 giữa các fold | Báo cáo khoảng tin cậy bằng bootstrap; giám sát drift (PSI) và retrain định kỳ |
| **Chưa thử CatBoost/LightGBM thật** | Môi trường chưa cài các gói này | Cài đặt và so sánh. CatBoost xử lý TLD categorical trực tiếp bằng ordered target statistics |
| **Chưa có SHAP** | Notebook dùng permutation importance và gain | `shap.TreeExplainer` cho meta-XGBoost để giải thích từng cảnh báo cho analyst |

---

### Phụ lục: Truy xuất nguồn số liệu

| Nội dung báo cáo | Vị trí trong notebook 06 |
|---|---|
| Thống kê làm sạch, chia dữ liệu | §4, §6 |
| Phân phối đặc trưng, bảng cờ nhị phân | §8 |
| Heatmap Spearman, 9 cặp |ρ| ≥ 0.9, tương quan với nhãn | §9 |
| Metrics và biểu đồ ROC / PR / Confusion Matrix từng mô hình | §11–§16 |
| Permutation importance (HGB), XGBoost gain, Stacking gain, hệ số LR | §12, §13, §14, §15 |
| ROC/PR chồng lớp, reliability diagram | §17 |
| Kiểm tra bất biến | §18 |
| Bảng 1–3 tổng hợp và so sánh | §19 (đồng thời lưu tại `models/metrics_v2_test.csv` và `models/metrics_v2_vs_nb05.csv`) |
| Số liệu tính bổ sung (²) | Total gain từ `models/phishing_url_models_v2.joblib`; tỉ lệ phishing theo mẫu URL tính trực tiếp trên `data/raw/phishing_site_urls.csv` |
