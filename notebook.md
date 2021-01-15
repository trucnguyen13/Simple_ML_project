---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.9.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Đồ án cuối kỳ NMKHDL


### Câu hỏi: Sử dụng các mô hình máy học để dự đoán giá cả laptop trên thị trường theo các thông tin cấu hình từ máy.

### Giới thiệu thành viên

1. Nguyễn Đức Trực,
MSSV: 18120621

2. Nguyễn Trần Trung,
MSSV: 18120625


### Ý nghĩa của việc trả lời câu hỏi
* Khi một máy tính mới ra nhưng chưa bán trên thị trường (chưa có giá) thì có thể dùng mô hình dự đoán để tham khảo giá.
* Nhiều laptop tuy cấu hình yếu nhưng giá lại cao, việc dự đoán có thể giúp người mua xem xét có nên mua hay không.


## Phần 1: Khám phá dữ liệu (để tách các tập)

```python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns # seaborn là thư viện được xây trên matplotlib, giúp việc visualization đỡ khổ hơn
import pandas as pd
import numpy as np
```

```python
data_df = pd.read_csv('data.csv', index_col='SKU') # Dùng mã SKU của sản phẩm để làm index
data_df.head()
```

### Ý nghĩa các cột

```python
with open('description.txt', 'r', encoding = 'utf-8') as f:
    print(f.read())
```

```python
data_df.shape
```

```python
data_df.index.duplicated().sum()
```

Không có dòng trùng.

```python
# Cột dùng làm output là Price
data_df['Price'].dtype
```

Cột output có dữ liệu dạng Object nên chúng ta cần chuyển về dạng số để phù hợp với bài toán hồi quy.

```python
data_df.Price = pd.to_numeric(data_df.Price.str.replace('.', ''), errors='coerce')
```

```python
data_df['Price'].isna().sum()
```

Cột output không có giá trị thiếu


## Phần 2: Tiền xử lý (tách các tập)

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import set_config
set_config(display='diagram') # Để trực quan hóa pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
```

### Bây giờ ta sẽ thực hiện bước tiền xử lý là tách tập tập kiểm tra, validation và tập test ra theo tỉ lệ: 70%:15%:15%.

```python
# Tách X và y
y_sr = data_df["Price"] # sr là viết tắt của series
X_df = data_df.drop("Price", axis=1)
```

```python
# Tách tập huấn luyện, tập validation và tập test theo tỉ lệ 70%:15%:15%
# Tách dữ liệu thu thập thành 2 tập: tập huấn luyện và tập other (validation và test) theo tỉ lệ 70%:30%
train_X_df, other_X_df, train_y_sr, other_y_sr = train_test_split(X_df, y_sr, test_size=0.3, random_state=0)

# Tách tập other (validation và test) thành 2 tập: tập validation và tập test theo tỉ lệ 50%:50% 
val_X_df, test_X_df, val_y_sr, test_y_sr = train_test_split(other_X_df, other_y_sr, test_size=0.5, random_state=0)
```

```python
train_X_df.shape, val_X_df.shape, test_X_df.shape, train_y_sr.shape, val_y_sr.shape, test_y_sr.shape
```

## Phần 3: Khám phá dữ liệu (Tập huấn luyện)


### Mỗi cột input hiện đang có kiểu dữ liệu gì? Có cột nào có kiểu dữ liệu chưa phù hợp để có thể xử lý tiếp không?

```python
train_X_df.dtypes
```

 * Nhận xét về tập dữ liệu
   - Dữ liệu có 28 thuộc tính.
   - Một số thuộc tính có kiểu dữ liệu chưa phù hợp.


### Với mỗi cột input có kiểu dữ liệu không phải dạng số, các giá trị được phân bố như thế nào?

```python
num_cols = ['Warranty', 'SKU']

cat_cols = list(set(train_X_df.columns) - set(num_cols))
df = train_X_df[cat_cols]
def missing_ratio(df):
    return (df.isna().mean() * 100).round(1)
def num_values(df):
    return df.nunique()
def value_ratios(c):
    return dict((c.value_counts(normalize=True) * 100).round(1))
df.agg([missing_ratio, num_values, value_ratios])
```

## Phần 4: Tiền xử lý (tập huấn luyện)


Ta tiến hành tiền xử lý như sau:
   *  Với cột "CPUgen", ta sẽ tiến hành rút trích ra cột "CPUs" tương ứng. Tuy nhiên, cột "CPUs" có khá nhiều giá trị khác nhau nên chỉ lấy `num_top_cpus` (ví dụ, 6) giá trị xuất hiện nhiều nhất. Tương tự đối với cột "chipCPU" được rút chích từ cột "CPU".
   * Loại bỏ nhiều cột có nhiều giá trị khác nhau hoặc ít ảnh hưởng đến giá thành và các cột thiếu nhiều giá trị.
   * Rút trích các dữ liệu chính từ các cột và thay thế chúng. Như:
       - Cột "CPUgen" chỉ lấy tên CPU.
       - Cột "GraphicChip" chỉ lấy hảng sản xuất.
       - Cột "RAM" chỉ lấy số lượng RAM.
       - Cột "Screen" chỉ lấy kích thước inch.
       - Cột "Pin" chỉ lấy số cell.
       - Cột "Security" xét Yes nếu có bảo mật và ngược lại là No.
       - Cột "ChipCPU" chỉ lấy tên của chip CPU.
       - Cột "gen" lấy đời thứ mấy của CPU.
       - Cột "SSD" lấy kích thước của bộ nhớ.
   * Chuyển các cột dạng số về số.

```python
def convert_col_dtype(col):
    if col.name == 'SSD':
        col.replace('1','1000', inplace = True)
        col.replace('2','2000', inplace = True)
        return pd.to_numeric(col, errors='coerce')
    if col.name == 'gen':
        col.replace('1000',np.NaN,inplace = True)
        return pd.to_numeric(col, errors='coerce')
    if col.name == 'Security':     
        return col.apply(lambda x: 'Yes' if not pd.isnull(x) else 'No')
    if col.name in ['Pin','Weight','Screen','RAM']:  
        return pd.to_numeric(col, errors='coerce')  
    return col
```

```python
class ColAdderDropper(BaseEstimator, TransformerMixin):
    def __init__(self, num_top_cpus= 6, num_top_chipCPU= 3):
        self.num_top_cpus = num_top_cpus
        self.num_top_chipCPU = num_top_chipCPU

    def fit(self, X_df, y=None):
        _CPUs = X_df['CPUgen'].str.extract(r'([A-z0-9\s]+)\,').iloc[:,0]
        _gen = X_df['CPU'].str.extract(r'([A-z0-9]+)\s\(').iloc[:,0]
        _chipCPU = _gen.str.extract(r'([A-z])').iloc[:,0]
        
        self.cpus_counts_ = _CPUs.value_counts()
        cpus_ = list(self.cpus_counts_.index)
        self.top_cpus_ = cpus_[:max(1, min(self.num_top_cpus, len(cpus_)))]
        
        self.chipCPU_counts_ = _chipCPU.value_counts()
        chipCPU_ = list(self.chipCPU_counts_.index)
        self.top_chipCPU_ = chipCPU_[:max(1, min(self.num_top_chipCPU, len(chipCPU_)))]        
        
        return self
    def transform(self, X_df, y=None):
        _df = X_df.copy()
        
        _df['CPUs'] = _df['CPUgen'].str.extract(r'([A-z0-9\s]+)\,')
        _df['chipCPU'] = _df['CPU'].str.extract(r'([A-z0-9]+)\s\(')
        _df['chipCPU'] = _df['chipCPU'].str.extract(r'([A-z])')
        
        _df['CPUs'] =_df['CPUs'].apply(lambda x: x if x in col_adderdropper.top_cpus_ else 'Others')
        _df['chipCPU'] =_df['chipCPU'].apply(lambda x: x if x in col_adderdropper.top_chipCPU_ else 'Others')
        
        _df['gen'] = _df['CPUgen'].str.extract(r'([0-9]+)$')
        _df['GraphicChip'] = _df['GraphicChip'].str.extract(r'([A-z]+)\s')
        _df['RAM'] = _df['RAM'].str.extract(r'([0-9]+)GB')
        _df['Screen'] = _df['Screen'].str.extract(r'([0-9.]+)')
        _df['SSD'] = _df['Storage'].str.extract(r'([0-9]+)[A-z]')
        _df['Pin'] = _df['Pin'].str.extract(r'([0-9A-z]+)\scell')
        _df['Weight'] = _df['Weight'].str.extract(r'([0-9.]+)')
        
        unused_cols = ['Title', 'Warranty','Color','PartNum','MaxStoPortNum',
               'SupportM2','OutVideoPort','ConnectPort','Wireless','Keyboard',
               'Size','LED','Accessories','OptDrive','Feature','OS',
               'Storage', 'CPUgen','CPU','SeriesLaptop']

        _df = _df.apply(convert_col_dtype)
        _df = _df.drop(unused_cols,axis=1)
        return _df
```

```python
# TEST FIT METHOD
col_adderdropper = ColAdderDropper(num_top_cpus= 6, num_top_chipCPU= 3)
col_adderdropper.fit(train_X_df)
print(col_adderdropper.cpus_counts_)
print()
print(col_adderdropper.top_cpus_)
print()
print(col_adderdropper.chipCPU_counts_)
print()
print(col_adderdropper.top_chipCPU_)
```

```python
# Transform tập train
fewer_cols_train_X_df = col_adderdropper.transform(train_X_df)
fewer_cols_train_X_df.head()
```

```python
fewer_cols_train_X_df.dtypes
```

Có vẽ như kiểu dữ liệu các cột đã phù hợp.


### Với mỗi cột input có kiểu dữ liệu dạng số, các giá trị được phân bố như thế nào?

```python
num_cols = ['RAM', 'Screen', 'Pin', 'Weight','gen','SSD']
df = fewer_cols_train_X_df[num_cols]
def missing_ratio(df):
    return (df.isna().mean() * 100).round(1)
def lower_quartile(df):
    return df.quantile(0.25).round(1)
def median(df):
    return df.quantile(0.5).round(1)
def upper_quartile(df):
    return df.quantile(0.75).round(1)
df.agg([missing_ratio, 'min', lower_quartile, median, upper_quartile, 'max'])
```

 - Tỉ lệ phân chia khá đồng đều


### Với mỗi cột input có kiểu dữ liệu không phải dạng số, các giá trị được phân bố như thế nào?

```python
cat_cols = list(set(fewer_cols_train_X_df.columns) - set(num_cols))
df = fewer_cols_train_X_df[cat_cols]
df.agg([missing_ratio, num_values, value_ratios])
```

 - Các kiểu dữ liệu dạng categorical có khá ít giá trị, rất phù hợp để train.


### Các bước tiền xử lý tiếp theo như sau:

 - Với các cột dạng số, ta sẽ điền giá trị thiếu bằng giá trị mean của cột.
 - Với các cột không phải dạng số ta sẽ điền giá trị thiếu bằng giá trị mode (giá trị xuất hiện nhiều nhất) của cột. Sau đó, ta sẽ chuyển sang dạng số bằng phương pháp mã hóa one-hot (vì các cột này đều có dạng nominal).
 - Cuối cùng, khi tất cả các cột đã được điền giá trị thiếu và đã có dạng số, ta sẽ tiến hành chuẩn hóa.

```python
nume_cols = ['RAM','gen','SSD','Screen','Weight','Pin']
unorder_cate_cols = ['GraphicChip', 'Brand','Security','CPUs','chipCPU']

# YOUR CODE HERE
mean_numcols = SimpleImputer(missing_values = np.nan, strategy = 'mean')
mode_ordercols = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
mode_unordercols = make_pipeline(mode_ordercols, OneHotEncoder(handle_unknown='ignore'))

col_transform = ColumnTransformer([('nume_cols', mean_numcols, nume_cols),\
                                ('unorder_cate_cols', mode_unordercols,unorder_cate_cols)])

preprocess_pipeline = make_pipeline(col_adderdropper,col_transform,StandardScaler())
preprocessed_train_X = preprocess_pipeline.fit_transform(train_X_df)
```

```python
preprocessed_train_X.shape
```

```python
preprocess_pipeline
```

## Phần 5: Tiền xử lý + mô hình hóa


### Tìm mô hình tốt nhất


Sử dụng độ do **R-Squared** cho mô hình hồi quy.

```python
# Tính độ đo r^2 trên tập huấn luyện
def compute_mse(y, preds):
    return ((y - preds) ** 2).mean()
def compute_rr(y, preds, baseline_preds):
    return 1 - compute_mse(y, preds) / compute_mse(y, baseline_preds)
baseline_preds = train_y_sr.mean()
```

### 1. Mô hình SGDRegressor


 * Tham số: random_state=0
 * Siêu tham số alpha với 5 giá trị khác nhau.
 * Tham số num_top_cpus với 6 giá trị khác nhau.

```python
SGD_Regressort_model =  SGDRegressor(random_state=0)
full_pipeline1 = make_pipeline(col_adderdropper, col_transform, StandardScaler(), SGD_Regressort_model)

# Thử nghiệm với các giá trị khác nhau của các siêu tham số
# và chọn ra các giá trị tốt nhất
train_errs1 = []
val_errs1 = []
alphas = [0.01, 0.1, 1, 10, 100]
num_top_cpus_s = [1,3,5,6,8,9]
best_val_err1 = float('inf'); best_alpha1 = None; best_num_top_cpus1 = None
for alpha in alphas:
    for num_top_cpus in num_top_cpus_s:
        full_pipeline1.set_params(coladderdropper__num_top_cpus = num_top_cpus,coladderdropper__num_top_chipCPU = 3, sgdregressor__alpha=alpha)
        full_pipeline1.fit(train_X_df, train_y_sr)

        train_errs1.append(100 - compute_rr(train_y_sr, full_pipeline1.predict(train_X_df), baseline_preds) * 100)
        val_errs1.append(100 - compute_rr(val_y_sr, full_pipeline1.predict(val_X_df), baseline_preds) * 100)

        
        
        if val_errs1[-1] < best_val_err1:
            best_val_err1 = val_errs1[-1]
            best_alpha1 = alpha
            best_num_top_cpus1 = num_top_cpus

'Finish!'
```

```python
# Trực quan hóa kết quả
train_errs_df = pd.DataFrame(data=np.array(train_errs1).reshape(len(alphas), -1),
                             index=alphas, columns=num_top_cpus_s)
val_errs_df = pd.DataFrame(data=np.array(val_errs1).reshape(len(alphas), -1), 
                           index=alphas, columns=num_top_cpus_s)
min_err = min(min(train_errs1), min(val_errs1))
max_err = max(max(train_errs1), max(val_errs1))
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.heatmap(train_errs_df, vmin=min_err, vmax=max_err, square=True, annot=True, 
            cbar=False, fmt='.1f', cmap='Reds')
plt.title('train errors'); plt.xlabel('num_top_cpus'); plt.ylabel('alpha')
plt.subplot(1, 2, 2)
sns.heatmap(val_errs_df, vmin=min_err, vmax=max_err, square=True, annot=True, 
            cbar=False, fmt='.1f', cmap='Reds')
plt.title('validation errors'); plt.xlabel('num_top_cpus'); plt.ylabel('alpha');
```

```python
print(best_val_err1)
print(best_num_top_cpus1)
print(best_alpha1)
```

### 2. Mô hình RandomforestRegressor


 * Tham số: random_state=0
 * Siêu tham số max_depth với 5 giá trị khác nhau.
 * Tham số num_top_cpus với 5 giá trị khác nhau.

```python
RandomForest_Regressorneural_model = RandomForestRegressor(random_state=0)
full_pipeline3 = make_pipeline(col_adderdropper, col_transform, StandardScaler(), RandomForest_Regressorneural_model)

train_errs3 = []
val_errs3 = []
num_top_cpus_s = [1,3,6,8,9]
max_depths = [16, 32, 64, 128, 256]
best_val_err3 = float('inf');  best_num_top_cpus3 = None;  best_depth3 = None;


for max_depth in max_depths:
    for num_top_cpus in num_top_cpus_s:
        full_pipeline3.set_params(coladderdropper__num_top_cpus = num_top_cpus,coladderdropper__num_top_chipCPU = 3, randomforestregressor__max_depth = max_depth)
        full_pipeline3.fit(train_X_df, train_y_sr)

        train_errs3.append(100 - compute_rr(train_y_sr, full_pipeline3.predict(train_X_df), baseline_preds) * 100)
        val_errs3.append(100 - compute_rr(val_y_sr, full_pipeline3.predict(val_X_df), baseline_preds) * 100)

        if val_errs3[-1] < best_val_err3:
            best_val_err3 = val_errs3[-1]
            best_depth3 = max_depth
            best_num_top_cpus3 = num_top_cpus

'Finish!'
```

```python
# Trực quan hóa kết quả
train_errs_df = pd.DataFrame(data=np.array(train_errs3).reshape(len(max_depths), -1),
                             index=max_depths, columns=num_top_cpus_s)
val_errs_df = pd.DataFrame(data=np.array(val_errs3).reshape(len(max_depths), -1), 
                           index=max_depths, columns=num_top_cpus_s)
min_err = min(min(train_errs3), min(val_errs3))
max_err = max(max(train_errs3), max(val_errs3))
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.heatmap(train_errs_df, vmin=min_err, vmax=max_err, square=True, annot=True, 
            cbar=False, fmt='.5f', cmap='Reds')
plt.title('train errors'); plt.ylabel('best_depth'); plt.xlabel('num_top_cpus_s');
plt.subplot(1, 2, 2)
sns.heatmap(val_errs_df, vmin=min_err, vmax=max_err, square=True, annot=True, 
            cbar=False, fmt='.5f', cmap='Reds')
plt.title('validation errors'); plt.ylabel('best_depth'); plt.xlabel('num_top_cpus_s');
```

```python
print(best_val_err3)
print(best_num_top_cpus3)
print(best_depth3)
```

### Đánh giá kết quả trên tập validation của mô hình thu được


* Cả mô hình random forest regression và sgd regression cho kết quả khả quan trên tập validation, tuy nhiên kết quả vẫn còn khá chủ quan vì việc lựa chọn các siêu tham số đều được làm bằng tay.
* Cả 2 mô hình đều chạy khá nhanh, nhóm em chọn mô hình random forest regression vì kết quả có vẻ tốt hơn.


#### Train lại bằng mô hình random forest regression với các siêu tham số tối ưu tìm được trên tập train + validation.

```python
full_train_X_df = train_X_df.append(val_X_df)
full_train_y_sr = train_y_sr.append(val_y_sr)

full_pipeline3.set_params(coladderdropper__num_top_cpus = best_num_top_cpus3,coladderdropper__num_top_chipCPU = 3,randomforestregressor__max_depth = best_depth3)
full_pipeline3.fit(full_train_X_df, full_train_y_sr)
```

#### So sánh kết quả predict với kết quả thực

```python
preds = full_pipeline3.predict(test_X_df).round(0)
# So sánh kết quả predict với kết quả actual
preds_df = pd.DataFrame(preds, index=test_y_sr.index).rename(columns={0: 'Predict'})
preds_df = preds_df.assign(Actual = test_y_sr)
preds_df.sample(20)
```

#### Độ chính xác trên tập test

```python
compute_rr(test_y_sr, full_pipeline3.predict(test_X_df), baseline_preds)
```

## Phần 6: Nhìn lại quá trình làm đồ án


* Khó khăn
    * Khó khăn trong việc lấy dữ liệu từ các trang web bán laptop.
    * Dữ liệu parse được không nhiều.
    * Khó khăn trong việc tìm hiểu các siêu tham số cho mô hình.
    * Khó khăn trong việc chọn các thuộc tính, đặc trưng phù hợp để train.
    * Thời gian hạn chế vì đồ án diễn ra trong thời gian thi cử.
* Những điều hữu ích học được
    * Kĩ năng làm việc nhóm.
    * Kĩ năng sử dụng các công cụ hỗ trợ làm đồ án (như github, trello,...).
    * Các kĩ năng khám phá dữ liệu.
    * Tìm hiểu được thêm nhiều mô hình máy học hay.
    * Hiểu sâu sắc hơn quy trình Khoa học dữ liệu qua việc tự tìm hiểu và làm đồ án.
* Những dự định nếu có thời gian thêm
    * Thêm phần phân tích tương quan dữ liệu để chọn thuộc tính, đặc trưng phù hợp hơn.
    * Tìm hiểu kỹ hơn các mô hình hiện tại cũng như tìm hiểu thêm các mô hình khác để đưa ra các siêu tham số tối ưu hơn.
    * Tiền xử lý dữ liệu sạch hơn.
    * Chuẩn bị slide báo cáo hoàn chỉnh hơn.


## Phần 7: Tài liệu tham khảo


* https://www.kaggle.com/danielbethell/laptop-prices-prediction
* https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/#one
* https://machinelearningcoban.com/2017/01/16/gradientdescent2/
* https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html
* https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
* https://towardsdatascience.com/machine-learning-basics-random-forest-regression-be3e1e3bb91a
