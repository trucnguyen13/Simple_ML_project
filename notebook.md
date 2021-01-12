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


## Phần 1: Thu thập dữ liêu

```python
import requests
from bs4 import BeautifulSoup
```

### Get tất cả các link laptop

```python
items = []
for page in range(1, 23):
    pv_laptop_url = 'https://phongvu.vn/laptop-va-linh-kien-macbook-715.cat?pv_source=homepage&pv_medium=de-megamenu-text&page=' + str(page)
    html_text = requests.get(pv_laptop_url).text
    tree = BeautifulSoup(html_text, 'html.parser')
    items = items + tree.find_all('a', {'class': 'css-1rhapru'})
```

```python tags=[]
for item in items[:5]:
    print(item['href'])
```

```python
len(items)
```

```python
import csv
```

### Parse từng link và ghi ra file csv

```python
fields = {'Thương hiệu': 'Brand', 'Bảo hành': 'Warranty', 'Màu sắc': 'Color', 'Series laptop': 'SeriesLaptop', 'Part-number': 'PartNum', 'Thế hệ CPU': 'CPUgen', 'CPU': 'CPU', 'Chip đồ họa': 'GraphicChip', 'RAM': 'RAM', 'Màn hình': 'Screen', 'Lưu trữ': 'Storage', 'Số cổng lưu trữ tối đa': 'MaxStoPortNum', 'Kiểu khe M.2 hỗ trợ': 'SupportM2', 'Cổng xuất hình': 'OutVideoPort', 'Cổng kết nối': 'ConnectPort', 'Kết nối không dây': 'Wireless', 'Bàn phím': 'Keyboard', 'Hệ điều hành': 'OS', 'Kích thước': 'Size', 'Pin': 'Pin', 'Khối lượng': 'Weight', 'Đèn LED trên máy': 'LED', 'Phụ kiện đi kèm': 'Accessories', 'Bảo mật': 'Security', 'Ổ đĩa quang': 'OptDrive', 'Tính năng': 'Feature'}

fieldnames = ['SKU', 'Title', 'Price', 'Brand', 'Warranty', 'Color', 'SeriesLaptop', 'PartNum', 'CPUgen', 'CPU', 'GraphicChip', 'RAM', 'Screen', 'Storage', 'MaxStoPortNum', 'SupportM2', 'OutVideoPort', 'ConnectPort', 'Wireless', 'Keyboard', 'OS', 'Size', 'Pin', 'Weight', 'LED', 'Accessories', 'Security', 'OptDrive', 'Feature']
```

```python
file = open('data.csv', 'w', encoding='utf-8')
file_writer = csv.DictWriter(file, fieldnames=fieldnames)
file_writer.writeheader()

count = 0
for item in items:
    count += 1
    url = 'https://phongvu.vn' + item['href']
    laptop_html_text = requests.get(url).text
    laptop_html_tree = BeautifulSoup(laptop_html_text, 'html.parser')
    title = laptop_html_tree.find('div', {'class': 'css-1jpdzyd'}).text
    sku = laptop_html_tree.find('div', {'class': 'css-5nimvs'}).text.split(' ')[-1]
    price = laptop_html_tree.find('span', {'class': 'css-3725in'}).text[:-1]
    values = {'SKU': sku, 'Title': title, 'Price': price} 
    for i in laptop_html_tree.find_all('div', {'class': 'css-7j9rw7'}):
        name = i.find('span', {'class': 'css-6z2lgz'}).text.strip(' ')
        if name in fields:
            values[fields[name]] = i.find('div', {'class': 'css-111s35w'}).text.strip(' ')
    file_writer.writerow(values)
    if count % 10 == 0:
        print(count)
file.close()
```

```python
import pandas as pd
```

```python
df = pd.read_csv('data.csv', sep=',', index_col='SKU')
df.head()
```

```python
df.shape
```
