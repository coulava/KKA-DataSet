# =========================================
# ANALISIS PERFORMA PENJUALAN E-COMMERCE
# =========================================

# 🔹 Langkah 1: Import Library & Load Data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# ✅ FIX NAMA FILE CSV
df = pd.read_csv('data_praktikum_analisis_data.csv')

print("=== DATA AWAL ===")
print(df.head())


# 🔹 Langkah 2: Data Understanding
print("\n=== INFO DATA ===")
print(df.info())

print("\n=== MISSING VALUE ===")
print(df.isnull().sum())


# 🔹 Langkah 3: Data Cleaning
# ⚠️ Cek dulu apakah kolom ada
if 'Price' in df.columns:
    df = df[df['Price'] > 0]

if 'Order_Date' in df.columns:
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])


# 🔹 Langkah 4: Analisis Tren Penjualan
if 'Order_Date' in df.columns and 'Total_Sales' in df.columns:
    df['Month'] = df['Order_Date'].dt.to_period('M').astype(str)

    monthly_sales = df.groupby('Month')['Total_Sales'].sum()

    plt.figure()
    plt.plot(monthly_sales.index, monthly_sales.values, marker='o')
    plt.title('Tren Penjualan Bulanan')
    plt.xticks(rotation=45)
    plt.show()


# 🔹 Langkah 5: Analisis Korelasi
cols = ['Total_Sales', 'Ad_Budget', 'Discount_Percentage']
available_cols = [col for col in cols if col in df.columns]

if len(available_cols) >= 2:
    correlation = df[available_cols].corr()

    sns.heatmap(correlation, annot=True)
    plt.title('Korelasi Antar Variabel')
    plt.show()


# 🔹 Langkah 6: Produk Underperformer
if 'Price_Per_Unit' in df.columns and 'Quantity' in df.columns:
    plt.figure()
    plt.scatter(df['Price_Per_Unit'], df['Quantity'])
    plt.xlabel('Price per Unit')
    plt.ylabel('Quantity')
    plt.title('Produk Underperformer')
    plt.show()


# 🔹 Langkah 7: RFM Analysis
if 'CustomerID' in df.columns and 'Order_Date' in df.columns:
    snapshot_date = df['Order_Date'].max() + dt.timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'Order_Date': lambda x: (snapshot_date - x.max()).days,
        'Order_ID': 'count' if 'Order_ID' in df.columns else 'size',
        'Total_Sales': 'sum'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])

    rfm['RFM_Group'] = (
        rfm['R_Score'].astype(str) +
        rfm['F_Score'].astype(str) +
        rfm['M_Score'].astype(str)
    )

    print("\n=== RFM ===")
    print(rfm.head())


# 🔹 Langkah 8: Efisiensi Kategori
if 'Category' in df.columns and 'Total_Sales' in df.columns:
    category_analysis = df.groupby('Category').agg({
        'Total_Sales': 'sum',
        'Ad_Budget': 'sum' if 'Ad_Budget' in df.columns else 'count'
    })

    category_analysis.plot(kind='barh')
    plt.title('Efisiensi Kategori')
    plt.show()


# 🔹 Langkah 9: Uji Hipotesis
if 'Ad_Budget' in df.columns and 'Total_Sales' in df.columns:
    median_ads = df['Ad_Budget'].median()

    high_ads = df[df['Ad_Budget'] > median_ads]
    low_ads = df[df['Ad_Budget'] <= median_ads]

    print("\n=== UJI HIPOTESIS ===")
    print("Rata-rata High Ads:", high_ads['Total_Sales'].mean())
    print("Rata-rata Low Ads:", low_ads['Total_Sales'].mean())


# 🔹 Langkah 10: Regresi Linear
if 'Ad_Budget' in df.columns and 'Total_Sales' in df.columns:
    from sklearn.model_selection import train_test_split 
    from sklearn.linear_model import LinearRegression 

    X = df[['Ad_Budget']]
    y = df['Total_Sales']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("\n=== REGRESI LINEAR ===")
    print("Koefisien:", model.coef_[0])
    print("Akurasi (R2):", model.score(X_test, y_test))