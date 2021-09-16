from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#########################
# Verinin Veri Tabanından Okunması
#########################

# credentials.
creds = {'user': 'group_02',
         'passwd': 'hayatguzelkodlarucuyor',
         'host': '34.88.156.118',
         'port': 3306,
         'db': 'group_02'}

# MySQL conection string.
connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'

# sqlalchemy engine for MySQL connection.
conn = create_engine(connstr.format(**creds))

# conn.close()


pd.read_sql_query("show databases;", conn)
pd.read_sql_query("show tables", conn)



pd.read_sql_query("select * from online_retail_2010_2011 limit 10", conn)


retail_mysql_df = pd.read_sql_query("select * from online_retail_2010_2011", conn)



##### Görev 1 ##########

df = retail_mysql_df.copy()
df = df[df['Country'] == 'United Kingdom']
df.head()


# VERI ON ISLEME

df.describe().T
df.dropna(inplace=True)   # na boş değer(missing-value) silmek için
df.drop('Unix', axis=1 , inplace=True ) # Unix isimli değişkene ihtiyacımız yok, df'den kalıcı olarak siliyoruz.
df.drop('id', axis=1, inplace=True) # id isimli değişkene ihtiyacımız yok, df'den kalıcı olarak siliyoruz.
df.head()
df = df[~df["Invoice"].str.contains("C", na=False)]   # Fatura'da iade edilenleri çıkarıyoruz.
df = df[df["Quantity"] > 0]     # Ürün adedi negatif olamaz.
df = df[df["Price"] > 0]     # Fiyat negatif olamaz.

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df.describe().T

df["TotalPrice"] = df["Quantity"] * df["Price"]   # Toplam ürün tutarı
df.head()
today_date = dt.datetime(2011, 12, 11)


# LifeTime Veri Yapısının Hazırlanması

# recency: Son satın alma ile ilk satın alma üzerinden geçen zaman. Haftalık. (daha önce analiz gününe göre, burada kullanıcı özelinde)
# T: Analiz tarihinden ne kadar süre önce ilk satın alma yapılmış. Haftalık. (müşteri yaşı)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary_value: satın alma başına ortalama kazanç


cltv_df = df.groupby('CustomerID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})
cltv_df.head()

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
cltv_df.columns

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

# cltv_df = cltv_df[cltv_df["monetary"] > 0]

cltv_df["recency"] = cltv_df["recency"] / 7

cltv_df["T"] = cltv_df["T"] / 7

# cltv_df["frequency"] = cltv_df["frequency"].astype(int)

cltv_df.info()
cltv_df = cltv_df.reset_index()
cltv_df.head()

# BGNBD MODELININ KURULMASI

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

# 1 hafta içerisinde en çok satın alma beklediğimiz 10 müşteri kimlerdir?
cltv_df['expected_purc_1_week'] = bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T']).sort_values(ascending=False)


cltv_df['expected_purc_1_month'] = bgf.conditional_expected_number_of_purchases_up_to_time(4,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T']).sort_values(ascending=False)
cltv_df.head()


# GAMMA GAMMA SUBMODEL MODELININ KURULMASI

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

cltv_df['expected_average_profit_clv'] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                                 cltv_df['monetary'])

cltv_df.head()
###### BG-NBD ve GG modeli ile CLTV'nin hesaplanması ########


cltv_6_aylik = (ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01))
cltv_6_aylik = cltv_6_aylik.reset_index()
cltv_6_aylik.head()
cltv_6_aylik.sort_values(by='clv', ascending=False).head(20)
cltv_6_aylik.describe().T

# Yorum: Gelen sonuçlara bakıldığında 2010-2011 verilerine göre UK müşterilerinin 6 aylık CLTV Prediction'ları yapılmıştır.
# 6 aylık UK müşterilerinin ortalama getirisinin 1516 birim olması beklenmektedir.


###### Görev 2 ######

cltv_1_aylik = (ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=1,  # 1 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01))
cltv_1_aylik = cltv_1_aylik.reset_index()
cltv_1_aylik.sort_values(by='clv' , ascending=False).head(10)
type(cltv_1_aylik)
cltv_1_aylik.describe().T

cltv_12_aylik = (ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=12,  # 12 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01))
cltv_12_aylik = cltv_12_aylik.reset_index()
cltv_12_aylik.sort_values(by='clv', ascending=False).head(10)
cltv_12_aylik.describe().T

# 1 aylık ve 12 aylık  en çok getirisi olan müşteriler kıyaslandığında; 1 aylık listede olup 12 aylık listede olmayan müşteriler
# gözlemlenmiştir, bu müşterilerin churn olduğu düşünülebilir. Zaman olarak kıyaslama yaptığımızda bir müşterinin en son satın alma
# tarihinden ne kadar zaman geçtiyse müşterinin tekrar alışveriş yapma ihtimali o kadar yüksek olacaktır. Bu yüzden 12 aylık müşterilerin
# de ortalama getirisi daha yüksek olması beklenir.


###### Görev 3 ######

cltv_6_aylik['segment'] = pd.qcut(cltv_6_aylik['clv'], 4, labels=['D', 'C', 'B', 'A'])
cltv_6_aylik.head()

cltv_6_aylik[cltv_6_aylik['segment'] == 'A'].head()
cltv_6_aylik[cltv_6_aylik['segment'] == 'C'].head()

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_6_aylik[["clv"]])
cltv_6_aylik["scaled_clv"] = scaler.transform(cltv_6_aylik[["clv"]])

cltv_6_aylik.drop('clv', axis=1, inplace=True)
cltv_6_aylik.head()
cltv_df.head()

###### Görev 4 ######

cltv_final = cltv_df.merge(cltv_6_aylik, on='CustomerID', how='left')
cltv_final.head()

cltv_final["CustomerID"] = cltv_final["CustomerID"].astype(int)

cltv_final.to_sql(name='nigaresra_kin', con=conn, if_exists='replace', index=False)

pd.read_sql_query("show databases;", conn)
pd.read_sql_query("show tables", conn)


conn.close()