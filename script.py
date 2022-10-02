import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
from matplotlib.pyplot import figure
from scipy import stats
from scipy.stats import norm, skew # for some statistics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot
from sklearn.tree import DecisionTreeRegressor
from sklearn import set_config 
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs



plt.style.use('ggplot')
matplotlib.rcParams['figure.figsize'] = (12,8)
pd.options.mode.chained_assignment = None


def clean_data():
    df = pd.read_csv('2020.csv',sep=';')

    # shape and data types of the data
    print(df.shape)
    print(df.dtypes)

    # select numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    numeric_cols = df_numeric.columns.values
    print(numeric_cols)

    # select non numeric columns
    df_non_numeric = df.select_dtypes(exclude=[np.number])
    non_numeric_cols = df_non_numeric.columns.values
    print(non_numeric_cols)

    # convert data to numeric
    df['Nilai_UMR'] = pd.to_numeric(df['Nilai_UMR'], errors='coerce')
    df['Pengguna_Ponsel'] = pd.to_numeric(df['Pengguna_Ponsel'], errors='coerce')
    df['Jumlah_Agen_Pulsa'] = pd.to_numeric(df['Jumlah_Agen_Pulsa'], errors='coerce')

    # check missing values
    for col in df.columns:
        pct_missing = df[col].isnull().sum()
        print(f'{col} - {pct_missing}')

    # MISSING VALUES PEMILIK PONSEL(MEDIAN PROVINSI)
    nan_in_col  = df[df['Pemilik_Ponsel'].isnull()]
    print(nan_in_col)

    # handle Sulawesi Data
    df_sul = pd.DataFrame(df[df['Regional'] == 'Sulawesi'])
    median_pemilik_ponsel_sulawesi = df_sul['Pemilik_Ponsel'].median()
    df_nan_pemilik_ponsel = pd.DataFrame(nan_in_col)
    df.at[297,'Pemilik_Ponsel'] = median_pemilik_ponsel_sulawesi
    df.at[360,'Pemilik_Ponsel'] = median_pemilik_ponsel_sulawesi
    df.at[380,'Pemilik_Ponsel'] = median_pemilik_ponsel_sulawesi
    df.at[407,'Pemilik_Ponsel'] = median_pemilik_ponsel_sulawesi
    df.at[423,'Pemilik_Ponsel'] = median_pemilik_ponsel_sulawesi
    df.at[452,'Pemilik_Ponsel'] = median_pemilik_ponsel_sulawesi

    # MISSING VALUES PENGGUNA PONSEL
    nan_in_col2  = df[df['Pengguna_Ponsel'].isnull()]
    print(nan_in_col2)

    median_pengguna_ponsel = df_sul['Pengguna_Ponsel'].median()
    df.at[301,'Pengguna_Ponsel'] = median_pengguna_ponsel
    df.at[312,'Pengguna_Ponsel'] = median_pengguna_ponsel
    df.at[323,'Pengguna_Ponsel'] = median_pengguna_ponsel
    df.at[329,'Pengguna_Ponsel'] = median_pengguna_ponsel
    df.at[332,'Pengguna_Ponsel'] = median_pengguna_ponsel
    df.at[353,'Pengguna_Ponsel'] = median_pengguna_ponsel
    df.at[362,'Pengguna_Ponsel'] = median_pengguna_ponsel
    df.at[363,'Pengguna_Ponsel'] = median_pengguna_ponsel
    df.at[369,'Pengguna_Ponsel'] = median_pengguna_ponsel
    df.at[371,'Pengguna_Ponsel'] = median_pengguna_ponsel
    df.at[375,'Pengguna_Ponsel'] = median_pengguna_ponsel
    df.at[392,'Pengguna_Ponsel'] = median_pengguna_ponsel
    df.at[412,'Pengguna_Ponsel'] = median_pengguna_ponsel
    df.at[416,'Pengguna_Ponsel'] = median_pengguna_ponsel
    df.at[419,'Pengguna_Ponsel'] = median_pengguna_ponsel
    df.at[432,'Pengguna_Ponsel'] = median_pengguna_ponsel
    df.at[471,'Pengguna_Ponsel'] = median_pengguna_ponsel

    # **MISSING VALUES JUMLAH AGEN PULSA**
    nan_in_col3  = df[df['Jumlah_Agen_Pulsa'].isnull()]
    print(nan_in_col3)

    median_agen_pulsa_sulawesi = df_sul['Jumlah_Agen_Pulsa'].median()
    df_papua_barat = pd.DataFrame(df[df['Provinsi'] == 'Papua Barat'])
    median_agen_pulsa_papua_barat = df_papua_barat['Jumlah_Agen_Pulsa'].median()

    df.at[329,'Jumlah_Agen_Pulsa'] = median_agen_pulsa_sulawesi
    df.at[332,'Jumlah_Agen_Pulsa'] = median_agen_pulsa_sulawesi
    df.at[344,'Jumlah_Agen_Pulsa'] = median_agen_pulsa_sulawesi
    df.at[363,'Jumlah_Agen_Pulsa'] = median_agen_pulsa_sulawesi
    df.at[383,'Jumlah_Agen_Pulsa'] = median_agen_pulsa_sulawesi
    df.at[444,'Jumlah_Agen_Pulsa'] = median_agen_pulsa_papua_barat
    df.at[451,'Jumlah_Agen_Pulsa'] = median_agen_pulsa_papua_barat
    df.at[471,'Jumlah_Agen_Pulsa'] = median_agen_pulsa_sulawesi

    # mengisi missing values Provinsi
    df.at[98,'Provinsi'] = 'Sumatera Utara'
    df.at[167,'Provinsi'] = 'Jawa Barat'
    df.at[175,'Provinsi'] = 'Jawa Barat'
    df.at[235,'Provinsi'] = 'Jawa Tengah'
    df.at[237,'Provinsi'] = 'Jawa Tengah'
    df.at[261,'Provinsi'] = 'Jawa Timur'
    df.at[332,'Provinsi'] = 'Sulawesi Tenggara'
    df.at[338,'Provinsi'] = 'Papua Barat'

    # sanity check
    for col in df.columns:
        pct_missing = df[col].isnull().sum()
        print(f'{col} - {pct_missing}')

    # missing values di sumatera utara
    df.loc[(df.Provinsi == 'Sumatera Utara') & (df['Nilai_UMR'].isnull())]
    UMR_SUMUT = 3222526
    df.at[14,'Nilai_UMR'] = UMR_SUMUT
    df.at[20,'Nilai_UMR'] = UMR_SUMUT
    df.at[58,'Nilai_UMR'] = UMR_SUMUT

    # missing values diprovinsi sulawesi utara
    df.loc[(df.Provinsi == 'Sulawesi Utara') & (df['Nilai_UMR'].isnull())]
    UMR_SULUT = 3310000
    df.at[311,'Nilai_UMR'] = UMR_SULUT
    df.at[326,'Nilai_UMR'] = UMR_SULUT
    df.at[327,'Nilai_UMR'] = UMR_SULUT
    df.at[331,'Nilai_UMR'] = UMR_SULUT
    df.at[333,'Nilai_UMR'] = UMR_SULUT
    df.at[335,'Nilai_UMR'] = UMR_SULUT
    df.at[337,'Nilai_UMR'] = UMR_SULUT
    df.at[340,'Nilai_UMR'] = UMR_SULUT
    df.at[342,'Nilai_UMR'] = UMR_SULUT
    df.at[351,'Nilai_UMR'] = UMR_SULUT
    df.at[352,'Nilai_UMR'] = UMR_SULUT
    df.at[355,'Nilai_UMR'] = UMR_SULUT
    df.at[376,'Nilai_UMR'] = UMR_SULUT
    df.at[399,'Nilai_UMR'] = UMR_SULUT
    df.at[404,'Nilai_UMR'] = UMR_SULUT

    # missing values Sulawesi Tengah
    df.loc[(df.Provinsi == 'Sulawesi Tengah') & (df['Nilai_UMR'].isnull())]
    UMR_SULTENG = 2390739
    df.at[325,'Nilai_UMR'] = UMR_SULTENG
    df.at[365,'Nilai_UMR'] = UMR_SULTENG
    df.at[378,'Nilai_UMR'] = UMR_SULTENG
    df.at[383,'Nilai_UMR'] = UMR_SULTENG
    df.at[413,'Nilai_UMR'] = UMR_SULTENG
    df.at[426,'Nilai_UMR'] = UMR_SULTENG
    df.at[428,'Nilai_UMR'] = UMR_SULTENG
    df.at[434,'Nilai_UMR'] = UMR_SULTENG

    # sanity Check
    print(df[df.isnull().any(axis=1)])

    # FILL ZERO
    for col in df.columns:
        pct_zero = (df[col] == 0).sum()
        print(f'{col} - {pct_zero}')

    print(df.loc[(df.Nilai_UMR == 0)])

    # input data UMR
    UMR_Jambi = 2930000
    UMR_Bengkulu = 2215000
    UMR_Babel = 3230000
    UMR_Jabar = 3700000
    UMR_Jateng = 2810000
    UMR_Sulsel = 3255000
    UMR_Sulbar = 2678000
    UMR_Maluku = 2604000
    UMR_Kalbar = 2434000
    UMR_Kalsel = 2877000
    UMR_Papuabarat = 3516000
    UMR_Kaltim = 3014000
    UMR_NTT = 1975000 
    UMR_Jatim = 4375000

    df.at[0,'Nilai_UMR'] = UMR_Jambi
    df.at[3,'Nilai_UMR'] = UMR_Bengkulu
    df.at[4,'Nilai_UMR'] = UMR_Jambi
    df.at[7,'Nilai_UMR'] = UMR_Bengkulu
    df.at[17,'Nilai_UMR'] = UMR_Bengkulu
    df.at[21,'Nilai_UMR'] = UMR_Jambi
    df.at[24,'Nilai_UMR'] = UMR_Bengkulu
    df.at[28,'Nilai_UMR'] = UMR_Jambi
    df.at[29,'Nilai_UMR'] = UMR_Jambi
    df.at[33,'Nilai_UMR'] = UMR_Bengkulu
    df.at[42,'Nilai_UMR'] = UMR_Jambi
    df.at[46,'Nilai_UMR'] = UMR_Bengkulu
    df.at[49,'Nilai_UMR'] = UMR_Bengkulu
    df.at[54,'Nilai_UMR'] = UMR_Bengkulu
    df.at[56,'Nilai_UMR'] = UMR_Bengkulu
    df.at[59,'Nilai_UMR'] = UMR_Jambi
    df.at[62,'Nilai_UMR'] = UMR_Bengkulu
    df.at[82,'Nilai_UMR'] = UMR_Jambi
    df.at[94,'Nilai_UMR'] = UMR_Jambi
    df.at[99,'Nilai_UMR'] = UMR_Jambi
    df.at[104,'Nilai_UMR'] = UMR_Jambi
    df.at[107,'Nilai_UMR'] = UMR_Babel
    df.at[109,'Nilai_UMR'] = UMR_Babel
    df.at[136,'Nilai_UMR'] = UMR_Babel
    df.at[137,'Nilai_UMR'] = UMR_Babel
    df.at[139,'Nilai_UMR'] = UMR_Babel
    df.at[142,'Nilai_UMR'] = UMR_Babel
    df.at[144,'Nilai_UMR'] = UMR_Babel
    df.at[171,'Nilai_UMR'] = UMR_Jabar
    df.at[207,'Nilai_UMR'] = UMR_Jatim
    df.at[212,'Nilai_UMR'] = UMR_Jateng
    df.at[236,'Nilai_UMR'] = UMR_Jateng
    df.at[266,'Nilai_UMR'] = UMR_Jateng
    df.at[364,'Nilai_UMR'] = UMR_Sulsel
    df.at[380,'Nilai_UMR'] = UMR_Sulbar
    df.at[391,'Nilai_UMR'] = UMR_Maluku
    df.at[393,'Nilai_UMR'] = UMR_Kalbar
    df.at[405,'Nilai_UMR'] = UMR_Kalsel
    df.at[411,'Nilai_UMR'] = UMR_Papuabarat
    df.at[456,'Nilai_UMR'] = UMR_Kaltim
    df.at[498,'Nilai_UMR'] = UMR_NTT

    print(df.loc[(df.Nilai_UMR == 0)])

    # **FILL ZERO DANA ALOKASI UMUM**
    print("FILL ZERO DANA ALOKASI UMUM ...")
    print(df.loc[(df.Dana_Alokasi_Umum == 0)])

    df_jawa2 = pd.DataFrame(df[df['Area'] == 'Area 2'])
    median_jawa_area2 = df_jawa2['Dana_Alokasi_Umum'].median()

    df_jawa3 = pd.DataFrame(df[df['Area'] == 'Area 3'])
    median_jawa_area3 = df_jawa3['Dana_Alokasi_Umum'].median()
    
    df_jawa_2_dan_3 = pd.concat([df_jawa2, df_jawa3])
    median_jawa_2_dan_3 = df_jawa_2_dan_3['Dana_Alokasi_Umum'].median()

    df.at[156,'Dana_Alokasi_Umum'] = median_jawa_2_dan_3
    df.at[160,'Dana_Alokasi_Umum'] = median_jawa_2_dan_3
    df.at[166,'Dana_Alokasi_Umum'] = median_jawa_2_dan_3
    df.at[168,'Dana_Alokasi_Umum'] = median_jawa_2_dan_3
    df.at[176,'Dana_Alokasi_Umum'] = median_jawa_2_dan_3
    df.at[192,'Dana_Alokasi_Umum'] = median_jawa_2_dan_3

    print(df.loc[(df.Dana_Alokasi_Umum == 0)])

    # **ZERO JUMLAH PENDUDUK BEKERJA**
    print("FILL ZERO JUMLAH PENDUDUK BEKERJA ...")
    print(df.loc[(df.Jumlah_Penduduk_Bekerja == 0)])

    jumlah_penduduk_bekerja_bengkulu = 1002160
    df.at[3,'Jumlah_Penduduk_Bekerja'] = jumlah_penduduk_bekerja_bengkulu
    df.at[7,'Jumlah_Penduduk_Bekerja'] = jumlah_penduduk_bekerja_bengkulu
    df.at[17,'Jumlah_Penduduk_Bekerja'] = jumlah_penduduk_bekerja_bengkulu
    df.at[24,'Jumlah_Penduduk_Bekerja'] = jumlah_penduduk_bekerja_bengkulu
    df.at[33,'Jumlah_Penduduk_Bekerja'] = jumlah_penduduk_bekerja_bengkulu
    df.at[46,'Jumlah_Penduduk_Bekerja'] = jumlah_penduduk_bekerja_bengkulu
    df.at[49,'Jumlah_Penduduk_Bekerja'] = jumlah_penduduk_bekerja_bengkulu
    df.at[54,'Jumlah_Penduduk_Bekerja'] = jumlah_penduduk_bekerja_bengkulu
    df.at[56,'Jumlah_Penduduk_Bekerja'] = jumlah_penduduk_bekerja_bengkulu
    df.at[62,'Jumlah_Penduduk_Bekerja'] = jumlah_penduduk_bekerja_bengkulu

    print(df.loc[(df.Jumlah_Penduduk_Bekerja == 0)])
    for col in df.columns:
        pct_zero = (df[col] == 0).sum()
        print('{} - {}'.format(col,pct_zero))

    df.to_csv('2020cleaned.csv')
    return df

def run_eda(df):
    dfint=df.select_dtypes(include='int64')
    dffloat=df.select_dtypes(include='float64')

    # EDA
    for column in dfint:
        plt.figure()
        dfint.boxplot([column])

    for column in dffloat:
        plt.figure()
        dffloat.boxplot([column])

    sns.distplot(df['PDRB'] , fit=norm);
    (mu, sigma) = norm.fit(df['PDRB'])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    #Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title('PDRB')

    df['LOG_PDRB'] = np.log1p(df['PDRB'])

    sns.distplot(df['LOG_PDRB'] , fit=norm);
    (mu, sigma) = norm.fit(df['LOG_PDRB'])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    #Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title('LOG PDRB')

def convert_df_to_log(df):
    df['log_PDRB'] = np.log1p(df['PDRB']) 
    df['log_PDRB_Per_Kapita'] = np.log1p(df['PDRB_Per_Kapita'])
    df['log_Indeks_Pembangunan_Manusia'] = np.log1p(df['Indeks_Pembangunan_Manusia'])
    df['log_Dana_Alokasi_Umum'] = np.log1p(df['Dana_Alokasi_Umum'])
    df['log_Pengeluaran_Riil_per_Kapita_ per_Tahun'] = np.log1p(df['Pengeluaran_Riil_per_Kapita_ per_Tahun'])
    df['log_Nilai_UMR'] = np.log1p(df['Nilai_UMR'])
    df['log_Jumlah_Penduduk_Miskin'] = np.log1p(df['Jumlah_Penduduk_Miskin'])
    df['log_Jumlah_Penduduk_Bekerja'] = np.log1p(df['Jumlah_Penduduk_Bekerja'])
    df['log_Pengguna_Internet'] = np.log1p(df['Pengguna_Internet'])
    df['log_Pemilik_Ponsel'] = np.log1p(df['Pemilik_Ponsel'])
    df['log_Pengguna_Ponsel'] = np.log1p(df['Pengguna_Ponsel'])
    df['log_Jumlah_Penduduk'] = np.log1p(df['Jumlah_Penduduk'])  

    df["Area_encode"] = df.Area.map({'Area 1':1, 'Area 2':2, 'Area 3':3, 'Area 4':4, 'Area 5':5})
    df["Regional_Encode"] = df.Regional.map({'Sumbagsel':1, 'Lampung':2, 'Sumbagut':3, 'Area 4':4, 'Sumbagteng':5, 'Jabar':6, 'Jabo Inner':7, 'Jabo Outer':8, 'Jatim':9, 'Jateng': 10, 'Sulawesi':11, 'Kalimantan':12, 'Malpua':13, 'Balnus':14})
    df.to_csv('2020cleanedlog.csv')
    return df

def handle_duplicate_data(df):
    key = ['PDRB', 'PDRB_Per_Kapita', 'Indeks_Pembangunan_Manusia', 'Jumlah_Penduduk', 'Dana_Alokasi_Umum', 'Pengeluaran_Riil_per_Kapita_ per_Tahun', 'Nilai_UMR', 'Jumlah_Penduduk_Miskin', 'Jumlah_Penduduk_Bekerja', 'Pengguna_Internet', 'Pemilik_Ponsel', 'Pengguna_Ponsel', 'Jumlah_Agen_Pulsa']
    df_dedupped2 = df.drop_duplicates(subset=key)
    print(df.shape)
    print(df_dedupped2.shape)
    df = df.drop_duplicates(subset=key)
    df[['Kelurahan', 'Desa']] = df['Jumlah_Kelurahan_Desa'].str.split('/', expand=True)
    df['Kelurahan'] = df.Kelurahan.replace("-", np.nan)
    df['Desa'] = df.Desa.replace("-", np.nan)
    df.Desa.fillna(value=np.nan, inplace=True)
    df['Kelurahan'] = df['Kelurahan'].fillna(0)
    df['Desa'] = df['Desa'].fillna(0)

    #mengganti menjadi numeric agar bisa diolah
    df['Kelurahan'] = pd.to_numeric(df['Kelurahan'], errors='coerce')
    df['Desa'] = pd.to_numeric(df['Desa'], errors='coerce')
    # dfkabupatenkota = df.groupby(['Kota_Kabupaten']).sum()

    return df


if __name__ == "__main__":
    df = clean_data()
    # run_eda(df)
    df = convert_df_to_log(df)

    # handle duplicate data
    

    X = df[['log_PDRB_Per_Kapita', 'log_Indeks_Pembangunan_Manusia', 'log_PDRB', 'log_Jumlah_Penduduk', 'log_Dana_Alokasi_Umum', 'log_Pengeluaran_Riil_per_Kapita_ per_Tahun', 'log_Nilai_UMR', 'log_Jumlah_Penduduk_Miskin', 'log_Jumlah_Penduduk_Bekerja', 'log_Pengguna_Internet', 'log_Pemilik_Ponsel', 'log_Pengguna_Ponsel', 'Area_encode', 'Regional_Encode']]
    y = df['Jumlah_Agen_Pulsa']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = LinearRegression()
    clf = model.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    clf.score(X_test, y_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    importance = clf.coef_
    
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))

    # pyplot.bar([x for x in range(len(importance))], importance)
    # pyplot.show()

    # Perform lasso reggresion in x_train and y to find feature importance
    lasso=Lasso(alpha=0.001)
    modellasso = lasso.fit(X_train,y_train)
    y_predlasso = modellasso.predict(X_test)
    modellasso.score(X_test, y_test)
    
    FI_lasso = pd.DataFrame({"Feature Importance":lasso.coef_}, index=X_train.columns)
    FI_lasso.sort_values("Feature Importance",ascending=False)
    FI_lasso[FI_lasso["Feature Importance"]!=0].sort_values("Feature Importance").plot(kind="barh",figsize=(15,25))
    # plt.xticks(rotation=90)
    # plt.show()

    # DECISION TREE REGRESSOR
    set_config(print_changed_only=False) 
    dtr = DecisionTreeRegressor(random_state = 42)
    print(dtr)
    dtr.fit(X_train, y_train)

    score = dtr.score(X_test, y_test)
    print("R-squared:", score) 

    ypred = dtr.predict(X_test)
    mse = mean_squared_error(y_test, ypred)

    print("MSE: ", mse)
    print("RMSE: ", mse**(1/2.0)) 

    # x_ax = range(len(y_test))
    # plt.plot(x_ax, y_test, linewidth=1, label="original")
    # plt.plot(x_ax, ypred, linewidth=1.1, label="predicted")
    # plt.title("y-test and y-predicted data")
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.legend(loc='best',fancybox=True, shadow=True)
    # plt.grid(True)
    # plt.show() 

    # # **FEATURE SELECTION DECISION TREE REGRESSOR**
    sfs1 = sfs(dtr, k_features=(4, 14),
          forward=True, 
          floating=False, 
          scoring='r2',
          )
        
    sfs2 = sfs(dtr, k_features=(4,14),
          forward=False, 
          floating=False, 
          scoring='r2',
          )

    sfs1 = sfs1.fit(X_train, y_train)

    feat_names = list(sfs1.k_feature_names_)
    print(feat_names)

    print(sfs1.k_score_)

    print('best combination (ACC: %.3f): %s\n' % (sfs1.k_score_, sfs1.k_feature_idx_))
    print('all subsets:\n', sfs1.subsets_)

    print('Selected features:', sfs1.k_feature_idx_)

    # Generate the new subsets based on the selected features
    # Note that the transform call is equivalent to
    # X_train[:, sfs1.k_feature_idx_]

    X_train_sfs = sfs1.transform(X_train)
    X_test_sfs = sfs1.transform(X_test)

    # Fit the estimator using the new feature subset
    # and make a prediction on the test data
    dtr.fit(X_train_sfs, y_train)

    y_pred = dtr.predict(X_test_sfs)

    # Compute the accuracy of the prediction
    score = dtr.score(X_test_sfs, y_test)
    print("R-squared:", score) 

    # fig = plot_sfs(sfs1.get_metric_dict(), kind='std_err')
    # plt.title('Sequential Forward Selection (w. StdErr)')
    # plt.grid()
    # plt.show()

    # x_ax = range(len(y_test))
    # plt.plot(x_ax, y_test, linewidth=1, label="original")
    # plt.plot(x_ax, y_pred, linewidth=1.1, label="predicted")
    # plt.title("y-test and y-predicted data")
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.legend(loc='best',fancybox=True, shadow=True)
    # plt.grid(True)
    # plt.show() 

    sfs2 = sfs2.fit(X_train, y_train)
    feat_names = list(sfs2.k_feature_names_)
    print(feat_names)

    print(sfs2.k_score_)

    X_train_sfs2 = sfs2.transform(X_train)
    X_test_sfs2 = sfs2.transform(X_test)
    dtr.fit(X_train_sfs2, y_train)

    y_pred2 = dtr.predict(X_test_sfs2)

    # Compute the accuracy of the prediction
    score = dtr.score(X_test_sfs2, y_test)
    print("R-squared:", score) 


    print('done')