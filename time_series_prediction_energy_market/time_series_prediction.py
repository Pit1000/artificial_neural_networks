import pandas as pd # biblioteka do działań na danych w postaci tabelarycznej
import matplotlib.pyplot as plt
import seaborn as sns # druga biblioteka do wykresów
import tensorflow as tf

df_train = pd.read_csv('rdn_train.csv', sep=',')
df_test = pd.read_csv('rdn_test.csv', sep=',')
df_train.head()
df_test.head()

# formatowanie dat
df_train['date_hour'] = df_train['date'] + ' ' + df_train['hour'].astype(str).str.pad(width=2, fillchar='0')
df_train['date_hour'] = pd.to_datetime(df_train['date_hour'], format="%Y-%m-%d %H")
df_train.set_index('date_hour', inplace=True)

#wyekstrachowanie ważnych danych
df_train['date'] = pd.to_datetime(df_train['date'])
df_train['DzienTygodnia'] = df_train['date'].dt.dayofweek

df_train['date'] = pd.to_datetime(df_train['date'])
df_train['msc'] = df_train['date'].dt.month

df_test['date'] = pd.to_datetime(df_test['date'])
df_test['DzienTygodnia'] = df_test['date'].dt.dayofweek

df_test['date'] = pd.to_datetime(df_test['date'])
df_test['msc'] = df_test['date'].dt.month

df_test['E1'] = pd.to_numeric(((df_test['Generacja JWCD'] / df_test['Generacja nJWCD'].values)).round(2) * 100)
df_train['E1'] = pd.to_numeric(((df_train['Generacja JWCD'] / df_train['Generacja nJWCD'].values)).round(2) * 100)

df_test['delta'] = pd.to_numeric(((df_test['fixing_minus_1d'] - df_test['fixing_minus_2d'].values)).round(2) * 100)
df_train['delta'] = pd.to_numeric(((df_train['fixing_minus_1d'] - df_train['fixing_minus_2d'].values)).round(2) * 100)

#df_test['avg'] = df_test[['fixing_minus_4d','fixing_minus_5d','fixing_minus_6d']].mean()
#df_train['avg'] = df_train[['fixing_minus_4d','fixing_minus_5d','fixing_minus_6d']].mean()

df_train['avg'] = df_train.loc[:, ['fixing_minus_1d','fixing_minus_2d','fixing_minus_3d','fixing_minus_4d','fixing_minus_5d','fixing_minus_6d','fixing_minus_7d']].mean(axis=1).round(2)
df_test['avg'] = df_test.loc[:, ['fixing_minus_1d','fixing_minus_2d','fixing_minus_3d','fixing_minus_4d','fixing_minus_5d','fixing_minus_6d','fixing_minus_7d']].mean(axis=1).round(2)

df_test['E2'] =  pd.to_numeric(((df_test['Suma zdolności wytwórczych JWCD'] / df_test['Suma zdolności wytwórczych nJWCD'].values)).round(2) * 100)
df_train['E2'] = pd.to_numeric(((df_train['Suma zdolności wytwórczych JWCD'] / df_train['Suma zdolności wytwórczych nJWCD'].values)).round(2) * 100)

df_test.head()

plt.figure(figsize=(18,6))
plt.plot(df_train['fixing'])
plt.xticks(rotation=45);
#plt.show()

sns.relplot(data=df_train, x='hour', y='fixing');

# lista kolumn, które będą danymi wejściowymi
cols = ['DzienTygodnia','Generacja nJWCD','Generacja źródeł wiatrowych', 'Wymagana rezerwa mocy ponad zapotrzebowanie', 'hour','fixing_minus_1d','fixing_minus_2d','fixing_minus_3d','fixing_minus_4d','fixing_minus_5d','fixing_minus_6d','fixing_minus_7d', 'avg', 'msc']

y_train = df_train['fixing']
X_train = df_train[cols]
X_test = df_test[cols]

def Leaky_Relu(x):
    return tf.nn.leaky_relu(x, alpha=0.01)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(32, activation=Leaky_Relu, input_shape=(14,)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation=Leaky_Relu))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation=Leaky_Relu))
model.add(tf.keras.layers.Dense(1, activation=Leaky_Relu))

model.compile(optimizer='adam', loss='mean_squared_error', metrics = [tf.keras.metrics.RootMeanSquaredError(), 'mean_absolute_error'])

model.fit(X_train, y_train, epochs=40)

# predykcja modelu dla danych testowych
y_hat = model.predict(X_test)

# zapis wyniku dopliku csv
result = pd.DataFrame(y_hat, columns=['fixing'])
result.to_csv('prediction.csv', sep=',', index_label='nr')

y_hat_train = model.predict(X_train)
df_plot = pd.DataFrame(y_train)
df_plot['y_hat'] = y_hat_train
# wybór zakresu dat
df_plot = df_plot[df_plot.index < '02-01-2018']

#wykres porównawczy predykcji
plt.figure(figsize=(18, 6))
plt.plot(df_plot['fixing'], 'blue')
plt.plot(df_plot['y_hat'], 'orange')
plt.xticks(rotation=45);
plt.legend(['fixing', 'fixing z modelu'])
plt.show()