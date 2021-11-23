import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings('ignore')

df = pd.read_csv('Zomato_df.csv')

df.drop('Unnamed: 0', axis=1, inplace=True)
X = df.drop('rate', axis=1)
y = df['rate']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

ET_Model = ExtraTreesRegressor(n_estimators=120)
ET_Model.fit(x_train, y_train)

y_predict = ET_Model.predict(x_test)


pickle.dump(ET_Model, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
print(y_predict)
