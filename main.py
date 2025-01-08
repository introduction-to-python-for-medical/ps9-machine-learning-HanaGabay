import pandas as pd
df = pd.read_csv('parkinsons.csv')
features = ['HNR', 'DFA']
target = 'status'
x = df[features]
y = df[target]
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x = scaler.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.svm import SVC
svm_model = SVC(kernel='linear') 
svm_model.fit(x_train, y_train)


y_pred = svm_model.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
