
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
%load_ext autoreload
%autoreload 2

# Download the data from your GitHub repository
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv

df = pd.read_csv('parkinsons.csv')

features = ['PPE', 'RPDE','DFA']
target = 'status'
x = df[features]
y = df[target]

scaler = MinMaxScaler()
x = scaler.fit_transform(x)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) # Example test_size and random_state


svm_model = SVC(kernel='linear') # You can change the kernel (e.g., 'rbf', 'poly')
svm_model.fit(x_train, y_train)


y_pred = svm_model.predict(x_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
