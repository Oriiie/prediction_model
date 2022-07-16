import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


file_name = 'testtask2.xlsx'

df = pd.read_excel(io=file_name, sheet_name=0)

columns = df.columns[1:]

X = df[columns]
Y = df.target_flag

os = SMOTE(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

os_data_X, os_data_y = os.fit_resample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
os_data_y = pd.DataFrame(data=os_data_y, columns=['target_flag'])

logreg = LogisticRegression(max_iter=2500, solver='lbfgs')
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Точность логистической регрессии: {:.5f}'.format(logreg.score(X_test, y_test)))

confusion_matrix = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

sensitivity = confusion_matrix[0][0] / (confusion_matrix[1][0] + confusion_matrix[0][0])
specificity = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])
false_positive_rate = 1 - specificity

print('Чувствительность: ', sensitivity)
print('Специфичность: ', specificity)
print('False positive rate: ', false_positive_rate)
