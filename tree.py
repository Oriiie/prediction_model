import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt

file_name = 'testtask2.xlsx'

df = pd.read_excel(io=file_name, sheet_name=0)

columns = df.columns[1:]

X = df[columns]
Y = df.target_flag

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

clf = DecisionTreeClassifier(criterion="log_loss", max_depth=7)

clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Точность:", metrics.accuracy_score(y_test, y_pred))

'''plt.figure(figsize=(30, 10))

a = tree.plot_tree(clf,
                   feature_names=columns,
                   fontsize=14)
plt.show()'''

f = open('res.txt', 'w')

for i in range(len(y_pred)):
    f.write('Шанс дефолта - ' + str(round(100 * clf.predict_proba(X_test[i:i + 1]).tolist()[0][0], 3)) + '% ')

    credit_score = 300 + clf.predict_proba(X_test[i:i + 1]).tolist()[0][1] * 550
    if credit_score < 560:
        result = 'Very Bad'
    elif 560 < credit_score < 650:
        result = 'Bad'
    elif 650 < credit_score < 700:
        result = 'Slightly Good'
    elif 700 < credit_score < 780:
        result = 'Good'
    else:
        result = 'Very Good'

    f.write('; Скоринг: ' + str(round(credit_score)) + '; Оценка: ' + result + '\n')

f.close()

parameter_list = []
WOE_IV = []

for parameter in range(len(df.columns)-1):
    columns = df.columns[parameter+1]

    X = df[columns].tolist()
    Y = df.target_flag.tolist()

    temp_len = (max(X) - min(X)) / 5
    temp_start = min(X)

    hold_counter = []
    hold_good = []
    hold_bad = []

    for i in range(5):
        counter = 0
        good = 0
        bad = 0

        for j in range(len(X)):
            if temp_start <= X[j] <= temp_start+temp_len:
                counter += 1
                if Y[j] == 'good':
                    good += 1
                else:
                    bad += 1

        temp_start += temp_len

        hold_counter.append(counter)
        hold_good.append(good)
        hold_bad.append(bad)

    event_percent = [100*i/sum(hold_good) for i in hold_good]
    non_event_percent = [100*i/sum(hold_bad) for i in hold_bad]

    WOE = []
    event_difference = [event_percent[i] - non_event_percent[i] for i in range(5)]

    for i in range(5):
        if hold_good[i] == 0:
            WOE.append(-1.5)
        elif hold_bad[i] == 0:
            WOE.append(1.5)
        else:
            WOE.append(math.log(event_percent[i]/non_event_percent[i]))

    IV = [WOE[i] * event_difference[i] for i in range(5)]

    parameter_list.append(columns)
    WOE_IV.append(sum(IV))

f = open('woe_iv.txt', 'w')

for i in range(len(parameter_list)):
    f.write(parameter_list[i] + ' имеет IV = ' + str(round(WOE_IV[i], 3)) + '\n')

f.close()

useless_list = []
weak_list = []
medium_list = []
strong_list = []
TGTBT_list = []

for i in range(len(parameter_list)):
    if WOE_IV[i] < 0.02:
        useless_list.append(parameter_list[i])
    elif 0.02 < WOE_IV[i] < 0.1:
        weak_list.append(parameter_list[i])
    elif 0.1 < WOE_IV[i] < 0.3:
        medium_list.append(parameter_list[i])
    elif 0.3 < WOE_IV[i] < 0.5:
        strong_list.append(parameter_list[i])
    else:
        TGTBT_list.append(parameter_list[i])

f = open('woe_sorted.txt', 'w')

f.write('Useless: ' + '\n\n')
for i in range(len(useless_list)):
    f.write(useless_list[i] + '\n')

f.write('\n\n\n' + 'Weak: ' + '\n\n')
for i in range(len(weak_list)):
    f.write(weak_list[i] + '\n')

f.write('\n\n\n' + 'Medium: ' + '\n\n')
for i in range(len(medium_list)):
    f.write(medium_list[i] + '\n')

f.write('\n\n\n' + 'Strong: ' + '\n\n')
for i in range(len(strong_list)):
    f.write(strong_list[i] + '\n')

f.write('\n\n\n' + 'Too good to be true: ' + '\n\n')
for i in range(len(TGTBT_list)):
    f.write(TGTBT_list[i] + '\n')

f.close()
