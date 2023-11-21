import pandas as pd
df = pd.read_csv("train.csv")

df['Age'].fillna(df['Age'].mean(), inplace=True)

df.dropna(subset=['Embarked'], inplace=True)

df = pd.get_dummies(data=df, columns=['Sex'], drop_first=True)

df = pd.get_dummies(data=df, columns=['Embarked'])

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

from sklearn.model_selection import train_test_split

X = df[['Pclass','Sex_male', 'Age','SibSp','Parch', 'Fare', 'Embarked_C' , 'Embarked_Q', 'Embarked_S']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report

confusion_matrix(y_test, predictions)

pc = int(input("Enter Passenger Class : "))
sex = int(input("Enter sex (1 for Male/0 for Female) : "))
age = int(input("Enter age : "))

person = {
    'Pclass': pc,
    'Sex_male': sex,
    'Age': age,
    'SibSp': 2,
    'Parch': 4,
    'Fare': 200,
    'Embarked_C':0,
    'Embarked_Q':1,
    'Embarked_S':0
}


person_df = pd.DataFrame([person])

if model.predict(person_df)[0] == 0:
    print("Not Survived !")
else:
    print("Survived !")

