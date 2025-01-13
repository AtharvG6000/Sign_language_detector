import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Pad or truncate the sequences
fixed_length = max(len(seq) for seq in data_dict['data'])
padded_data = [
    seq[:fixed_length] + [0] * (fixed_length - len(seq)) 
    for seq in data_dict['data']
]
data = np.asarray(padded_data)
labels = np.asarray(data_dict['labels'])

# Split the data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# Predict and evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print('{}% of samples were classified correctly using Decision Tree!'.format(score * 100))

# Save the model
with open('decision_tree_model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
