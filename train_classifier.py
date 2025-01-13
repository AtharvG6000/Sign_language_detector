import pickle


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))


from itertools import zip_longest
# Define the desired fixed length (e.g., the length of the longest sequence)
fixed_length = max(len(seq) for seq in data_dict['data'])
# Pad or truncate each sequence to the fixed length
padded_data = [
    seq[:fixed_length] + [0] * (fixed_length - len(seq))  # Truncate or pad with 0
    for seq in data_dict['data']
]
# Convert to NumPy array
data = np.asarray(padded_data)


# data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])


x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
