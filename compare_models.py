import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

# Models to compare
models = {
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# Evaluate each model
results = {}
for name, model in models.items():
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_predict),
        "Precision": precision_score(y_test, y_predict, average='weighted'),
        "Recall": recall_score(y_test, y_predict, average='weighted'),
        "F1 Score": f1_score(y_test, y_predict, average='weighted')
    }

# Prepare data for plotting
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
x = range(len(metrics))  # Indices for metrics
model_names = list(results.keys())
metric_values = {model: [results[model][metric] for metric in metrics] for model in model_names}

# Plot the metrics
plt.figure(figsize=(10, 6))

for model in model_names:
    plt.plot(x, metric_values[model], marker='o', label=model)

# Customize the plot
plt.xticks(x, metrics)  # Set x-axis labels to metrics
plt.title("Model Performance Comparison")
plt.xlabel("Metrics")
plt.ylabel("Scores")
plt.ylim(0, 1)  # Scores range from 0 to 1
plt.grid(alpha=0.3)
plt.legend(title="Models")
plt.tight_layout()

# Show plot
plt.show()
w