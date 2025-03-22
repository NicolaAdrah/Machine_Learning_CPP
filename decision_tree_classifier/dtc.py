import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import dtc as decision_tree

# Load and prepare data (same as original)
url = "https://www.physi.uni-heidelberg.de/~reygers/lectures/2021/ml/data/heart.csv"
df = pd.read_csv(url)
df['target'] = (df['target'] == 1).astype(int)

# Convert data to C++-compatible format
X = df.drop('target', axis=1).values.tolist()  # Convert to list of lists
y = df['target'].values.astype(int).tolist()   # Convert to list of integers

# Split data (same as original)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# Train and evaluate (modified for C++ interface)
max_depth_values = [1, 2, 3, 4, 5, 6, 7, None]  # None means no max depth in C++ (-1)

train_acc = []
test_acc = []

for depth in max_depth_values:
    # Convert "None" to -1 (C++ uses -1 for unlimited depth)
    cpp_depth = depth if depth is not None else -1
    
    # Initialize and train classifier
    clf = decision_tree.DecisionTreeClassifier(max_depth=cpp_depth)
    clf.fit(X_train, y_train)
    
    # Predictions (C++ returns lists)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    
    # Calculate accuracy (using numpy for comparison)
    train_acc.append(np.mean(np.array(y_train) == np.array(train_pred)))
    test_acc.append(np.mean(np.array(y_test) == np.array(test_pred)))
    
    print(f"Depth: {str(depth).ljust(4)} | Train Acc = {train_acc[-1]:.10f} | Test Acc = {test_acc[-1]:.10f}")