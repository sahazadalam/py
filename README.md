import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load the iris data set
df=pd.read_csv("iris.csv")
print(df)

#separate the features and target
X=df.drop('Species',axis=1)
y=df['Species']
print(X)
print(y)

#preprocessing by StandardScaler and display
scaler=StandardScaler()
X_Data=scaler.fit_transform(X)
print("")
print(pd.DataFrame(X_Data))
print("")

#split the data to train and test data
X_train,X_test,y_train,y_test=train_test_split(X_Data,y,test_size=0.3,random_state=42)
print("-----------X train data --------------")
print(pd.DataFrame(X_train))
print("")
print("-----------X train test --------------")
print(pd.DataFrame(X_test))
print("")
print("-----------y train data --------------")
print(pd.DataFrame(y_train))
print("")
print("-----------y train test --------------")
print(pd.DataFrame(y_test))








#Write a program to learn how to select features for machine learning
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1. Univariate Feature Selection
print("Univariate Feature Selection:")
# Apply SelectKBest
selector = SelectKBest(score_func=f_classif, k=2)
X_new = selector.fit_transform(X_train, y_train)
# Display scores and selected features
print("Feature scores:", selector.scores_)
print("Selected features:", X_train.columns[selector.get_support()])

# 2. Recursive Feature Elimination (RFE)
model = LogisticRegression()
print("\nRecursive Feature Elimination (RFE):")
rfe = RFE(estimator=model, n_features_to_select=2)
rfe.fit(X_train, y_train)
# Display selected features
print("Selected features:", X_train.columns[rfe.support_])


# 3. Feature Importance from Random Forest
print("\nFeature Importance from Random Forest:")
model = RandomForestClassifier()
model.fit(X_train, y_train)
# Display feature importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print(f"{X_train.columns[indices[f]]}: {importances[indices[f]]}")








Data Pre-processing using SimpleImputer, LabelEncoder, Standardization, Normalization

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# -----------------------------
# Step 1: Create Sample Dataset
# -----------------------------
data = {
    'Name': ['Ravi', 'Priya', 'Amit', 'Neha', np.nan, 'Suresh'],
    'Age': [25, 30, np.nan, 28, 22, 35],
    'Department': ['IT', 'HR', 'Finance', 'IT', 'Finance', 'HR'],
    'Salary': [50000, 60000, 58000, np.nan, 52000, 61000]
}

df = pd.DataFrame(data)
print("Original Dataset:\n", df)

# ---------------------------------------------
# Step 2: Find and display the count of missing values
# ------------------------------------------

print("Missing Data Count:\n")
print(df.isnull().sum())

# -----------------------------
# Step 2: Data Cleaning with SimpleImputer
# -----------------------------
# Replace missing numeric values with mean
imputer_num = SimpleImputer(strategy='mean')
df[['Age', 'Salary']] = imputer_num.fit_transform(df[['Age', 'Salary']])

# Replace missing categorical values with most frequent
imputer_cat = SimpleImputer(strategy='most_frequent')
df[['Name']] = imputer_cat.fit_transform(df[['Name']])

print("\nAfter Data Cleaning (SimpleImputer):\n", df)

# -----------------------------
# Step 3: Data Transformation (Label Encoding)
# -----------------------------
le = LabelEncoder()
df['Department_encoded'] = le.fit_transform(df['Department'])

print("\nAfter Label Encoding:\n", df)

# -----------------------------
# Step 4: Standardization
# -----------------------------
scaler = StandardScaler()
df[['Age_std', 'Salary_std']] = scaler.fit_transform(df[['Age', 'Salary']])

print("\nAfter Standardization:\n", df)

# -----------------------------
# Step 5: Normalization
# -----------------------------
minmax = MinMaxScaler()
df[['Age_norm', 'Salary_norm']] = minmax.fit_transform(df[['Age', 'Salary']])

print("\nAfter Normalization:\n", df)







import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# 1. Load Dataset (Iris dataset)
iris = load_iris()
X = iris.data
y = iris.target

# Convert to Binary Classification: Is the flower 'setosa' (class 0) or not?
y_binary = (y == 0).astype(int)

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

# 3. Build and Train KNN Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 4. Predict
y_pred = knn.predict(X_test)

# 5. Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)

# Lift Measure = Precision / (Baseline Rate of Positive Class)
baseline_rate = np.mean(y_test)  # proportion of positives in test set
lift = precision / baseline_rate if baseline_rate > 0 else 0

# 6. Print Results
print("KNN Classification Results:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"Lift     : {lift:.4f}")

# 7. Optional: Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))











import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Load dataset (Iris dataset for demonstration)
data = load_iris()
X, y = data.data, data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier()
}

# Evaluate classifiers
results = []

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    results.append({
        "Classifier": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='macro'),
        "Recall": recall_score(y_test, y_pred, average='macro'),
        "F1-Score": f1_score(y_test, y_pred, average='macro')
    })

# Convert results to DataFrame
df_results = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)

# Display results
print("\n=== Classifier Performance Comparison ===")
print(df_results.to_string(index=False))







mport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load Dataset
data = pd.read_csv("Breast Cancer Wisconsin.csv")
print("Dataset shape:", data.shape)
print("\nPreview:\n", data.head())

# 2. Drop ID or non-numeric columns
if 'id' in data.columns:
    data = data.drop(['id'], axis=1)

# Drop rows with missing values
data = data.dropna()

# 3. Select numeric columns
X = data.select_dtypes(include=[np.number])

# 4. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Apply K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# 6. Evaluate clustering performance
sil_score = silhouette_score(X_scaled, labels)
print(f"\nSilhouette Score: {sil_score:.4f}")

# 7. Add cluster labels to dataset
data['Cluster'] = labels
print("\nClustered Data Sample:\n", data.head())

# 8. Visualize clusters (first 2 features)
plt.figure(figsize=(7,5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='coolwarm', s=40, edgecolor='k')
plt.title("K-Means Clustering on Breast Cancer Dataset")
plt.xlabel(f"{X.columns[0]} (standardized)")
plt.ylabel(f"{X.columns[1]} (standardized)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# 9. Display cluster centers
print("\nCluster Centers (Standardized):")
centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
print(centers_df)