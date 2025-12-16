import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Inspect data
print(df.head())
print(df.describe())

# Histograms
df.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

# Scatter plot
plt.scatter(df["sepal length (cm)"], df["petal length (cm)"])
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Sepal Length vs Petal Length")
plt.show()

# Correlation scatter plot (with regression line)
plt.figure(figsize=(6, 4))
sns.regplot(
    x="sepal length (cm)",
    y="petal length (cm)",
    data=df
)
plt.title("Correlation Between Sepal Length and Petal Length")
plt.tight_layout()

# Save figure
plt.savefig("correlation.png")
plt.show()
