# Import Required Packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Loading and examining the dataset
penguins_df = pd.read_csv("penguins.csv")
penguins_df.head()
penguins_df.info()
penguins_df.boxplot()  
plt.show()
penguins_clean = penguins_df.dropna()
print(penguins_clean[penguins_clean['flipper_length_mm']>4000])
print(penguins_clean[penguins_clean['flipper_length_mm']<0])  #24/2
penguins_clean = penguins_clean.drop([9,14])
df=pd.get_dummies(penguins_clean).drop('sex_.',axis=1)
print(df.head())
scaler=StandardScaler()
x=scaler.fit_transform(df)
print(x)
penguins_preprocessed = pd.DataFrame(data=x,columns=df.columns)
penguins_preprocessed.head() # 25/2
pca =PCA(n_components=None)
dfx_pca=pca.fit(penguins_preprocessed)
explained_variance_ratio = dfx_pca.explained_variance_ratio_
print(explained_variance_ratio)
n_components=sum(dfx_pca.explained_variance_ratio_>0.1)
pca = PCA(n_components=n_components)
penguins_PCA = pca.fit_transform(penguins_preprocessed)
print(penguins_PCA)
inertia = []

for k in range (1,19):
    kmeans=KMeans(n_clusters=k,random_state=42).fit(penguins_PCA)
    inertia.append(kmeans.inertia_)
    
plt.figure(figsize=(8, 6))
plt.plot(range(1, 19), inertia, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()       # KASPER_AS
n_clusters=4
kmeans=KMeans(n_clusters=n_clusters,random_state=42).fit(penguins_PCA)

plt.scatter(penguins_PCA[:, 0], penguins_PCA[:, 1], c=kmeans.labels_, cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title(f'K-means Clustering (K={n_clusters})')
plt.legend()
plt.show()

penguins_clean ['label']=kmeans.labels_
numeric_columns = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm','label']
stat_penguins = penguins_clean[numeric_columns].groupby('label').mean()
stat_penguins
