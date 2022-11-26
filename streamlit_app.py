#importing general objects
import pandas as pd
import plotly
import plotly.express as px
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from PIL import Image
import collections

#TODO: reading in the dataset
df = pd.read_csv("exoplanet_data.csv")
del df['rowid']
df.dropna(axis=1, how="any", inplace=True)
df = df.select_dtypes(['number'])

kmeans_df = pd.read_csv("exoplanet_data.csv")

scat_df = pd.read_csv("exoplanet_data.csv")
scat_df.drop([3585], inplace=True)

#Some basic commands in streamlit -- you can find an amazing cheat sheet here: https://docs.streamlit.io/library/cheatsheet
st.title('Exoplanet Data EDA')
st.write('This dashboard analyzes exoplanet data from http://exoplanetarchive.ipac.caltech.edu. Here we attempt to analyze habitable exoplanets that are capable of supporting life within the dataset and make several observations about stellar phenomena.')
st.markdown("""---""")
#generate random data for my example dataframe -- howto: https://stackoverflow.com/questions/32752292/how-to-create-a-dataframe-of-random-integers-with-pandas
# example_data = pd.DataFrame(np.random.randint(0,100,size=(150, 4)), columns=list('ABCD'))

#show off a bit of your data. 
st.header('The Data')
col1, col2 = st.columns(2) #here is how you can use columns in streamlit. 
col1.dataframe(kmeans_df.head())
col2.markdown('This dataset provides information on over 3,800 exoplanets, giving values such as the orbital period, the stellar mass, the planet radius, the planet density, and more.') #you can add multiple items to each column.
col2.markdown('- Notice how a lot of the data is NaN values, likely because radial velocity imaging reduces in accuracy over distance.')
st.markdown("""---""")

st.header('Correlation Matrix')
col1, col2 = st.columns(2)
col1.plotly_chart(px.imshow(df.corr(method = "pearson", min_periods=3500)))
col2.markdown(" ")
col2.markdown(" ")
col2.markdown(" ")
col2.markdown(" ")
col2.markdown("After dropping columns with missing values, a correlation was run on the rest of the data. **Significant** correlations were found between the following: ")
col2.markdown(" 1. RA (right ascension), st_elon (elliptic longitude)")
col2.markdown(" 2. st_glon (galactic longitude), st_elat (elliptic lat)")
col2.markdown(" 3. st_elat (elliptic lat), dec (decimal degrees)")
col2.markdown("\n Each of these correlations were above 0.8 and make sense from an astrophysics perspective.")
st.markdown("""---""")

st.header('Regression Analysis')
col1, col2, col3 = st.columns(3)

# sns.lmplot(x='st_logg',y='pl_orbper',data=scat_df, fit_reg=True, line_kws={'color': 'red'})
# mpl_fig = plt.gcf()
# plotly_fig = plotly.tools.mpl_to_plotly(mpl_fig)
# col1.plotly_chart(mpl_fig)
image = Image.open("logg_orbper.png")
col1.image(image)
col1.markdown("As the stellar surface gravity increases, the orbital period decreases.")

# sns.lmplot(x='pl_orbeccen',y='pl_bmassj',data=scat_df, fit_reg=True, line_kws={'color': 'red'}) 
# mpl_fig = plt.gcf()
# plotly_fig = plotly.tools.mpl_to_plotly(mpl_fig)
# col2.plotly_chart(mpl_fig)
image = Image.open("eccen_mass.png")
col2.image(image)
col2.markdown("As the planet's mass increases, the eccentricity of the orbit of the planet increases.")

# sns.lmplot(x='pl_orbper',y='st_mass',data=scat_df, fit_reg=True, line_kws={'color': 'red'}, ci=None, robust=True) 
# mpl_fig = plt.gcf()
# plotly_fig = plotly.tools.mpl_to_plotly(mpl_fig)
# col3.plotly_chart(mpl_fig)
image = Image.open("orbper_mass.png")
col3.image(image)
col3.markdown("As the stellar mass increases, the orbital period increases.")
st.markdown("""---""")

st.header('K-Means Analysis')
col1, col2 = st.columns(2)

kmeans_df = kmeans_df.select_dtypes(['number'])
del kmeans_df['rowid']
kmeans_df = kmeans_df.fillna(kmeans_df.mean())
kmeans_df.dropna(axis=1, inplace=True)
kmeans_np = kmeans_df.to_numpy()

# X = kmeans_np
# Sc = StandardScaler()
# X = Sc.fit_transform(kmeans_df)
# pca = PCA(2) 
# pca_data = pd.DataFrame(pca.fit_transform(X),columns=['PC1','PC2']) 
# kmeans = KMeans(n_clusters=6).fit(X)
# pca_data['cluster'] = pd.Categorical(kmeans.labels_)
# g = sns.scatterplot(x="PC1",y="PC2",hue="cluster",data=pca_data)
# g.legend_.remove()

# mpl_fig = plt.gcf()
# plotly_fig = plotly.tools.mpl_to_plotly(mpl_fig)
# col1.plotly_chart(mpl_fig)
image = Image.open("pcaplot.png")
col1.image(image)
col1.markdown("This shows a k-means clustering on a PCA graph after mean-imputing the numerical columns.")

# counter = collections.Counter(list(kmeans.labels_))
# cd = dict(counter)
# D = collections.OrderedDict(sorted(cd.items()))
# plt.bar(range(len(D)), list(D.values()), align='center')
# plt.xticks(range(len(D)), list(D.keys()))

# mpl_fig = plt.gcf()
# plotly_fig = plotly.tools.mpl_to_plotly(mpl_fig)
# col2.plotly_chart(mpl_fig)
image = Image.open("elbowcurve.png")
col2.image(image)
col2.markdown("This shows why 6 clusters was the optimal choice for the k-means algorithm.")
st.markdown("""---""")

# col1.plotly_chart(px.imshow ))

#Always good to section out your code for readability.
st.header('Conclusions')
st.markdown('- First, planets in the same k-means cluster as Earth were looked at. Then, after comprehensive filtering based on spectral type, stellar distance, stellar mass, orbital period, and orbital eccentricity, five potentially habitable planets were found: HD 19994, HD 27442, HR 810, HD 114783, and HD 160691.')
st.markdown('- Various stellar phenomena regarding stellar surface gravity, planet mass, orbital eccentrictity, and orbital period have been experimentally verified.')
st.markdown('- Outliers and missing values greatly skewed regression and clustering analysis, meaning removal of these values is critical for accurate conclusions.')
st.markdown("""---""")

st.header('Further Steps')
st.markdown('- Run feature selection algorithms to reduce the dimensionality of the dataset and increase the accuracy of future machine learning analysis.')
st.markdown('- Identify other phenomena with Pearson correlation analysis and cross-reference this data with observed astrophysical patterns.')