# Homework 6, due Friday Week 7
# Author: Leo Cheung
# ID: 19421084

import numpy as np
import pandas as pd
import altair as alt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import streamlit as st

st.set_page_config(page_title = "HW6 Leo Cheung", page_icon = ":sunrise:")
st.title("How the number of iterations alters the K-Means clusters")

st.markdown("Author: Leo Cheung, [GitHub link](https://github.com/ctZN4)")

iters = st.slider(label = "Choose the number of iterations", min_value = 1, max_value = 100, value = 1, step = 1)

st.write("Notice how the clusters remain relatively consistent after a certain number of iterations")

X, _ = make_blobs(n_samples=1000, centers=5, n_features=2, random_state = 1)
df = pd.DataFrame(X, columns = list("ab"))
starting_points = np.array([[0,0],[-2,0],[-4,0],[0,2],[0,4]])
kmeans = KMeans(n_clusters = 5, max_iter=iters, init=starting_points, n_init = 1)
kmeans.fit(X);
df["c"] = kmeans.predict(X)
chart1 = alt.Chart(df).mark_circle().encode(
    x = "a",
    y = "b",
    color = "c:N"
)

df_centers = pd.DataFrame(kmeans.cluster_centers_, columns = list("ab"))

chart_centers = alt.Chart(df_centers).mark_point().encode(
    x = "a",
    y = "b",
    color = alt.value("black"),
    shape = alt.value("diamond"),
)

st.altair_chart(chart1 + chart_centers)