import pandas as pd
import numpy as np
import streamlit as st

st.write("hello")
st.write("Hi")

pdver = pd.__version__
npver = np.__version__

st.write(pdver, npver)
