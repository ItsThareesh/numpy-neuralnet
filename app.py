# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from planar_utils import load_planar_dataset, load_extra_datasets, plot_decision_boundary
from model import build_model, predict

st.session_state["Model_Trained"] = False

st.title("🧠 Explore Neural Networks in Action!")

st.markdown("""
Dive into how a simple 2-layer neural network learns to classify non-linearly separable 2D data.

- 🔍 Choose different datasets (moons, circles, blobs, and more)
- 🎛️ Tune hyperparameters like hidden units and learning rate

Perfect for beginners trying to demystify backpropagation and decision boundaries.

""")

# 1. Dataset choice
st.sidebar.header("Dataset Settings")
dataset_choice = st.sidebar.selectbox("Select a dataset",
                                      ["planar_flower_data", "noisy_circles", "noisy_moons", "blobs", "gaussian_quantiles"])

# 2. Model hyperparameters
st.sidebar.header("Model Hyperparameters")
hidden_units = st.sidebar.slider("Hidden Layer Units", 1, 128, 64)
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 5.0, 0.1, step=0.01)
iterations = st.sidebar.slider("Training Iterations", 500, 10000, 5000, step=500)

# Load data
if dataset_choice == "planar_flower_data":
    X, Y = load_planar_dataset()
else:
    dataset = load_extra_datasets()
    X, Y = dataset[dataset_choice]

X = X.reshape(2, -1)
Y = Y.reshape(1, -1)

# Plot dataset
st.subheader("Selected Dataset")
fig, ax = plt.subplots()
ax.scatter(X[0, :], X[1, :], c=Y.flatten(), s=40, cmap='coolwarm')
st.pyplot(fig)

st.markdown("<br>", unsafe_allow_html=True)

_, col2, _ = st.columns([1, 1, 1])
with col2:
    if st.button("Train Model"):
        with st.spinner("Training in progress..."):
            parameters = build_model(X, Y, hidden_units, learning_rate, iterations)
            st.session_state["Model_Trained"] = True


if st.session_state.get("Model_Trained", False):
    predictions = predict(parameters, X)
    acc = np.mean(predictions == Y) * 100
    st.success(f"✅ Accuracy: {acc:.2f}%")

    # Plot decision boundary
    fig2 = plt.figure()
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    st.pyplot(fig2)
