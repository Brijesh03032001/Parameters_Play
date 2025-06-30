import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.base import clone
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="ML Classifier Explorer",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Data Loading and Preprocessing ---
@st.cache_data
def load_data():
    """Loads the Iris dataset and splits it into training and testing sets."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        iris.feature_names,
        iris.target_names,
    )


X_train, X_test, y_train, y_test, feature_names, target_names = load_data()


# --- Visualization Functions ---
def plot_decision_boundary(X, y, clf, title):
    """Plots the decision boundary for a classifier."""
    # Reduce data to 2 dimensions using PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # We need to retrain the classifier on the 2D data for visualization
    clf.fit(X_pca, y)

    # Create a mesh to plot in
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

    # Plot the training points
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.coolwarm, s=40, edgecolor="k"
    )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title(title)
    legend = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend)

    st.pyplot(fig)


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plots a confusion matrix using seaborn."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    st.pyplot(fig)


# --- UI: Sidebar ---
st.sidebar.title("Classifier Settings")
st.sidebar.header("1. Choose a Classifier")

classifier_name = st.sidebar.radio(
    "Select a classifier:",
    ("K-Nearest Neighbors (KNN)", "Support Vector Machine (SVM)", "Random Forest"),
)

# --- UI: Main Panel ---
st.title("🤖 Interactive Machine Learning Classifier Explorer")
st.write(
    """
    This application allows you to experiment with different classification algorithms and their hyperparameters.
    Select a classifier and adjust its parameters in the sidebar. Then, click the 'Run Algorithm' button to
    see how your choices impact the model's accuracy and view performance visualizations.
    """
)
st.divider()

# --- Classifier Parameter Selection ---
params = {}
st.sidebar.header("2. Set Hyperparameters")

if classifier_name == "K-Nearest Neighbors (KNN)":
    params["n_neighbors"] = st.sidebar.slider("Number of neighbors (K)", 1, 15, 5, 1)
    params["weights"] = st.sidebar.select_slider(
        "Weight function", options=["uniform", "distance"], value="uniform"
    )
    params["algorithm"] = st.sidebar.selectbox(
        "Algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
    )

elif classifier_name == "Support Vector Machine (SVM)":
    params["C"] = st.sidebar.slider(
        "Regularization parameter (C)", 0.01, 10.0, 1.0, 0.01
    )
    params["kernel"] = st.sidebar.selectbox(
        "Kernel", ["rbf", "linear", "poly", "sigmoid"]
    )
    if params["kernel"] == "poly":
        params["degree"] = st.sidebar.slider("Degree (for poly kernel)", 2, 5, 3, 1)
    params["gamma"] = st.sidebar.select_slider(
        "Kernel coefficient (gamma)", options=["scale", "auto"], value="scale"
    )

elif classifier_name == "Random Forest":
    params["n_estimators"] = st.sidebar.slider("Number of estimators", 10, 200, 100, 10)
    params["max_depth"] = st.sidebar.slider("Max depth of the tree", 2, 15, 5, 1)
    params["criterion"] = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
    params["min_samples_split"] = st.sidebar.slider(
        "Min samples required to split", 2, 10, 2, 1
    )


# --- Model Training and Evaluation ---
st.sidebar.header("3. Run the Model")
if st.sidebar.button("Run Algorithm", use_container_width=True, type="primary"):
    # Instantiate the classifier
    if classifier_name == "K-Nearest Neighbors (KNN)":
        clf = KNeighborsClassifier(
            n_neighbors=params["n_neighbors"],
            weights=params["weights"],
            algorithm=params["algorithm"],
        )
    elif classifier_name == "Support Vector Machine (SVM)":
        degree = params.get("degree", 3)
        clf = SVC(
            C=params["C"],
            kernel=params["kernel"],
            gamma=params["gamma"],
            degree=degree,
            probability=True,
        )
    elif classifier_name == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            criterion=params["criterion"],
            min_samples_split=params["min_samples_split"],
            random_state=42,
        )

    # Train the model and make predictions
    with st.spinner("Training the model and generating visualizations..."):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

    # --- Display Results ---
    st.subheader(f"Results for: {classifier_name}")

    col1, col2 = st.columns(2)
    with col1:
        st.info("Selected Hyperparameters")
        st.json(params)

    with col2:
        st.metric(label="Model Accuracy", value=f"{accuracy:.4f}")

    st.success(
        f"The {classifier_name} model achieved an accuracy of **{accuracy * 100:.2f}%** on the test set."
    )
    st.divider()

    # --- Display Visualizations ---
    st.subheader("Performance Visualizations")
    vis_col1, vis_col2 = st.columns(2)

    with vis_col1:
        st.write("#### Confusion Matrix")
        plot_confusion_matrix(y_test, y_pred, target_names)

    with vis_col2:
        st.write("#### Decision Boundary (on 2D PCA data)")
        # We pass a clone of the classifier to avoid altering the original one,
        # which was trained on the 4D data. The plotting function will retrain
        # this clone on the 2D PCA-transformed data for visualization.
        clf_for_plot = clone(clf)
        plot_decision_boundary(
            X_train, y_train, clf_for_plot, f"Decision Boundary for {classifier_name}"
        )

    # Display feature importances for Random Forest
    if classifier_name == "Random Forest" and hasattr(clf, "feature_importances_"):
        st.subheader("Feature Importances")
        importances = pd.DataFrame(
            {"Feature": feature_names, "Importance": clf.feature_importances_}
        ).sort_values(by="Importance", ascending=False)
        st.bar_chart(importances.set_index("Feature"))

else:
    st.info(
        "Select a classifier and its parameters in the sidebar, then click 'Run Algorithm' to see the results."
    )

# --- Dataset Information ---
st.divider()
with st.expander("About the Dataset"):
    st.write(
        """
        The Iris dataset is a classic and perhaps the most famous dataset in pattern recognition literature.
        It contains 3 classes of 50 instances each, where each class refers to a type of iris plant.
        """
    )
    st.write("**Features:**")
    st.write(", ".join([f"`{name}`" for name in feature_names]))
    st.write("**Target Classes:**")
    st.write(", ".join([f"`{name}`" for name in target_names]))
