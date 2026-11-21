
import shap
import matplotlib.pyplot as plt


def explain_model(model, X_sample):
    """
    Generate SHAP explanations for a tree-based model.
    """

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values = explainer(X_sample)

    # Global feature importance
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(
        "shap_feature_importance.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()

    # SHAP summary plot
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(
        "shap_summary.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()

    print("SHAP plots saved successfully!")

    return explainer, shap_values
