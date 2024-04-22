from sklearn.metrics import roc_auc_score, roc_curve, f1_score, fbeta_score
from src.visualization.visualize import plot_evaluation
import mlflow

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, average_precision_score
import numpy as np

def multilabel_roc_analysis_and_plot(learn, target_names:list, dl=None):
    """
    Perform ROC curve and ROC AUC score analysis for multilabel classification using fastai and sklearn.
    Plot ROC curves for each class in a single matplotlib figure.

    Parameters:
    - learn: fastai Learner object
    - dl: fastai DataLoader object (optional, if None, validation DataLoader from learn is used)

    Returns:
    - roc_auc: array, shape = [n_classes]
      Area under ROC curve for each class
    """

    # Get predictions
    if dl is None:
        dl = learn.dls.valid
        title = 'Receiver Operator Characteristics (ROC)'

    preds, targets = learn.get_preds(dl=dl)
    title = 'Test dataset - Receiver Operator Characteristics (ROC)'
    # Convert to numpy arrays
    preds_np = preds.numpy()
    targets_np = targets.numpy()

    # Initialize arrays for storing results
    n_classes = preds_np.shape[1]
    roc_auc = np.zeros(n_classes)

    # Set up matplotlib figure
    plt.figure(figsize=(6, 6))

    # Plot ROC curve for each class
    for i in range(n_classes):
        fpr, tpr, thresholds = roc_curve(targets_np[:, i], preds_np[:, i])
        roc_auc[i] = roc_auc_score(targets_np[:, i], preds_np[:, i])
        tn = target_names[i]
        plt.plot(fpr, tpr, label=f'{tn:<15} (AUC = {roc_auc[i]:>.2f})')

    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')

    # Set plot properties
    plt.xlim([0.0, 1.])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)

    # Show plot
    plt.show()

    return roc_auc

def multilabel_roc_pr_analysis_and_plot(learn, target_names:list, dl=None, show=True):
    """
    Perform ROC curve and Precision-Recall curve analysis for multilabel classification using fastai and sklearn.
    Plot ROC and Precision-Recall curves for each class in a subplot.

    Parameters:
    - learn: fastai Learner object
    - dl: fastai DataLoader object (optional, if None, validation DataLoader from learn is used)

    Returns:
    - roc_auc: array, shape = [n_classes]
      Area under ROC curve for each class
    - pr_auc: array, shape = [n_classes]
      Area under Precision-Recall curve for each class
    """
    # Change the font type.
    plt.rcParams['font.family'] = 'monospace'

    # Get predictions
    if dl is None:
        dl = learn.dls.valid
        roc_title = 'Receiver Operator Characteristics (ROC)'
        pr_title = 'Precision Recall (PR)'

    preds, targets = learn.get_preds(dl=dl)
    roc_title = 'Test dataset - Receiver Operator Characteristics (ROC)'
    pr_title = 'Test dataset - Precision Recall (PR)'
    # Convert to numpy arrays
    preds_np = preds.numpy()
    targets_np = targets.numpy()

    # Initialize arrays for storing results
    n_classes = preds_np.shape[1]
    roc_auc = np.zeros(n_classes)
    pr_auc = np.zeros(n_classes)

    # Set up matplotlib figure with subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    title_fontsize = 12

    # Plot ROC curve for each class
    ax_roc = axes[0]
    ax_roc.set_title(roc_title, fontsize= title_fontsize)
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.02])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')

    # Plot Precision-Recall curve for each class
    ax_pr = axes[1]
    ax_pr.set_title(pr_title, fontsize= title_fontsize)
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.02])
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')

    # Plot curves for each class
    for i in range(n_classes):
        tn = target_names[i].removesuffix("_major")
        # ROC curve
        fpr, tpr, _ = roc_curve(targets_np[:, i], preds_np[:, i])
        roc_auc[i] = roc_auc_score(targets_np[:, i], preds_np[:, i])
        ax_roc.plot(fpr, tpr, label=f'{tn:<10} (ROC-AUC = {roc_auc[i]:.2f})')

        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(targets_np[:, i], preds_np[:, i])
        pr_auc[i] = auc(recall, precision)
        ax_pr.plot(recall, precision, label=f'{tn:<10} (PR-AUC = {pr_auc[i]:.2f})')

    # Add diagonal line to ROC plot (random classifier)
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Random' ,c ="darkgray")
    #ax_pr.plot([0, 1], [0.05, 0.05], 'k--', label='Average (P/N)',c ="darkgray")

    ax_roc.grid(True)
    ax_pr.grid(True)

    # Set legend for both subplots
    ax_roc.legend(loc='lower right')
    ax_pr.legend(loc='upper right')

    # Show plots
    #plt.tight_layout()
    
    plt.savefig("reports/figures/metric_plot.png")
    if show:
        plt.show()

    return roc_auc, pr_auc

def get_metrics( learn, f, plot=True):
    y_preds, ys = learn.get_preds()
    test_y_preds, test_ys = learn.get_preds(dl=f.test_mixed_dls.valid)
    if plot:
        plot_evaluation(y_preds, ys, f.target)
        plot_evaluation(test_y_preds, test_ys, f.target)

    fpr, tpr, thresholds = roc_curve(ys, y_preds[:,1])
    roc_auc = roc_auc_score(ys, y_preds[:,1])

    test_fpr, test_tpr, test_thresholds = roc_curve(test_ys, test_y_preds[:,1])
    test_roc_auc = roc_auc_score(test_ys, test_y_preds[:,1])
    
# Use threshold moving to calc f-scores
   #f1 = f1_score(test_ys, test_y_preds[:,1])
    #f_beta = fbeta_score(test_ys, test_y_preds[:,1], beta=2.0)

    log_metrics = { "validation_roc_auc":roc_auc,
                    "test_roc_auc":test_roc_auc,
                   # "test_f1_score":f1,
                   # "test_f2_score":f_beta
                }
    for name, var in log_metrics.items():
        mlflow.log_metric(name, var) # type: ignore