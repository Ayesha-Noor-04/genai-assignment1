from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support
)


def classifier_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support
    }


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(
        y_true,
        y_pred,
        normalize="true"
    )


def get_classification_report(y_true, y_pred):
    return classification_report(
        y_true,
        y_pred,
        target_names=[
            "clean",
            "salt",
            "blur",
            "occlusion"
        ],
        digits=4
    )