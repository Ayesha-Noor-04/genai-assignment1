from __future__ import annotations

import torch


def predict_corruption(classifier, x):
    classifier.eval()

    with torch.no_grad():
        logits = classifier(x)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

    return predictions, probabilities


def route_hard(
    predictions,
    specialists,
    clean_id=0,
):
    routes = []

    for prediction in predictions.tolist():
        if prediction == clean_id:
            routes.append(None)
        else:
            routes.append(specialists[prediction])

    return routes