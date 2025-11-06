import numpy as np

def predict_test_data(model, test_data):
    true_labels = []
    pred_labels = []

    for images, labels in test_data:
        preds = model.predict(images)
        true_labels.extend(np.argmax(labels.numpy(), axis=1))
        pred_labels.extend(np.argmax(preds, axis=1))
    return true_labels, pred_labels