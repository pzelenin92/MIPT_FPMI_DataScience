import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    #найдем tp,fn,fp,tn

    tp = np.count_nonzero((prediction == ground_truth) & (prediction == True)) # tp - True positive = zeros
    tn = np.count_nonzero((prediction == ground_truth) & (prediction == False)) # tn - True negative = ones
    fp = np.count_nonzero((prediction != ground_truth) & (prediction == True)) # fp - false positives = zeros
    fn = np.count_nonzero((prediction != ground_truth) & (prediction == False)) # fn - false negative = ones

    #метрики
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy

    maska = (prediction == ground_truth)
    tp_tn = np.count_nonzero(maska) # tp_tn - true positive + true negative
    accuracy = tp_tn / ground_truth.size

    return accuracy
