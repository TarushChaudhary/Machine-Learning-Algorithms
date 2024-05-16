import numpy as np

def NB_XGivenY(XTrain, yTrain, a=0.001, b=0.9):
    """
    Compute the probability of P(X|Y).
    """
    num_samples, vocab_size = XTrain.shape
    D = np.zeros((2, vocab_size))

    for c in [1, 2]:
        class_indices = (yTrain == c).flatten()
        XTrain_class = XTrain[class_indices, :]
        word_counts = np.sum(XTrain_class, axis=0)
        total_words = np.sum(word_counts)
        D[c-1, :] = (word_counts + a) / (total_words + a/b)

    return D

def NB_YPrior(yTrain):
    """
    Compute the probability of P(Y).
    """
    num_samples = yTrain.shape[0]
    p = np.sum(yTrain == 1) / num_samples
    return p

def NB_Classify(D, p, X):
    """
    Predict the labels of X.
    """
    num_samples = X.shape[0]
    y = np.zeros((num_samples, 1))

    log_D = np.log(D)
    log_D_neg = np.log(1 - D)
    log_p = np.log(p)
    log_p_neg = np.log(1 - p)

    for i in range(num_samples):
        log_likelihood_1 = X[i, :] @ log_D[0, :].T + (1 - X[i, :]) @ log_D_neg[0, :].T + log_p
        log_likelihood_2 = X[i, :] @ log_D[1, :].T + (1 - X[i, :]) @ log_D_neg[1, :].T + log_p_neg
        y[i, 0] = 1 if log_likelihood_1 > log_likelihood_2 else 2

    return y

def NB_ClassificationAccuracy(yHat, yTruth):
    """
    Compute the accuracy of predictions.
    """
    num_samples = yHat.shape[0]
    acc = np.sum(yHat == yTruth) / num_samples
    return acc
