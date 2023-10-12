import numpy as np


def cross_entropy_loss(scores, t, reg, model_params):
    """Calcule la loss avec la méthode "entropie croisée multi-classe".

    Arguments:
        scores {ndarray} -- Scores du réseau (sortie de la dernière couche).
                            Shape (N, C)
        t {ndarray} -- Labels attendus pour nos échantillons d'entraînement.
                       Shape (N, )
        reg {float} -- Terme de régulatisation
        model_params {dict} -- Dictionnaire contenant l'ensemble des paramètres
                               du modèle. Obtenus avec la méthode parameters()
                               de Model.

    Returns:
        tuple -- Tuple contenant la loss et la dérivée de la loss par rapport
                 aux scores.
    """

    N = scores.shape[0]
    C = scores.shape[1]

    stable_scores = scores - np.max(scores, axis=1)[:, None]
    exp_scores = np.exp(stable_scores)
    softmax_output = exp_scores / np.sum(exp_scores, axis=1)[:, None]

    target_indices_mask = np.eye(C, dtype='bool')[t]
    target_outputs = softmax_output[target_indices_mask]
    
    loss = -np.sum(np.log(target_outputs)) / N
    # print("calculate loss")
    for layer_params in model_params.values():
        for param in layer_params.values():
            #print(param)
            #print(np.sum(np.square(param)))
            #print(loss)
            loss += 0.5 * reg * np.sum(np.square(param))
            #print(loss)

    dScores = (softmax_output - target_indices_mask.astype(int)) / N

    return loss, dScores, softmax_output


def hinge_loss(scores, t, reg, model_params):
    """Calcule la loss avec la méthode "hinge loss multi-classe".

    Arguments:
        scores {ndarray} -- Scores du réseau (sortie de la dernière couche).
                            Shape (N, C)
        t {ndarray} -- Labels attendus pour nos échantillons d'entraînement.
                       Shape (N, )
        reg {float} -- Terme de régulatisation
        model_params {dict} -- Dictionnaire contenant l'ensemble des paramètres
                               du modèle. Obtenus avec la méthode parameters()
                               de Model.

    Returns:
        tuple -- Tuple contenant la loss et la dérivée de la loss par rapport
                 aux scores.
    """

    N = scores.shape[0]
    loss = 0
    dScores = np.zeros(scores.shape)

    for n in range(N):
        predicted = np.argmax(scores[n])
        tn = t[n]
        m = max(0, 1 + scores[n, predicted] - scores[n, tn])
        loss += m
        if m > 0:
            dScores[n, predicted] += 1
            dScores[n, tn] -= 1

    for layer_params in model_params.values():
        for param in layer_params.values():
            loss += 0.5 * reg * np.sum(np.square(param))

    score_correct_classes = scores[range(scores.shape[0]), t]
    return loss, dScores, score_correct_classes
