import numpy as np
import math


def cross_entropy_loss(scores, t, reg, model_params):
    """Calcule l'entropie croisée multi-classe.

    Arguments:
        scores {ndarray} -- Scores du réseau (sortie de la dernière couche).
                            Shape (N, C)
        t {ndarray} -- Labels attendus pour nos échantillons d'entraînement.
                       Shape (N, )
        reg {float} -- Terme de régulatisation
        model_params {dict} -- Dictionnaire contenant les paramètres de chaque couche
                               du modèle. Voir la méthode "parameters()" de la
                               classe Model.

    Returns:
        loss {float} 
        dScore {ndarray}: dérivée de la loss par rapport aux scores. : Shape (N, C)
        softmax_output {ndarray} : Shape (N, C)
    """
    N = scores.shape[0]
    C = scores.shape[1]
    loss = 0
    dScores = np.zeros(scores.shape)
    softmax_output = np.zeros(scores.shape)
    
    # TODO
    # Calculer la sortie softmax
    e_fi = np.exp(scores)
    sum_e_fi = np.sum(e_fi, axis=1).reshape(N, 1)

    # softmax_output de dimension N x C
    softmax_output = e_fi / sum_e_fi
    
    # Calcul de la cross_entropy
    # St de dimension N x 1
    St = softmax_output[np.arange(N), t]
    ce = np.log(St)
    
    # Calcul de la loss
    W_2 = 0
    b_2 = 0
    for params in model_params.values():
        W_2 += np.sum(params['W']**2)
        b_2 += np.sum(params['b']**2)
    loss = -np.sum(ce) / N + 0.5 * reg * (W_2 + b_2)
    
    # Let's calculate dScore
    t_one_hot = np.zeros((N, C))
    t_one_hot[np.arange(N), t] = 1
    dScores = (softmax_output - t_one_hot)


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
    C = scores.shape[1]
    loss = 0
    dScores = np.zeros(scores.shape)
    score_correct_classes = np.zeros(scores.shape)

    # TODO
    # Ajouter code ici
    return loss, dScores, score_correct_classes
