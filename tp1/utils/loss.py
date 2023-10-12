import numpy as np


def softmax_ce_naive_forward_backward(X, W, y, reg):
    """Implémentation naive qui calcule la propagation avant, puis la
       propagation arrière pour finalement retourner la perte entropie croisée
       (ce) + une régularisation L2 et le gradient des poids. Utilise une 
       activation softmax en sortie.
       
       NOTE : la fonction codée est : EntropieCroisée + 0.5*reg*||W||^2
       
       N'oubliez pas le 0.5!

    Inputs:
    - X: Numpy array, shape (N, D). N représente le nombre d'exemple d'entrainement
        dans X, et D représente la dimension des exemples de X.
    - W: Numpy array, shape (D, C)
    - y: Numpy array, shape (N,). y[i] = c veut dire que X[i] appartient à la
        classe c, 0 <= c < C
    - reg: float. Terme de regularisation L2

    Outputs:
    - loss: float. Perte du classifieur linéaire softmax
    - dW: Numpy array, shape (D, C). Gradients des poids W
    """

    N = X.shape[0]
    D = X.shape[1]
    C = W.shape[1]

    loss = 0
    dW = np.zeros(W.shape)

    # Let's calculate loss
    
    # layer_1 is N x C
    layer_1 = []
    for n in range(N):
        layer_1_sub = []
        for c in range(C):
            product_sum = 0
            for d in range(D):
                product_sum += X[n][d] * W[d][c]
            layer_1_sub.append(product_sum)
        layer_1.append(layer_1_sub)
                
    # we calculate e^fi
    e_fi = []
    for n in range(N):
        e_fi_sub = []
        for c in range(C):
            e_fi_sub.append(np.exp(layer_1[n][c]))
        e_fi.append(e_fi_sub)
            
    # we calculate sum over i of e_fi
    sum_e_fi = []
    for n in range(N):
        temp = 0
        for c in range(C):
            temp += e_fi[n][c]
        sum_e_fi.append(temp)
            
    # we calculate Si
    #Si = e_fi / sum_e_fi
    Si = []
    for n in range(N):
        Si_sub = []
        for c in range(C):
            Si_sub.append(e_fi[n][c] / sum_e_fi[n])
        Si.append(Si_sub)
    
    # we calculate St
    St = []
    for n in range(N):
        St.append(Si[n][y[n]])
    
    # we calculate the cross entropy
    #ce = np.log(St)
    ce = []
    for n in range(N):
        ce.append(np.log(St[n]))
    
    # finally calculate the loss
    #loss = -np.sum(ce) / N + 0.5 * reg * np.sum(W**2)
    ce_sum = 0
    for n in range(N):
        ce_sum += ce[n]
    w2_sum = 0
    for n in range(N):
        for c in range(C):
            w2_sum += W[n][c]**2
    loss = -ce_sum / N + 0.5 * reg * w2_sum
    
    
    # Let's calculate dW
    dW = np.zeros((D, C))
    for d in range(D):
        for c in range(C):
            temp = 0
            for n in range(N):
                if c == y[n]:
                    temp += X[n][d] * (Si[n][c] - 1)
                else:
                    temp += X[n][d] * Si[n][c]
            dW[d][c] = temp / N + reg * W[d][c]

    return loss, dW


def softmax_ce_forward_backward(X, W, y, reg):
    """Implémentation vectorisée qui calcule la propagation avant, puis la
       propagation arrière pour finalement retourner la perte entropie croisée
       (ce) et le gradient des poids. Utilise une activation softmax en sortie.
        
       NOTE : la fonction codée est : EntropieCroisée + 0.5*reg*||W||^2      
       N'oubliez pas le 0.5!

    Inputs:
    - X: Numpy array, shape (N, D). N représente le nombre d'exemples d'entrainement
        dans X, et D représente la dimension des exemples de X.
    - W: Numpy array, shape (D, C)
    - y: Numpy array, shape (N,). y[i] = c veut dire que X[i] appartient à la
        classe c, 0 <= c < C
    - reg: float. Terme de regularisation L2

    Outputs:
    - loss: float. Perte du classifieur linéaire softmax
    - dW: Numpy array, shape (D, C). Gradients des poids W
    """
    N = X.shape[0]
    C = W.shape[1]
    loss = 0.0
    dW = np.zeros(W.shape)

    # Let's calculate loss
    
    # layer_1 is N x C
    layer_1 = X.dot(W)
    # we calculate e^fi, sum over i of e_fi and St
    e_fi = np.exp(layer_1)
    sum_e_fi = np.sum(e_fi, axis=1).reshape(N, 1)
    # Si is N x C
    Si = e_fi / sum_e_fi
    # St is N x 1
    St = Si[np.arange(N), y]
    # we calculate the cross entropy
    ce = np.log(St)
    # finally calculate the loss
    loss = -np.sum(ce) / N + 0.5 * reg * np.sum(W**2)

    
    # Let's calculate dW
    
    t_one_hot = np.zeros((N, C))
    t_one_hot[np.arange(N), y] = 1
    dW = X.transpose().dot(Si - t_one_hot)/N + reg * W
    
    return loss, dW


def hinge_naive_forward_backward(X, W, y, reg):
    """Implémentation naive calculant la propagation avant, puis la
       propagation arrière, pour finalement retourner la perte hinge et le
       gradient des poids.
       
       NOTE : la fonction codée est : loss = Hinge + 0.5*reg*||W||^2
       N'oubliez pas le 0.5!

    Inputs:
    - X: Numpy array, shape (N, D)
    - W: Numpy array, shape (D, C)
    - y: Numpy array, shape (N,). y[i] = c veut dire que X[i] appartient à la
         classe c, 0 <= c < C
    - reg: float. Terme de regularisation L2

    Outputs:
    - loss: float. Perte du classifieur linéaire hinge
    - dW: Numpy array, shape (D, C). Gradients des poids W
    """
    loss = 0.0
    dW = np.zeros(W.shape)
    
    # learning rate
    lr = 1
    nb_data = X.shape[0]
    
    for n in range(nb_data):
        pred_x = W.T @ X[n]
        predict = np.argmax(pred_x)
        true_value = y[n]
        
        Hinge = max(0, 1 + pred_x[predict] -pred_x[true_value])
        loss += Hinge
        
        dW.T[true_value] -= X[n]
        dW.T[predict] += X[n]
    
    regularisation = 0.5 * reg * np.sum(W**2)
    loss = loss/nb_data + regularisation
    dW /= nb_data
    dW += reg * W
    
    return loss, dW



def hinge_forward_backward(X, W, y, reg):
    """Implémentation vectorisée calculant la propagation avant, puis la
       propagation arrière, pour finalement retourner la perte hinge et le
       gradient des poids.

       NOTE : la fonction codée est : Hinge + 0.5*reg*||W||^2
       N'oubliez pas le 0.5!

       RAPPEL hinge loss one-vs-one :
       loss = max(0, 1 + score_classe_predite - score_classe_vérité_terrain)
       
    Inputs:
    - X: Numpy array, shape (N, D)
    - W: Numpy array, shape (D, C)
    - y: Numpy array, shape (N,). y[i] = c veut dire que X[i] appartient à la
         classe c, 0 <= c < C
    - reg: float. Terme de regularisation L2

    Outputs:
    - loss: float. Perte du classifieur linéaire hinge
    - dW: Numpy array, shape (D, C). Gradients des poids W
    """
    loss = 0.0
    dW = np.zeros(W.shape)

    # Let's calculate loss
    
    N = len(X)
    # layer_1 is N x C
    layer_1 = X.dot(W)
    # predicted_class_score is a 1D array of N elements
    predicted_class_score = np.amax(layer_1, axis=1)
    # good_class_score is a 1D array of N elements
    good_class_score = layer_1[np.arange(N), y.transpose()]
    
    zeros = np.zeros(N)
    ones = np.ones(N)
    loss_before_max = ones + predicted_class_score - good_class_score
    xn_losses = np.amax(np.array([zeros, loss_before_max]), axis=0)
    loss = xn_losses.sum() / N + 0.5 * reg * np.sum(W**2)
    
    
    # Let's calculate dW
    
    condition = np.where(loss_before_max != 0, 1, 0).reshape(N, 1)
    #grad_Wj_E = X * condition
    #grad_Wtn_E = -X * condition
    X_backprop = X * condition
    dW = dW.transpose()
    for i, Xn in enumerate(X_backprop):
        j = np.where(layer_1[i] == predicted_class_score[i])[0]
        tn = y[i]
        dW[j] += Xn
        dW[tn] -= Xn
        
    dW = dW.transpose() / N + reg * W

    return loss, dW
