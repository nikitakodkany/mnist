"""
SCRIPT CONTAINING ADAM INITIALIZATION
FOR A NUERAL NETWORK OF 5 LAYERS
"""

def initialize_adam():
    """
    Initializes v and s with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.

    Arguments:
    .get -- p contains your parameters.


    Returns:
    v -- will contain the exponentially weighted average of the gradient.
                    vdW  = ...
                    vdb  = ...
    s --  will contain the exponentially weighted average of the squared gradient.
                    sdW = ...
                    sdb = ...

    """
    w1, w2, w3, w4, w5, b1, b2, b3, b4, b5 = self.get('w1', 'w2', 'w3', 'w4', 'w5', 'b1', 'b2', 'b3', 'b4', 'b5')

    vdw1 = np.zeros(w1)
    vdb1 = np.zeros(b1)
    sdw1 = np.zeros(w1)
    sdb1 = np.zeros(b1)

    vdw2 = np.zeros(w2)
    vdb2 = np.zeros(b2)
    sdw1 = np.zeros(w1)
    sdb1 = np.zeros(b1)

    vdw3 = np.zeros(w3)
    vdb3 = np.zeros(b3)
    sdw1 = np.zeros(w1)
    sdb1 = np.zeros(b1)

    vdw4 = np.zeros(w4)
    vdb4 = np.zeros(b4)
    sdw1 = np.zeros(w1)
    sdb1 = np.zeros(b1)

    vdw5 = np.zeros(w5)
    vdb5 = np.zeros(b5)
    sdw1 = np.zeros(w1)
    sdb1 = np.zeros(b1)

    self.put(vdw1=vdw1,vdw2=vdw2,vdw3=vdw3,vdw4=vdw4,vdw5=vdw5,vdb1=vdb1,vdb2=vdb2,vdb3=vdb3,vdb4=vdb4,vdb5=vdb5)
    self.put(sdw1=sdw1,sdw2=sdw2,sdw3=sdw3,sdw4=sdw4,sdw5=sdw5,sdb1=sdb1,sdb2=sdb2,sdb3=sdb3,sdb4=sdb4,sdb5=sdb5)

def update_parameters_with_adam(self, t, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-9, rate = 0.05):
    """
    Update parameters using Adam

    Arguments:
    .get -- containing your parameters:
                        Wl
                        bl
                        dWl
                        dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates
    beta2 -- Exponential decay hyperparameter for the second moment estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    .put -- containing your updated parameters
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """

    w1,w2,w3,w4,w5,dw1,dw2,dw3,dw4,dw5,b1,b2,b3,b4,b5,db1,db2,db3,db4,db5 = self.get('w1','w2','w3','w4','w5','dw1','dw2','dw3','dw4','dw5','b1','b2','b3','b4','b5','db1','db2','db3','db4','db5')
    vdw1,vdw2,vdw3,vdw4,vdw5,vdb1,vdb2,vdb3,vdb4,vdb5 = self.get('vdw1','vdw2','vdw3','vdw4','vdw5','vdb1','vdb2','vdb3','vdb4','vdb5')
    sdw1,sdw2,sdw3,sdw4,sdw5,sdb1,sdb2,sdb3,sdb4,sdb5 = self.get('sdw1','sdw2','sdw3','sdw4','sdw5','sdb1','sdb2','sdb3','sdb4','sdb5')

    #moving average of the gradients
    vdw1 = (beta1*vdw1) + ((1-beta1)*dw1)
    vdb1 = (beta1*vdb1) + ((1-beta1)*db1)

    vdw2 = (beta1*vdw2) + ((1-beta1)*dw2)
    vdb2 = (beta1*vdb2) + ((1-beta1)*db2)

    vdw3 = (beta1*vdw3) + ((1-beta1)*dw3)
    vdb3 = (beta1*vdb3) + ((1-beta1)*db3)

    vdw4 = (beta1*vdw4) + ((1-beta1)*dw4)
    vdb4 = (beta1*vdb4) + ((1-beta1)*db4)

    vdw5 = (beta1*vdw5) + ((1-beta1)*dw5)
    vdb5 = (beta1*vdb5) + ((1-beta1)*db5)

    #compute bias-corrected first memnt estimate
    vc_dw1 = vdw1 / (1-np.power(beta1,t))
    vc_db1 = vdb1 / (1-np.power(beta1,t))

    vc_dw2 = vdw2 / (1-np.power(beta1,t))
    vc_db2 = vdb2 / (1-np.power(beta1,t))

    vc_dw3 = vdw3 / (1-np.power(beta1,t))
    vc_db3 = vdb3 / (1-np.power(beta1,t))

    vc_dw4 = vdw4 / (1-np.power(beta1,t))
    vc_db4 = vdb4 / (1-np.power(beta1,t))

    vc_dw5 = vdw5 / (1-np.power(beta1,t))
    vc_db5 = vdb5 / (1-np.power(beta1,t))

    #moving average of the squared gradients
    sdw1 = (beta2*sdw1) + ((1-beta2)*dw1)
    sdb1 = (beta2*sdb1) + ((1-beta2)*db1)

    sdw2 = (beta2*sdw2) + ((1-beta2)*dw2)
    sdb2 = (beta2*sdb2) + ((1-beta2)*db2)

    sdw3 = (beta2*sdw3) + ((1-beta2)*dw3)
    sdb3 = (beta2*sdb3) + ((1-beta2)*db3)

    sdw4 = (beta2*sdw4) + ((1-beta2)*dw4)
    sdb4 = (beta2*sdb4) + ((1-beta2)*db4)

    sdw5 = (beta2*sdw5) + ((1-beta2)*dw5)
    sdb5 = (beta2*sdb5) + ((1-beta2)*db5)

    #compute bias-corrected first memnt estimate
    sc_dw1 = sdw1 / (1-np.power(beta2,t))
    sc_db1 = sdb1 / (1-np.power(beta2,t))

    sc_dw2 = sdw2 / (1-np.power(beta2,t))
    sc_db2 = sdb2 / (1-np.power(beta2,t))

    sc_dw3 = sdw3 / (1-np.power(beta2,t))
    sc_db3 = sdb3 / (1-np.power(beta2,t))

    sc_dw4 = sdw4 / (1-np.power(beta2,t))
    sc_db4 = sdb4 / (1-np.power(beta2,t))

    sc_dw5 = sdw5 / (1-np.power(beta2,t))
    sc_db5 = sdb5 / (1-np.power(beta2,t))

    #update parameters
    w1 -= rate * vc_dw1 / np.sqrt(sc_dw1+epsilon)
    b1 -= rate * vc_db1 / np.sqrt(sc_db1+epsilon)

    w2 -= rate * vc_dw2 / np.sqrt(sc_dw2+epsilon)
    b2 -= rate * vc_db2 / np.sqrt(sc_db2+epsilon)

    w3 -= rate * vc_dw3 / np.sqrt(sc_dw3+epsilon)
    b3 -= rate * vc_db3 / np.sqrt(sc_db3+epsilon)

    w4 -= rate * vc_dw4 / np.sqrt(sc_dw4+epsilon)
    b4 -= rate * vc_db4 / np.sqrt(sc_db4+epsilon)

    w5 -= rate * vc_dw5 / np.sqrt(sc_dw5+epsilon)
    b5 -= rate * vc_db5 / np.sqrt(sc_db5+epsilon)

    self.put(w1 = w1, w2 = w2, w3 = w3, w4 = w4, w5 = w5, b1 = b1, b2 = b2, b3 = b3, b4 = b4, b5 = b5)
    self.put(vdw1=vdw1,vdw2=vdw2,vdw3=vdw3,vdw4=vdw4,vdw5=vdw5,vdb1=vdb1,vdb2=vdb2,vdb3=vdb3,vdb4=vdb4,vdb5=vdb5)
    self.put(sdw1=sdw1,sdw2=sdw2,sdw3=sdw3,sdw4=sdw4,sdw5=sdw5,sdb1=sdb1,sdb2=sdb2,sdb3=sdb3,sdb4=sdb4,sdb5=sdb5)
