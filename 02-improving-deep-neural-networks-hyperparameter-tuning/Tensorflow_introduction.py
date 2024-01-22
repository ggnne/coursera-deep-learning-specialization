# GRADED FUNCTION: linear_function
def linear_function():
    """
    Implements a linear function: 
            Initializes X to be a random tensor of shape (3,1)
            Initializes W to be a random tensor of shape (4,3)
            Initializes b to be a random tensor of shape (4,1)
    Returns: 
    result -- Y = WX + b 
    """

    np.random.seed(1)
    
    """
    Note, to ensure that the "random" numbers generated match the expected results,
    please create the variables in the order given in the starting code below.
    (Do not re-arrange the order).
    """
    X = tf.constant(np.random.randn(3,1), name = "X")
    W = tf.Variable(np.random.randn(4,3), name = "X")
    b = tf.Variable(np.random.randn(4,1), name = "X")
    Y = tf.matmul(W, X) + b     
    
    return Y


# GRADED FUNCTION: sigmoid
def sigmoid(z):
    
    """
    Computes the sigmoid of z
    
    Arguments:
    z -- input value, scalar or vector
    
    Returns: 
    a -- (tf.float32) the sigmoid of z
    """

    z = tf.cast(z, tf.float32)
    a = tf.keras.activations.sigmoid(z)

    return a


# GRADED FUNCTION: one_hot_matrix
def one_hot_matrix(label, C=6):
    """
    Computes the one hot encoding for a single label
    
    Arguments:
        label --  (int) Categorical labels
        C --  (int) Number of different classes that label can take
    
    Returns:
         one_hot -- tf.Tensor A single-column matrix with the one hot encoding.
    """

    one_hot = tf.reshape(tf.one_hot(label, C, axis=0), [C, ])
    
    return one_hot


# GRADED FUNCTION: initialize_parameters
def initialize_parameters():
    """
    Initializes parameters to build a neural network with TensorFlow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
                                
    initializer = tf.keras.initializers.GlorotNormal(seed=1)   
    W1 = tf.Variable(initializer(shape=([25, 12288])))
    b1 = tf.Variable(initializer(shape=([25, 1])))
    W2 = tf.Variable(initializer(shape=([12, 25])))
    b2 = tf.Variable(initializer(shape=([12, 1])))
    W3 = tf.Variable(initializer(shape=([6, 12])))
    b3 = tf.Variable(initializer(shape=([6, 1])))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters


# GRADED FUNCTION: forward_propagation
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.math.add(tf.linalg.matmul(W1,X), b1)                           # Z1 = np.dot(W1, X) + b1
    A1 = tf.keras.activations.relu(Z1)                                      # A1 = relu(Z1)
    Z2 = tf.math.add(tf.linalg.matmul(W2,A1), b2)                                   # Z2 = np.dot(W2, A1) + b2
    A2 = tf.keras.activations.relu(Z2)                                       # A2 = relu(Z2)
    Z3 = tf.math.add(tf.linalg.matmul(W3,A2), b3)                                  # Z3 = np.dot(W3, A2) + b3
    
    return Z3


# GRADED FUNCTION: compute_total_loss 
def compute_total_loss(logits, labels):
    """
    Computes the total loss
    
    Arguments:
    logits -- output of forward propagation (output of the last LINEAR unit), of shape (6, num_examples)
    labels -- "true" labels vector, same shape as Z3
    
    Returns:
    total_loss - Tensor of the total loss value
    """
    
    
    total_loss = tf.reduce_sum(tf.keras.losses.categorical_crossentropy(y_true=tf.transpose(labels),
                                                                        y_pred=tf.transpose(logits),
                                                                        from_logits=True))
    
    return total_loss
