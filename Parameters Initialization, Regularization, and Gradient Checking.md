$\newcommand{\abs}[1]{\left| #1 \right|}$  
# $\text{Deep Learning Fundamentals: Parameter Initialization, Regularization \& Gradient Checking}$

*A comprehensive guide covering the essential techniques for training stable and efficient deep neural networks*

## $\text{Table of Contents}$
1. [Parameter Initialization](#parameter-initialization)
2. [Regularization Techniques](#regularization-techniques)
3. [Gradient Checking](#gradient-checking)
4. [Modern Best Practices](#modern-best-practices)
5. [Implementation Examples](#implementation-examples)

---

## $\text{Parameter Initialization}$

### $Why~Proper~Initialization~Matters$

Parameter initialization is critical for successful deep learning training. Poor initialization can lead to:
- **Vanishing gradients**: Gradients become exponentially small in deep networks
- **Exploding gradients**: Gradients become exponentially large
- **Symmetry breaking**: All neurons learn the same features
- **Slow convergence**: Training takes much longer to reach optimal solutions

### $The~Mathematical~Foundation$

For proper signal propagation through deep networks, we need to maintain the variance of activations and gradients across layers. The key insight is that _the variance of a layer's output should be approximately equal to the variance of its input._

**Forward Pass Variance Preservation:**  
$$Var(output) ≈ Var(input)$$  

**Backward Pass Variance Preservation:**  
$$Var(gradient) ≈ Var(upstream\_gradient)$$

### $Common~Initialization~Methods$

#### 1. Zero Initialization ❌
```python
W = np.zeros((n_in, n_out))
```
**Problem**: All neurons learn identical features due to _perfect symmetry_, essentially learning the same parameters, which makes it no different from a regular _machine learning model_ ($1$ layer, $1$ neuron).

#### 2. Random Small Values ❌
```python
W = np.random.randn(n_in, n_out) * 0.01
```
**Problem**: Activations and gradients shrink exponentially in deep networks, which casues _vanishing gradients_.

#### 3. Xavier/Glorot Initialization ✅
**Best for**: Tanh and Sigmoid activation functions

**Normal Distribution:**
```python
W = np.random.randn(n_in, n_out) * np.sqrt(1.0 / n_in)
```

**Uniform Distribution:**
```python
limit = np.sqrt(6.0 / (n_in + n_out))
W = np.random.uniform(-limit, limit, (n_in, n_out))
```

**Mathematical Justification:**
- Maintains unit variance for both forward and backward pass
- Derived assuming linear activations (works well for tanh/sigmoid)

#### 4. He Initialization ✅
**Best for**: ReLU and its variants

**Normal Distribution:**
```python
W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
```

**Uniform Distribution:**
```python
limit = np.sqrt(6.0 / n_in)
W = np.random.uniform(-limit, limit, (n_in, n_out))
```

**Mathematical Justification:**
- Accounts for the fact that ReLU kills half the neurons on average
- Factor of 2 compensates for the reduced variance due to ReLU

#### 5. Modern Initialization Strategies

**LeCun Initialization** (for SELU):
```python
W = np.random.randn(n_in, n_out) * np.sqrt(1.0 / n_in)
```

**Orthogonal Initialization** (for RNNs):
```python
# Uses orthogonal matrices to prevent vanishing/exploding gradients
W = orthogonal_matrix(n_in, n_out)
```

### Bias Initialization

**Standard Practice:**
```python
b = np.zeros(n_out)  # Initialize biases to zero
```

**Exception for ReLU:**
```python
b = np.full(n_out, 0.01)  # Small positive bias to ensure initial activation
```  

## $\text{Regularization Techniques}$

Regularization prevents overfitting by adding constraints or noise to the learning process, helping models generalize better to unseen data.

### $1.~L_1~and~L_2~Regularization$

#### L2 Regularization (Weight Decay)
**Most Common**: Penalizes large weights by adding their squared magnitude to the loss.

**Mathematical Form:**
$$ J_{regularized} = J_{original} + \frac{\lambda}{2m} ~ \sum^m W^2 $$

**Implementation:**
```python
# In loss function
l2_penalty = 0.5 * lambda_reg * np.sum(W**2)
total_loss = original_loss + l2_penalty

# In gradient computation
dW += lambda_reg * W
```

**Benefits:**
- Prevents weights from becoming too large
- Encourages weight sharing
- Smooth decision boundaries

#### L1 Regularization (Lasso)
**Sparsity-Inducing**: Promotes sparse weights (many weights become exactly zero).

**Mathematical Form:**
$$ J_{regularized} = J_{original} + \frac{\lambda}{m} ~ \sum^m \abs{W} $$

**Implementation:**
```python
# In loss function
l1_penalty = lambda_reg * np.sum(np.abs(W))
total_loss = original_loss + l1_penalty

# In gradient computation
dW += lambda_reg * np.sign(W)
```

**Benefits:**
- Automatic feature selection
- Sparse models (smaller memory footprint)
- Interpretable models

### $2.~Dropout$

**Core Idea**: Randomly set a fraction of neurons to zero during training, forcing the network to learn redundant representations.

**Mathematical Formulation:**  
During training:  
```python
h_dropout = h * mask / keep_prob
```
where $\text{mask} \approx \text{Bernoulli(keep\_prob)} $  

During inference:  
```python
h_inference = h  # No dropout, but scaled during training
```

**Implementation:**
```python
def dropout_forward(X, keep_prob, training=True):
    if training:
        mask = np.random.binomial(1, keep_prob, X.shape) / keep_prob
        return X * mask, mask
    else:
        return X, None

def dropout_backward(dout, mask):
    return dout * mask
```

**Key Parameters:**
- **keep_prob**: Probability of keeping a neuron (typically 0.5-0.8)
- **Inverted dropout**: Scale activations during training (modern approach)  

The division by `keep_prob` is called **inverted dropout** and it's crucial for maintaining the expected value of activations:  
>**Without Inverted Dropout:**  
>- **Training:** `E[X_dropout] = keep_prob * E[X]` (scaled down)
>- **Testing:** `E[X_test] = E[X]` (original scale)
>- **Problem:** _Different scales_ between training and testing!
>
>**With Inverted Dropout:**  
>- **Training:** `E[X_dropout] = E[X * mask / keep_prob] = E[X] * E[mask] / keep_prob = E[X] * keep_prob / keep_prob = E[X]`
>- **Testing:** `E[X_test] = E[X]` (no dropout applied)
>- **Benefit:** _Same expected scale_ in both training and testing!  

**Modern Standard:**  
The inverted dropout (scaling during training) is now the standard approach because:  
1. **No inference overhead:** No need to scale during testing
2. **Cleaner implementation:** Test time is just forward pass without modifications
3. **Framework compatibility:** All major frameworks (PyTorch, TensorFlow) use this approach

**Benefits:**
- Reduces overfitting significantly
- Improves generalization
- Acts as ensemble method (averaging multiple sub-networks)

**Modern Considerations:**
- Often not needed with batch normalization
- Can increase training time
- Less effective in very deep networks with proper normalization

### $3.~Batch~Normalization$

**Revolutionary Technique**: Normalizes _inputs to each layer_, dramatically improving training stability and speed.

**Mathematical Formulation:**
$$ 
\begin{align} 
\text{Batch mean: } \quad & \mu_B = \frac 1 m \times \sum^m X \\
\text{Batch variance: } \quad & \sigma^2_B = \frac 1 m \times \sum^m~(X - \mu_B)^2 \\      
\text{Normalize: } \quad & \hat X = \frac{X - \mu_B} {\sqrt{\sigma^2_B + \epsilon}} \\
\text{Scale and shift: } \quad & Y = \gamma \times \hat X + \beta 
\end{align} 
$$

**Implementation:**
```python
def batch_norm_forward(X, gamma, beta, eps=1e-8):
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0)
    X_norm = (X - mu) / np.sqrt(var + eps)
    out = gamma * X_norm + beta
    
    # Cache for backward pass
    cache = (X, X_norm, mu, var, gamma, beta, eps)
    return out, cache
```

**Benefits:**
- **Accelerates training**: Often 2-10x faster convergence
- **Reduces sensitivity to initialization**: Can use higher learning rates
- **Regularization effect**: Reduces need for dropout
- **Gradient flow**: Helps with vanishing gradient problem

**Modern Usage:**
- Standard in most CNN architectures
- Applied after linear transformation, before activation
- Learnable parameters γ (scale) and β (shift)

### 4. Early Stopping

**Simple yet Effective**: Stop training when validation performance starts degrading.

**Implementation Strategy:**
```python
def early_stopping(val_losses, patience=10, min_delta=1e-4):
    if len(val_losses) < patience:
        return False
    
    recent_losses = val_losses[-patience:]
    best_loss = min(val_losses[:-patience])
    
    return all(loss > best_loss + min_delta for loss in recent_losses)
```

**Benefits:**
- Prevents overfitting without hyperparameter tuning
- Computationally efficient
- Works with any model architecture

---

## Gradient Checking

**Purpose**: Verify that your backpropagation implementation is correct by comparing analytical gradients with numerical gradients.

### Mathematical Foundation

**Numerical Gradient (Two-sided difference):**
```
∂J/∂θ ≈ (J(θ + ε) - J(θ - ε)) / (2ε)
```

where ε is a small value (typically 1e-7).

### Implementation

```python
def gradient_check(parameters, gradients, X, Y, epsilon=1e-7):
    """
    Perform gradient checking
    
    Args:
        parameters: dictionary of parameters
        gradients: dictionary of computed gradients
        X: input data
        Y: true labels
        epsilon: small value for numerical differentiation
    
    Returns:
        difference: relative difference between numerical and analytical gradients
    """
    
    # Flatten parameters and gradients
    params_vector = dictionary_to_vector(parameters)
    grad_vector = dictionary_to_vector(gradients)
    
    num_parameters = params_vector.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    
    # Compute numerical gradient
    for i in range(num_parameters):
        
        # Compute J_plus[i]
        theta_plus = np.copy(params_vector)
        theta_plus[i] = theta_plus[i] + epsilon
        J_plus[i], _ = forward_propagation(X, vector_to_dictionary(theta_plus))
        
        # Compute J_minus[i]
        theta_minus = np.copy(params_vector)
        theta_minus[i] = theta_minus[i] - epsilon
        J_minus[i], _ = forward_propagation(X, vector_to_dictionary(theta_minus))
        
        # Compute numerical gradient
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
    
    # Compute relative difference
    numerator = np.linalg.norm(grad_vector - gradapprox)
    denominator = np.linalg.norm(grad_vector) + np.linalg.norm(gradapprox)
    difference = numerator / denominator
    
    return difference

def dictionary_to_vector(parameters):
    """Convert parameter dictionary to vector"""
    keys = []
    count = 0
    for key in parameters.keys():
        new_vector = np.reshape(parameters[key], (-1, 1))
        keys.append(key)
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count += 1
    
    return theta, keys

def vector_to_dictionary(theta, keys):
    """Convert vector back to parameter dictionary"""
    parameters = {}
    start = 0
    
    for key in keys:
        # Get original shape
        shape = original_shapes[key]  # You need to store original shapes
        size = np.prod(shape)
        
        # Extract and reshape
        parameters[key] = theta[start:start+size].reshape(shape)
        start += size
    
    return parameters
```

### Interpretation of Results

**Gradient Check Tolerance:**
- **difference < 1e-7**: Excellent! Your implementation is likely correct
- **1e-7 < difference < 1e-5**: Good. Probably correct, but double-check
- **1e-5 < difference < 1e-3**: Warning. Likely a bug in backpropagation
- **difference > 1e-3**: Error. Definitely a bug in your implementation

### Common Issues and Debugging Tips

1. **Regularization**: Don't forget to include regularization terms in both forward and backward pass
2. **Dropout**: Turn off dropout during gradient checking
3. **Batch Normalization**: Use the same batch for both forward passes
4. **Numerical Precision**: Use double precision (float64) for gradient checking
5. **Random Initialization**: Use fixed random seed for reproducibility

---

## Modern Best Practices

### 2024 Recommendations

1. **Initialization Strategy:**
   - **CNNs**: He initialization with ReLU activations
   - **Transformers**: Xavier initialization with layer normalization
   - **RNNs**: Orthogonal initialization for recurrent connections

2. **Regularization Hierarchy:**
   - **First choice**: Batch/Layer Normalization
   - **Second choice**: Weight decay (L2 regularization)
   - **Third choice**: Dropout (if needed)
   - **Always**: Early stopping

3. **Gradient Checking:**
   - Essential during development
   - Use only on small subsets of data
   - Disable all stochastic elements (dropout, batch norm in training mode)

### Framework-Specific Implementations

**PyTorch:**
```python
import torch.nn as nn

# He initialization
nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

# Xavier initialization
nn.init.xavier_uniform_(layer.weight)

# Regularization
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10)
)
```

**TensorFlow/Keras:**
```python
from tensorflow.keras import layers, initializers

model = tf.keras.Sequential([
    layers.Dense(256, activation='relu', 
                kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```

---

## Implementation Examples

### Complete Example: Deep Neural Network with All Techniques

```python
import numpy as np
import matplotlib.pyplot as plt

class DeepNeuralNetwork:
    def __init__(self, layer_dims, initialization='he', regularization=None, 
                 lambda_reg=0.01, keep_prob=0.8, use_batch_norm=True):
        """
        Initialize a deep neural network
        
        Args:
            layer_dims: list of layer dimensions [n_x, n_h1, n_h2, ..., n_y]
            initialization: 'he', 'xavier', or 'random'
            regularization: None, 'l1', 'l2', or 'dropout'
            lambda_reg: regularization parameter
            keep_prob: dropout keep probability
            use_batch_norm: whether to use batch normalization
        """
        self.layer_dims = layer_dims
        self.L = len(layer_dims) - 1
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.keep_prob = keep_prob
        self.use_batch_norm = use_batch_norm
        
        # Initialize parameters
        self.parameters = self._initialize_parameters(initialization)
        
        # Batch normalization parameters
        if use_batch_norm:
            self.bn_params = self._initialize_batch_norm()
    
    def _initialize_parameters(self, method):
        """Initialize parameters based on chosen method"""
        parameters = {}
        
        for l in range(1, self.L + 1):
            if method == 'he':
                # He initialization for ReLU
                parameters[f'W{l}'] = np.random.randn(
                    self.layer_dims[l], self.layer_dims[l-1]
                ) * np.sqrt(2.0 / self.layer_dims[l-1])
                
            elif method == 'xavier':
                # Xavier initialization for tanh/sigmoid
                parameters[f'W{l}'] = np.random.randn(
                    self.layer_dims[l], self.layer_dims[l-1]
                ) * np.sqrt(1.0 / self.layer_dims[l-1])
                
            elif method == 'random':
                # Random small initialization
                parameters[f'W{l}'] = np.random.randn(
                    self.layer_dims[l], self.layer_dims[l-1]
                ) * 0.01
            
            # Initialize biases
            parameters[f'b{l}'] = np.zeros((self.layer_dims[l], 1))
        
        return parameters
    
    def _initialize_batch_norm(self):
        """Initialize batch normalization parameters"""
        bn_params = {}
        
        for l in range(1, self.L):  # Not for output layer
            bn_params[f'gamma{l}'] = np.ones((self.layer_dims[l], 1))
            bn_params[f'beta{l}'] = np.zeros((self.layer_dims[l], 1))
        
        return bn_params
    
    def forward_propagation(self, X, training=True):
        """Forward propagation with all techniques"""
        caches = []
        A = X
        
        # Hidden layers
        for l in range(1, self.L):
            A_prev = A
            
            # Linear transformation
            Z = np.dot(self.parameters[f'W{l}'], A_prev) + self.parameters[f'b{l}']
            
            # Batch normalization
            if self.use_batch_norm:
                Z, bn_cache = self._batch_norm_forward(Z, l, training)
            else:
                bn_cache = None
            
            # Activation (ReLU)
            A = np.maximum(0, Z)
            
            # Dropout
            if self.regularization == 'dropout' and training:
                A, dropout_cache = self._dropout_forward(A, self.keep_prob)
            else:
                dropout_cache = None
            
            cache = (A_prev, Z, A, bn_cache, dropout_cache)
            caches.append(cache)
        
        # Output layer
        A_prev = A
        ZL = np.dot(self.parameters[f'W{self.L}'], A_prev) + self.parameters[f'b{self.L}']
        AL = 1 / (1 + np.exp(-ZL))  # Sigmoid activation
        
        cache = (A_prev, ZL, AL, None, None)
        caches.append(cache)
        
        return AL, caches
    
    def _batch_norm_forward(self, Z, l, training, eps=1e-8):
        """Batch normalization forward pass"""
        if training:
            mu = np.mean(Z, axis=1, keepdims=True)
            var = np.var(Z, axis=1, keepdims=True)
            
            Z_norm = (Z - mu) / np.sqrt(var + eps)
            Z_out = (self.bn_params[f'gamma{l}'] * Z_norm + 
                    self.bn_params[f'beta{l}'])
            
            cache = (Z, Z_norm, mu, var, eps)
            return Z_out, cache
        else:
            # Use running statistics for inference
            return Z, None
    
    def _dropout_forward(self, A, keep_prob):
        """Dropout forward pass"""
        mask = np.random.binomial(1, keep_prob, A.shape) / keep_prob
        A_drop = A * mask
        return A_drop, mask
    
    def compute_cost(self, AL, Y):
        """Compute cost with regularization"""
        m = Y.shape[1]
        
        # Cross-entropy cost
        cost = -(1/m) * np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL))
        
        # Add regularization
        if self.regularization == 'l2':
            l2_cost = 0
            for l in range(1, self.L + 1):
                l2_cost += np.sum(np.square(self.parameters[f'W{l}']))
            cost += (self.lambda_reg / (2 * m)) * l2_cost
            
        elif self.regularization == 'l1':
            l1_cost = 0
            for l in range(1, self.L + 1):
                l1_cost += np.sum(np.abs(self.parameters[f'W{l}']))
            cost += (self.lambda_reg / m) * l1_cost
        
        return cost
    
    def backward_propagation(self, AL, Y, caches):
        """Backward propagation with all techniques"""
        grads = {}
        m = AL.shape[1]
        
        # Output layer
        dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
        
        # Backpropagate through all layers
        for l in reversed(range(1, self.L + 1)):
            A_prev, Z, A, bn_cache, dropout_cache = caches[l-1]
            
            if l == self.L:
                # Output layer
                dZ = AL - Y
            else:
                # Hidden layers
                dA = dAL if l == self.L else dA_next
                
                # Dropout backward
                if dropout_cache is not None:
                    dA = dA * dropout_cache
                
                # ReLU backward
                dZ = dA * (Z > 0)
                
                # Batch normalization backward
                if bn_cache is not None:
                    dZ = self._batch_norm_backward(dZ, bn_cache, l)
            
            # Linear backward
            dW = (1/m) * np.dot(dZ, A_prev.T)
            db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            dA_prev = np.dot(self.parameters[f'W{l}'].T, dZ)
            
            # Add regularization to gradients
            if self.regularization == 'l2':
                dW += (self.lambda_reg / m) * self.parameters[f'W{l}']
            elif self.regularization == 'l1':
                dW += (self.lambda_reg / m) * np.sign(self.parameters[f'W{l}'])
            
            grads[f'dW{l}'] = dW
            grads[f'db{l}'] = db
            
            if l > 1:
                dA_next = dA_prev
        
        return grads
    
    def _batch_norm_backward(self, dZ_out, cache, l):
        """Batch normalization backward pass"""
        Z, Z_norm, mu, var, eps = cache
        m = Z.shape[1]
        
        # Gradients for gamma and beta
        dgamma = np.sum(dZ_out * Z_norm, axis=1, keepdims=True)
        dbeta = np.sum(dZ_out, axis=1, keepdims=True)
        
        # Gradient for normalized input
        dZ_norm = dZ_out * self.bn_params[f'gamma{l}']
        
        # Gradient for input
        dvar = np.sum(dZ_norm * (Z - mu) * -0.5 * (var + eps)**(-3/2), axis=1, keepdims=True)
        dmu = np.sum(dZ_norm * -1 / np.sqrt(var + eps), axis=1, keepdims=True) + dvar * np.sum(-2 * (Z - mu), axis=1, keepdims=True) / m
        
        dZ = dZ_norm / np.sqrt(var + eps) + dvar * 2 * (Z - mu) / m + dmu / m
        
        # Store gradients for batch norm parameters
        self.bn_grads = getattr(self, 'bn_grads', {})
        self.bn_grads[f'dgamma{l}'] = dgamma
        self.bn_grads[f'dbeta{l}'] = dbeta
        
        return dZ
    
    def update_parameters(self, grads, learning_rate):
        """Update parameters using gradient descent"""
        for l in range(1, self.L + 1):
            self.parameters[f'W{l}'] -= learning_rate * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= learning_rate * grads[f'db{l}']
        
        # Update batch normalization parameters
        if self.use_batch_norm and hasattr(self, 'bn_grads'):
            for l in range(1, self.L):
                self.bn_params[f'gamma{l}'] -= learning_rate * self.bn_grads[f'dgamma{l}']
                self.bn_params[f'beta{l}'] -= learning_rate * self.bn_grads[f'dbeta{l}']
    
    def train(self, X, Y, X_val, Y_val, learning_rate=0.01, num_epochs=1000, print_cost=True):
        """Train the neural network"""
        costs = []
        val_costs = []
        
        for epoch in range(num_epochs):
            # Forward propagation
            AL, caches = self.forward_propagation(X, training=True)
            
            # Compute cost
            cost = self.compute_cost(AL, Y)
            costs.append(cost)
            
            # Backward propagation
            grads = self.backward_propagation(AL, Y, caches)
            
            # Update parameters
            self.update_parameters(grads, learning_rate)
            
            # Validation cost
            AL_val, _ = self.forward_propagation(X_val, training=False)
            val_cost = self.compute_cost(AL_val, Y_val)
            val_costs.append(val_cost)
            
            # Print cost
            if print_cost and epoch % 100 == 0:
                print(f"Cost after epoch {epoch}: {cost:.6f}, Val cost: {val_cost:.6f}")
        
        return costs, val_costs
    
    def predict(self, X):
        """Make predictions"""
        AL, _ = self.forward_propagation(X, training=False)
        predictions = (AL > 0.5).astype(int)
        return predictions
```

### Usage Example

```python
# Create dataset
np.random.seed(42)
X_train = np.random.randn(784, 1000)  # 784 features, 1000 samples
Y_train = (np.random.randn(1, 1000) > 0).astype(int)  # Binary labels

X_val = np.random.randn(784, 200)
Y_val = (np.random.randn(1, 200) > 0).astype(int)

# Create and train model
model = DeepNeuralNetwork(
    layer_dims=[784, 256, 128, 64, 1],
    initialization='he',
    regularization='l2',
    lambda_reg=0.01,
    keep_prob=0.8,
    use_batch_norm=True
)

# Train model
costs, val_costs = model.train(
    X_train, Y_train, X_val, Y_val,
    learning_rate=0.01,
    num_epochs=1000
)

# Make predictions
predictions = model.predict(X_val)
accuracy = np.mean(predictions == Y_val)
print(f"Validation accuracy: {accuracy:.4f}")
```

---

## Key Takeaways

1. **Parameter Initialization**: Use He initialization for ReLU networks, Xavier for tanh/sigmoid networks
2. **Regularization**: Batch normalization is often sufficient; add L2 regularization and dropout as needed
3. **Gradient Checking**: Essential for debugging; use numerical gradients to verify analytical gradients
4. **Modern Practice**: Combine techniques thoughtfully - batch normalization often reduces need for dropout
5. **Implementation**: Always implement gradient checking first, then optimize for performance