# $\text{Complete Mathematical Derivation: L-Layer Neural Network}$
## $\text{Network Architecture and Notation}$
### Network Structure
- Input: $X \in \mathbb{R}^{n^{[0]} \times m}$ where:
    - $n^{[0]}$ = number of features
    - $m$ = number of examples
- Layers: $L$ layers total (including output layer)
- Layer $l$ has $n^{[l]}$ neurons for $l = 1, 2, \ldots, L$
- Parameters:
    - $W^{[l]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}$ (weight matrix for layer $l$)
    - $b^{[l]} \in \mathbb{R}^{n^{[l]} \times 1}$ (bias vector for layer $l$)
- Activations:
    - $A^{[l]} \in \mathbb{R}^{n^{[l]} \times m}$ (activation matrix for layer $l$)
    - $A^{[0]} = X$ (input layer)

## $\text{Forward Propagation}$

For each layer $l = 1, 2, \ldots, L$:  
**Linear Transformation:** $Z^{[l]}=W^{[l]}A^{[l−1]}+b{[l]}$  
**Activation Function:** $A^{[l]}=g^{[l]}(Z{[l]})$  
Where $g^{[l]}$ is the activation function for layer $l$.  

### Complete Forward Pass:

$$\begin{align}
Z^{[1]} &= W^{[1]} X + b^{[1]} &\quad  A^{[1]} &= g^{[1]}(Z^{[1]}) \\
Z^{[2]} &= W^{[2]} A^{[1]} + b^{[2]} &\quad A^{[2]} &= g^{[2]}(Z^{[2]}) \\
&\vdots & &\vdots \\
Z^{[L]} &= W^{[L]} A^{[L-1]} + b^{[L]} &\quad A^{[L]} &= g^{[L]}(Z^{[L]}) \quad \text{(Final output)}
\end{align}$$  

The final prediction is $\hat{Y} = A^{[L]}$.  

## $\text{Cost Function}$

For **binary cross-entropy** (logistic regression output):  

$$J = -\frac{1}{m} \sum_{i=1}^{m}~[ Y^{(i)} \log(A^{[L] (i)}) + (1-Y^{(i)}) \log(1-A^{[L] (i)})]$$  

For **mean squared error**:

$$J = \frac{1}{2m} \sum_{i=1}^{m} ||A^{[L] (i)} - Y^{(i)}||^2$$

## $\text{Backward Propagation: Complete Mathematical Derivation}$

### Step 1: Derivative with respect to Output Layer Activations

For **binary cross-entropy**:  

$$\frac{\partial J}{\partial A^{[L]}} = -\frac{1}{m} \left[ \frac{Y}{A^{[L]}} - \frac{1-Y}{1-A^{[L]}} \right]$$  

For **mean squared error**:  

$$\frac{\partial J}{\partial A^{[L]}} = \frac{1}{m} (A^{[L]} - Y)$$  

### Step 2: Derivative with respect to Output Layer Pre-activations

Using the chain rule:  

$$\frac{\partial J}{\partial Z^{[L]}} = \frac{\partial J}{\partial A^{[L]}} \cdot \frac{\partial A^{[L]}}{\partial Z^{[L]}}$$  

Since $A^{[L]} = g^{[L]}(Z^{[L]})$:  

$$\frac{\partial A^{[L]}}{\partial Z^{[L]}} = g'^{[L]}(Z^{[L]})$$  

Therefore:  

$$dZ^{[L]} = \frac{\partial J}{\partial Z^{[L]}} = \frac{\partial J}{\partial A^{[L]}} \odot g'^{[L]}(Z^{[L]})$$  

**Special case for sigmoid + cross-entropy:**  
When $g^{[L]}(z) = \sigma(z) = \frac{1}{1+e^{-z}}$ and using cross-entropy:  

$$dZ^{[L]} = A^{[L]} - Y$$  

### Step 3: Derivatives with respect to Parameters of Layer L

**Weight derivatives:**  

$$\frac{\partial J}{\partial W^{[L]}} = \frac{\partial J}{\partial Z^{[L]}} \cdot \frac{\partial Z^{[L]}}{\partial W^{[L]}}$$  

Since $Z^{[L]} = W^{[L]} A^{[L-1]} + b^{[L]}$:  

$$\frac{\partial Z^{[L]}}{\partial W^{[L]}} = A^{[L-1]T}$$  

Therefore:  

$$dW^{[L]} = \frac{1}{m} dZ^{[L]} (A^{[L-1]})^T$$  

**Bias derivatives:**  

$$\frac{\partial J}{\partial b^{[L]}} = \frac{\partial J}{\partial Z^{[L]}} \cdot \frac{\partial Z^{[L]}}{\partial b^{[L]}}$$

Since $\frac{\partial Z^{[L]}}{\partial b^{[L]}} = \mathbf{1}$ (broadcasting):

$$db^{[L]} = \frac{1}{m} \text{sum}(dZ^{[L]}, \text{axis}=1, \text{keepdims}=\text{True})$$

### Step 4: Derivative with respect to Previous Layer Activations

$$\frac{\partial J}{\partial A^{[L-1]}} = \frac{\partial J}{\partial Z^{[L]}} \cdot \frac{\partial Z^{[L]}}{\partial A^{[L-1]}}$$

Since $Z^{[L]} = W^{[L]} A^{[L-1]} + b^{[L]}$:  

$$\frac{\partial Z^{[L]}}{\partial A^{[L-1]}} = (W^{[L]})^T$$  

Therefore:

$$dA^{[L-1]} = (W^{[L]})^T dZ^{[L]}$$  

### Step 5: General Recursive Formula for Hidden Layers

For any layer $l$ where $1 \leq l < L$:

**Pre-activation derivatives:**  

$$dZ^{[l]} = dA^{[l]} \odot g'^{[l]}(Z^{[l]})$$

**Weight derivatives:**  

$$dW^{[l]} = \frac{1}{m} dZ^{[l]} (A^{[l-1]})^T$$

**Bias derivatives:**  

$$db^{[l]} = \frac{1}{m} \text{sum}(dZ^{[l]}, \text{axis}=1, \text{keepdims}=\text{True})$$

**Previous layer activation derivatives:**  

$$dA^{[l-1]} = (W^{[l]})^T dZ^{[l]}$$

## $\text{Complete Backward Propagation Algorithm}$
### Mathematical Formulation:

$\text{Output Layer:}$  

$$\begin{align}
dZ^{[L]} &= \frac{\partial J}{\partial A^{[L]}} \odot g'^{[L]}(Z^{[L]}) \\
dW^{[L]} &= \frac{1}{m} dZ^{[L]} (A^{[L-1]})^T \\
db^{[L]} &= \frac{1}{m} \sum_{i=1}^{m} dZ^{[L] (\cdot,i)} \\
dA^{[L-1]} &= (W^{[L]})^T dZ^{[L]} \\
\end{align}$$

$\text{Hidden Layers } ( L-1 \geq l \geq 1)\text{: }$  

$$\begin{align}
dZ^{[l]} &= dA^{[l]} \odot g'^{[l]}(Z^{[l]}) \\
dW^{[l]} &= \frac{1}{m} dZ^{[l]} (A^{[l-1]})^T \\
db^{[l]} &= \frac{1}{m} \sum_{i=1}^{m} dZ^{[l] (\cdot,i)} \\
dA^{[l-1]} &= (W^{[l]})^T dZ^{[l]} \quad \text{(if } l > 1\text{)}
\end{align}$$

## Activation Functions and Their Derivatives

### ReLU Activation:
$$g(z) = \max(0, z)$$
$$g'(z) = \begin{cases} 
1 & \text{if } z > 0 \\
0 & \text{if } z \leq 0 
\end{cases}$$

### Sigmoid Activation:
$$g(z) = \sigma(z) = \frac{1}{1 + e^{-z}}$$
$$g'(z) = \sigma(z)(1 - \sigma(z))$$

### Hyperbolic Tangent:
$$g(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$
$$g'(z) = 1 - \tanh^2(z)$$

### Leaky ReLU:
$$g(z) = \begin{cases} 
z & \text{if } z > 0 \\
\alpha z & \text{if } z \leq 0 
\end{cases}$$
$$g'(z) = \begin{cases} 
1 & \text{if } z > 0 \\
\alpha & \text{if } z \leq 0 
\end{cases}$$

## $\text{Dimensional Analysis}$

For layer $l$ with $n^{[l]}$ neurons and $n^{[l-1]}$ neurons in the previous layer:

### Forward Propagation Dimensions:
- $Z^{[l]} \in \mathbb{R}^{n^{[l]} \times m}$
- $W^{[l]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}$
- $A^{[l-1]} \in \mathbb{R}^{n^{[l-1]} \times m}$
- $b^{[l]} \in \mathbb{R}^{n^{[l]} \times 1}$

**Verification:**  

$$W^{[l]} A^{[l-1]} + b^{[l]} \rightarrow (n^{[l]} \times n^{[l-1]}) \cdot (n^{[l-1]} \times m) + (n^{[l]} \times 1) = (n^{[l]} \times m)$$

### Backward Propagation Dimensions:
- $dZ^{[l]} \in \mathbb{R}^{n^{[l]} \times m}$
- $dW^{[l]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}$ (same as $W^{[l]}$)
- $db^{[l]} \in \mathbb{R}^{n^{[l]} \times 1}$ (same as $b^{[l]}$)
- $dA^{[l-1]} \in \mathbb{R}^{n^{[l-1]} \times m}$ (same as $A^{[l-1]}$)

**Verification:**  

$$dZ^{[l]} (A^{[l-1]})^T \rightarrow (n^{[l]} \times m) \cdot (m \times n^{[l-1]}) = (n^{[l]} \times n^{[l-1]})$$  
$$(W^{[l]})^T dZ^{[l]} \rightarrow (n^{[l-1]} \times n^{[l]}) \cdot (n^{[l]} \times m) = (n^{[l-1]} \times m)$$

## Matrix Calculus Foundations

### Key Matrix Derivatives:
For $Z = WA + b$:  

$$\frac{\partial Z}{\partial W} = A^T, \quad \frac{\partial Z}{\partial A} = W^T, \quad \frac{\partial Z}{\partial b} = \mathbf{1}$$

### Chain Rule in Matrix Form:  

$$\frac{\partial J}{\partial W^{[l]}} = \frac{\partial J}{\partial Z^{[l]}} \frac{\partial Z^{[l]}}{\partial W^{[l]}} = dZ^{[l]} (A^{[l-1]})^T$$

### Vectorization Benefits:
- Process all $m$ training examples simultaneously
- Efficient GPU computation through BLAS operations
- Reduced computational complexity from $O(mn^2L)$ to $O(n^2L)$ per iteration

## Parameter Update Rule

After computing all gradients:  

$$W^{[l]} := W^{[l]} - \alpha \cdot dW^{[l]}$$
$$b^{[l]} := b^{[l]} - \alpha \cdot db^{[l]}$$

Where $\alpha$ is the learning rate.
