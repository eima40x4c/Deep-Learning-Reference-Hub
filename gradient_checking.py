import numpy as np

def gradient_check(parameters, gradients, X, Y, epsilon=1e-7):
    """
    Perform gradient checking to verify analytical gradients against numerical gradients.
    
    Args:
        parameters (dict): Dictionary of parameters (e.g., {'W1': array, 'b1': array, ...})
        gradients (dict): Dictionary of computed analytical gradients
        X (np.array): Input data
        Y (np.array): True labels
        cost_function (callable): Function that computes cost given (X, Y, parameters)
        epsilon (float): Small value for numerical differentiation
    
    Returns:
        float: Relative difference between numerical and analytical gradients
               - < 1e-7: Excellent (gradients are likely correct)
               - < 1e-5: Good (gradients are probably correct)
               - < 1e-3: Acceptable (check implementation)
               - > 1e-3: Poor (likely bug in gradient computation)
    """    
    params_vector, param_shapes = dictionary_to_vector(parameters)
    grad_vector, _ = dictionary_to_vector(gradients)
    
    num_parameters = params_vector.shape[0]
    gradapprox = np.zeros((num_parameters, 1))
    
    # Assuming `cost_function` is already implemented`
    for i in range(num_parameters):
        theta_plus = np.copy(params_vector)
        theta_plus[i] = theta_plus[i] + epsilon
        J_plus = cost_function(vector_to_dictionary(theta_plus, param_shapes))
        
        theta_minus = np.copy(params_vector)
        theta_minus[i] = theta_minus[i] - epsilon
        J_minus = cost_function(vector_to_dictionary(theta_minus, param_shapes))
        
        gradapprox[i] = (J_plus - J_minus) / (2 * epsilon)  # Numerical Gradient
    
    # Relative Difference Computation
    numerator = np.linalg.norm(grad_vector - gradapprox)
    denominator = np.linalg.norm(grad_vector) + np.linalg.norm(gradapprox)
    
    if denominator == 0:
        return 0.0
    difference = numerator / denominator
    
    print(f"Gradient Check Results:")
    print(f"  Numerical gradient norm: {np.linalg.norm(gradapprox):.6f}")
    print(f"  Analytical gradient norm: {np.linalg.norm(grad_vector):.6f}")
    print(f"  Relative difference: {difference:.2e}")
    
    if difference < 1e-7:
        print("  ✅ Excellent! Gradients are likely correct.")
    elif difference < 1e-5:
        print("  ✅ Good! Gradients are probably correct.")
    elif difference < 1e-3:
        print("  ⚠️  Acceptable, but check your implementation.")
    else:
        print("  ❌ Poor! Likely bug in gradient computation.")
    
    return difference


def dictionary_to_vector(parameters: dict):
    """
    Convert parameter dictionary to a single vector while preserving shape information.
    
    Args:
        parameters (dict): Dictionary with parameter names as keys and numpy arrays as values
    
    Returns:
        tuple: (theta, shapes) where:
            - theta: Single column vector containing all parameters
            - shapes: Dictionary mapping parameter names to their original shapes
    """
    shapes = {}
    theta = None
    
    for key in sorted(parameters.keys()):  # Sort for consistent ordering
        shapes[key] = parameters[key].shape
        
        param_vector = np.reshape(parameters[key], (-1, 1))
        if theta is None:
            theta = param_vector
        else:
            theta = np.concatenate((theta, param_vector), axis=0)
    
    return theta, shapes


def vector_to_dictionary(theta, shapes):
    """
    Convert a parameter vector back to dictionary format using stored shapes.
    
    Args:
        theta (np.array): Column vector containing all parameters
        shapes (dict): Dictionary mapping parameter names to their original shapes
    
    Returns:
        dict: Dictionary with parameter names as keys and reshaped arrays as values
    """
    parameters = {}
    start = 0
    
    for key in sorted(shapes.keys()):
        shape = shapes[key]
        size = np.prod(shape)  # Total number of elements
        
        parameters[key] = theta[start:start+size].reshape(shape)
        start += size
    
    return parameters
