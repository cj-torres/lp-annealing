from SimpleCNN import SimpleCNN
from mdl_methods import L0_Regularizer

# Defaults taken from mdl_methods.py
def getSimpleCNNWithL0( 
    lam=0.01,
    weight_decay=0.0,
    temperature=2/3,
    droprate_init=0.2,
    limit_a=-0.1,
    limit_b=1.1,
    epsilon=1e-6
):
    """
    Creates a SimpleCNN model wrapped with L0_Regularizer.

    Returns:
        L0_Regularizer: Wrapped SimpleCNN model with L0 regularization.
    """
    # Instantiate the SimpleCNN model
    simple_cnn = SimpleCNN()
    
    # Wrap the SimpleCNN model with L0_Regularizer
    l0_model = L0_Regularizer(
        original_module=simple_cnn,
        lam=lam,
        weight_decay=weight_decay,
        temperature=temperature,
        droprate_init=droprate_init,
        limit_a=limit_a,
        limit_b=limit_b,
        epsilon=epsilon
    )
    
    return l0_model