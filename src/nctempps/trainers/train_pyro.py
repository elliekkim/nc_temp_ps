"""
Helpful training functions for LGCP models.

"""

import tqdm
import pyro


def train_pyro_model(
    model, X, counts, population, lr=0.01, num_particles=32, num_iter=300, lrd=0.99
):
    """
    Boiler plate code for training pyro-based models.

    Parameters:
    model : pyro-based model with .model and .guide methods
    X : torch.Tensor
        Input coordinates of shape (N, D)
    counts : torch.Tensor
        Observed counts of shape (N,)
    population : torch.Tensor
        Population of shape (N,)
    lr : float
        Learning rate for the optimizer.
    num_particles : int
        Number of particles for the ELBO estimator.
    num_iter : int
        Number of training iterations.
    lrd : float
        Learning rate decay factor.
    Returns
    -------
    losses : list of float
        List of loss values at each iteration.
    """
    losses = []
    pyro.clear_param_store()

    optimizer = pyro.optim.ClippedAdam({"lr": lr, "lrd": lrd})
    loss = pyro.infer.Trace_ELBO(
        num_particles=num_particles, vectorize_particles=True, retain_graph=True
    )
    infer = pyro.infer.SVI(model.model, model.guide, optimizer, loss=loss)

    model.train()
    itr = tqdm.tqdm(range(num_iter))
    for i in itr:
        loss = infer.step(X, counts, population)
        losses.append(loss)
        itr.set_postfix(loss=loss)
    return losses
