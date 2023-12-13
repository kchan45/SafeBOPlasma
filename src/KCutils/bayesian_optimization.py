'''
custom Bayesian optimization utilities

Requirements:
* Python 3
* CasADi [https://web.casadi.org]
* PyTorch [https://pytorch.org]
* BoTorch [https://botorch.org] and Ax [https://ax.dev]
* Matplotlib [https://matplotlib.org] (for data visualization)

Copyright (c) 2023 Mesbah Lab. All Rights Reserved.

Author(s): Kimberly Chan

This file is under the MIT License. A copy of this license is included in the
download of the entire code package (within the root folder of the package).
'''

import math
from typing import Optional, Any, Dict
import torch
from torch import Tensor

from ax.storage.botorch_modular_registry import register_acquisition_function

from botorch.models.gpytorch import GPyTorchModel
from botorch.models import FixedNoiseGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from botorch.acquisition.monte_carlo import MCAcquisitionFunction, qUpperConfidenceBound
from botorch.models.model import Model, ModelList
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils import t_batch_mode_transform
# additional imports to register custom acquisition functions with BoTorch
from ax.models.torch.botorch_modular.optimizer_argparse import optimizer_argparse
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.input_constructors import MaybeDict, acqf_input_constructor, construct_inputs_analytic_base
from botorch.acquisition.objective import AcquisitionObjective, PosteriorTransform, MCAcquisitionObjective
from botorch.utils.datasets import SupervisedDataset
from botorch.sampling.base import MCSampler

# register the built-in upper confidence bound acqusition function from BoTorch to Ax
register_acquisition_function(qUpperConfidenceBound)

class qUpperConfidenceBoundWithLogBarrier(qUpperConfidenceBound):
    def __init__(
        self,
        model: ModelList,
        ucb_beta: float,
        tau: float,
        barrier_beta: float,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        super().__init__(
            model=model,
            beta=ucb_beta,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self.register_buffer("ucb_beta", torch.as_tensor(ucb_beta))
        self.register_buffer("tau", torch.as_tensor(tau))
        self.register_buffer("barrier_beta", torch.as_tensor(barrier_beta))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate custom UCB on the candidate set 'X'.

        Args:
            X: A '(b) x q x d'-dim Tensor of '(b)' t-batches with 'q' 'd'-dim
                design points each

        Returns:
            Tensor: A '(b)'-dim Tensor of custom UCB values at the given design
                points 'X'.
        """

        # print(self.model.models)
        posterior = self.model.posterior(X)
        samples = self.get_posterior_samples(posterior) # n x b x q x o
        mean = posterior.mean
        beta_prime = math.sqrt(self.ucb_beta * math.pi / 2)
        general_ucb = mean[:,:,0] + beta_prime * (samples[:,:,:,0] - mean[:,:,0]).abs()

        var = (samples[:,:,:,1:] - mean[:,:,1:]).abs()
        barrier_beta_prime = math.sqrt(self.barrier_beta * math.pi / 2)
        cons_ucb = torch.relu(mean[:,:,1:] + barrier_beta_prime * var)
        barrier_term = (torch.log(1e-20 + cons_ucb)).sum(dim=-1)
        custom_ucb_samples = general_ucb + self.tau * barrier_term
        return custom_ucb_samples.max(dim=-1)[0].mean(dim=0)

register_acquisition_function(qUpperConfidenceBoundWithLogBarrier)

from botorch.acquisition.objective import ScalarizedPosteriorTransform

@acqf_input_constructor(qUpperConfidenceBoundWithLogBarrier)
def construct_inputs_qUCBWLB(
    model: ModelList,
    training_data: MaybeDict[SupervisedDataset],
    objective_thresholds: Tensor,
    objective: Optional[AcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    sampler: Optional[MCSampler] = None,
    ucb_beta: float = 0.2,
    tau: float = 1e-3,
    barrier_beta: float = 0.2,
    **kwargs: Any,
) -> Dict[str, Any]:
    r""" Construct kwargs for the 'qUpperConfidenceBoundWithLogBarrier' constructor.
    Modeled after the 'qUpperConfidenceBound' under botorch.acquisition.input_constructors

    Args:
        model: the model to be used in the acquisition function
        training_data: dataset(s) used to train the model
        ucb_beta: scalar representing the trade-off parameter between mean and
            covariance for the UCB portion of this custom acquisition function
        tau: scalar weight of the barrier term
        barrier_beta: scalar representing the trade-off parameter between mean
            covarianve for the barrier portion of this custom acquisition function
        maximize: If true, consider the problem a maximization problem

    Returns:
        a dict mapping kwarg names of the constructor to values
    """
    weights = torch.tensor([1.0, 0.0], dtype=torch.double)
    posterior_transform = ScalarizedPosteriorTransform(weights)
    base_inputs = construct_inputs_analytic_base(
        model=model,
        training_data=training_data,
        posterior_transform=posterior_transform,
        **kwargs,
    )
    return {**base_inputs, "ucb_beta": ucb_beta, "tau": tau, "barrier_beta": barrier_beta}


class qUpperConfidenceBoundWithLogBarrierRelaxed(qUpperConfidenceBound):
    def __init__(
        self,
        model: ModelList,
        ucb_beta: float,
        tau: float,
        barrier_beta: float,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        super().__init__(
            model=model,
            beta=ucb_beta,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self.register_buffer("ucb_beta", torch.as_tensor(ucb_beta))
        self.register_buffer("tau", torch.as_tensor(tau))
        self.register_buffer("barrier_beta", torch.as_tensor(barrier_beta))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate custom UCB on the candidate set 'X'.

        Args:
            X: A '(b) x q x d'-dim Tensor of '(b)' t-batches with 'q' 'd'-dim
                design points each

        Returns:
            Tensor: A '(b)'-dim Tensor of custom UCB values at the given design
                points 'X'.
        """

        posterior = self.model.posterior(X)
        samples = self.get_posterior_samples(posterior) # n x b x q x o
        mean = posterior.mean
        beta_prime = math.sqrt(self.ucb_beta * math.pi / 2)
        general_ucb = mean[:,:,0] + beta_prime * (samples[:,:,:,0] - mean[:,:,0]).abs()

        var = (samples[:,:,:,1:] - mean[:,:,1:]).abs()
        barrier_beta_prime = math.sqrt(self.barrier_beta * math.pi / 2)
        cons_ucb = torch.relu(mean[:,:,1:] - barrier_beta_prime * var)
        barrier_term = (torch.log(1e-20 + cons_ucb)).sum(dim=-1)
        custom_ucb_samples = general_ucb + self.tau * barrier_term
        return custom_ucb_samples.max(dim=-1)[0].mean(dim=0)

register_acquisition_function(qUpperConfidenceBoundWithLogBarrierRelaxed)

@acqf_input_constructor(qUpperConfidenceBoundWithLogBarrierRelaxed)
def construct_inputs_qUCBWLBrx(
    model: ModelList,
    training_data: MaybeDict[SupervisedDataset],
    objective_thresholds: Tensor,
    objective: Optional[AcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    sampler: Optional[MCSampler] = None,
    ucb_beta: float = 0.2,
    tau: float = 1e-3,
    barrier_beta: float = 0.2,
    **kwargs: Any,
) -> Dict[str, Any]:
    r""" Construct kwargs for the 'qUpperConfidenceBoundWithLogBarrierRelaxed' constructor.
    Modeled after the 'qUpperConfidenceBound' under botorch.acquisition.input_constructors

    Args:
        model: the model to be used in the acquisition function
        training_data: dataset(s) used to train the model
        ucb_beta: scalar representing the trade-off parameter between mean and
            covariance for the UCB portion of this custom acquisition function
        tau: scalar weight of the barrier term
        barrier_beta: scalar representing the trade-off parameter between mean
            covarianve for the barrier portion of this custom acquisition function
        maximize: If true, consider the problem a maximization problem

    Returns:
        a dict mapping kwarg names of the constructor to values
    """
    weights = torch.tensor([1.0, 0.0], dtype=torch.double)
    posterior_transform = ScalarizedPosteriorTransform(weights)
    base_inputs = construct_inputs_analytic_base(
        model=model,
        training_data=training_data,
        posterior_transform=posterior_transform,
        **kwargs,
    )
    return {**base_inputs, "ucb_beta": ucb_beta, "tau": tau, "barrier_beta": barrier_beta}
