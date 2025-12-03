"""
Utilities for KL divergence and discretized Gaussian log-likelihood computations.

Includes:
- KL divergence between two normal distributions.
- Approximate standard normal CDF.
- Discretized Gaussian log-likelihood for pixel values.
"""

import numpy as np
import torch as th


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two normal distributions.

    KL(N1 || N2), where N1 ~ N(mean1, exp(logvar1)) and N2 ~ N(mean2, exp(logvar2)).

    Parameters:
        mean1 (Tensor or float): Mean of the first distribution.
        logvar1 (Tensor or float): Log-variance of the first distribution.
        mean2 (Tensor or float): Mean of the second distribution.
        logvar2 (Tensor or float): Log-variance of the second distribution.

    Returns:
        Tensor: The KL divergence.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "At least one argument must be a Tensor"

    # Convert log-variances to tensors if needed
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    Approximate the standard normal CDF using a tanh-based approximation.

    Parameters:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Approximated CDF values.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute log-likelihood of data x under a discretized Gaussian distribution.

    This is typically used for modeling 8-bit images in diffusion models.

    Parameters:
        x (Tensor): Input values in [-1, 1], shape (N, C, H, W).
        means (Tensor): Predicted means, same shape as x.
        log_scales (Tensor): Predicted log standard deviations, same shape as x.

    Returns:
        Tensor: Log-likelihood of x under the predicted Gaussian.
    """
    assert x.shape == means.shape == log_scales.shape, "Shapes of x, means, and log_scales must match"

    centered_x = x - means
    inv_stdv = th.exp(-log_scales)

    # Compute CDF at x + 1/255 and x - 1/255
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)

    cdf_plus = approx_standard_normal_cdf(plus_in)
    cdf_min = approx_standard_normal_cdf(min_in)

    # Handle edge cases near -1 and 1
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))

    cdf_delta = cdf_plus - cdf_min
    mid_log = th.log(cdf_delta.clamp(min=1e-12))

    log_probs = th.where(
        x < -0.999, log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, mid_log)
    )

    assert log_probs.shape == x.shape
    return log_probs
