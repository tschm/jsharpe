"""JSharpe: Sharpe Ratio Analysis and Statistical Testing.

This package provides comprehensive tools for Sharpe ratio analysis,
including statistical significance testing, multiple testing corrections,
and portfolio optimization utilities.

Key features:
    - Sharpe ratio variance estimation under non-Gaussian returns
    - Minimum track record length computation
    - Probabilistic Sharpe Ratio (PSR) calculation
    - False Discovery Rate (FDR) control for multiple strategy testing
    - Family-Wise Error Rate (FWER) corrections (Bonferroni, Šidák, Holm)
    - Minimum variance portfolio optimization

Example:
    >>> import numpy as np
    >>> from jsharpe.sharpe import sharpe_ratio_variance, probabilistic_sharpe_ratio
    >>> # Compute variance of a Sharpe ratio estimate
    >>> var = sharpe_ratio_variance(SR=0.5, T=24)
    >>> # Compute the Probabilistic Sharpe Ratio
    >>> psr = probabilistic_sharpe_ratio(SR=0.5, SR0=0, T=24)
"""
