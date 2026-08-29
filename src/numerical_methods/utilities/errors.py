def error_calculate(vv: float, va: float, type: str = "rel_abs"):
    """
    Calculate the error between a reference value and an approximate value.

    Parameters
    ----------
    vv : float
        Reference (true) value.
    va : float
        Approximate value.
    type : str, default="rel_abs"
        Error type. Options are:
        - "abs" : absolute error.
        - "rel_abs" : absolute relative error.
        - "rel_frac" : signed relative error.

    Returns
    -------
    float
        Calculated error.

    Raises
    ------
    ValueError
        If `type` is not one of "abs", "rel_abs", or "rel_frac".
    """

    type = type.lower()

    if type in ("rel_abs", "rel_frac") and vv == 0:
        raise ValueError("ERROR: Relative error (vv) is undefined when the exact value is zero.")

    if type == "abs":
        error = abs(vv - va)

    elif type == "rel_abs":
        error = abs((vv - va) / vv)

    elif type == "rel_frac":
        error = (vv - va) / vv

    else:
        raise ValueError("ERROR: Enter a valid error type in the 'type' field. The options are 'abs', 'rel_abs', and 'rel_frac'.")

    return error


def objective_functions(Qsim, Qobs, N: int, type: str = "MSE"):

    MSE = 1 / N
    for i in range(N):
        MSE += (Qsim[i] - Qobs[i]) ** 2

    return MSE