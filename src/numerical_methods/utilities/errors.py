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

    type = type.upper()

    if type == "MSE":
        MSE = 0
    
        for i in range(N):
            MSE += (Qsim[i] - Qobs[i]) ** 2

        return 1 / N * MSE

    elif type == "RMSE":
        return objective_functions(Qsim, Qobs, N, "MSE") ** (0.5)

    elif type == "MAE":
        MAE = 0

        for i in range(N):
            MAE += abs(Qsim[i] - Qobs[i])

        return 1 / N * MAE

    elif type == "NSE":
        NSE = 0
        Qobs_average = sum(Qobs) / len(Qobs)

        for i in range(N):
            NSE += (Qsim[i] - Qobs[i]) ** 2 / (Qobs[i] - Qobs_average) ** 2

        return 1 - NSE

    elif type == "R²":
        R = 0
        Qobs_average = sum(Qobs) / len(Qobs)
        Qsim_average = sum(Qsim) / len(Qsim)

        for i in range(N):
            R += (Qsim[i] - Qsim_average) * (Qobs[i] - Qobs_average) / ((Qsim[i] - Qsim_average) ** 2 * (Qobs[i] - Qobs_average) ** 2) ** (0.5)

        return R ** 2

    elif type == "MBE":
        MBE = 0

        for i in range(N):
            MBE += (Qsim[i] - Qobs[i])

        return 1 / N * MBE
    
    else:
        raise ValueError("ERROR: Enter a valid error type in the 'type' field.")