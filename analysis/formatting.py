import numpy
import math


def err_brackets(
    quantity, error, error_prec=2, form="Mix", texify=False, decimal_in_error=False
):
    """Formats quantites with errors in the form x.xxx(yy)e+aa.

    Parameters:
        quantity (float):  Input quantity.
        error (float):     Error/uncertainty of quantity in same units. Negative error is treated as positive (x -> |x|).
        error_prec (int):  Number of significant figures the error is taken to.
        form (str):        'Sci' or 'Mix'. 'Sci' forces scientifc notation while 'Mix' (default) allows regular notation for small numbers.
        texify (bool):     Replace 'e+aa' with '$\times 10^{aa}$', for TeX compatibility.
        decimal_in_error(bool): Place a decimal in the error brackets. E.g., False: 32.482(34), True: 32.484(0.034).

    Returns:
        result: The string with the formatted quantity.

    """
    quantity = float(quantity)
    error = float(error)
    if error_prec < 1 or (not isinstance(error_prec, int)):
        raise ValueError("error_prec must be an integer larger than 1")

    error = abs(error)

    if error != error or quantity != quantity:  # Check for NaN
        return "NaN"

    if error == 0:  # easy case
        return f"{quantity:{format_symb:}}"

    quantity_sig, quantity_exp = frexp10(quantity)
    error_sig, error_exp = frexp10(error)

    # number of sig figs in quantity needed
    exp_diff = quantity_exp - error_exp
    quantity_prec = exp_diff + error_prec
    rounded = False  # flag to change precision if error rounds an order

    # check for exp required; if so, force to Sci, else allow Mix if chosen
    check_q = (
        (len(f"{quantity:#.{quantity_prec:}g}".split("e"))) > 1
        if quantity_prec > 0
        else False
    )
    check_e = (len(f"{error:#.{error_prec:}g}".split("e"))) > 1
    is_exp_required = check_q or check_e

    # Two Modes: Sci and Mix
    # Mix
    if form == "Mix" and not is_exp_required:
        # error
        # with decimal
        if decimal_in_error == True:
            error_string = f"({error:#.{error_prec}g})"
        # normal
        else:
            error_temp_str = str(round(error_sig * 10 ** (error_prec - 1)))
            # if error rounds up an order, drop a digit and flag
            if len(error_temp_str) > error_prec:
                rounded = True
            error_temp = int(error_temp_str[:error_prec])
            error_string = f"({error_temp:})"
        # if error rounds up an order, need to drop a sigfig in quantity
        if rounded:
            quantity_prec -= 1
        # significand
        # normal
        if exp_diff >= 0 and quantity_prec > 0:
            quantity_string = f"{quantity:#.{quantity_prec}g}"
        # if error is larger than quantity
        elif quantity_prec <= 0:
            quantity_string = "0." + "0" * (error_prec + 1)
        else:
            # Leading zeros AFTER negative signs
            if quantity < 0:
                quantity_string = (
                    "-" + "0" * (-exp_diff) + f"{abs(quantity):#.{quantity_prec}g}"
                )
            else:
                # quantity_string = "0" * (-exp_diff) + f"{quantity:#.{quantity_prec}g}"
                print(f"{quantity_exp=}")
                print(f"{error_exp=}")
                print(f"{quantity_sig=}")
                print(f"{error_sig=}")
                print(f"{exp_diff=}")
                print(f"{quantity_prec=}\n")
                # print(exp_diff)
                # check if we need leading zeros
                if quantity_exp + error_prec >= error_prec:
                    quantity_string = (
                        "0" * (-exp_diff) + f"{quantity:#.{quantity_prec}g}"
                    )
                else:
                    quantity_string = f"{quantity:#.{quantity_prec}g}"

        # of course, no exp needed
        exp_string = ""

    # Sci
    if form == "Sci" or is_exp_required:
        # error
        # normal
        if quantity_prec > 0:
            error_exp_shift = -exp_diff
        # when signal supressed, shift magnitudes into order of error
        else:
            error_exp_shift = 0
        if decimal_in_error == True and (error_exp_shift - error_prec + 1) <= 0:
            error_temp = error_sig * 10 ** (error_exp_shift)
            error_string = f"({error_temp:#.{-error_exp_shift+error_prec-1}f})"
        elif decimal_in_error == True and (error_exp_shift - error_prec + 1) > 0:
            error_temp = round(error_sig * 10 ** (error_prec - 1)) * 10 ** (
                error_exp_shift - error_prec + 1
            )
            error_string = f"({error_temp:})"
        else:
            error_temp_str = str(round(error_sig * 10 ** (error_prec - 1)))
            # if error rounds up an order, drop a digit and flag
            if len(error_temp_str) > error_prec:
                rounded = True
            error_temp = int(error_temp_str[:error_prec])
            error_string = f"({error_temp:})"

        # if error rounds up an order, need to drop a sigfig in quantity
        if rounded:
            quantity_prec -= 1
        # significand
        # normal
        if exp_diff >= 0 or (
            exp_diff < 0 and decimal_in_error == True and quantity_prec > 0
        ):
            quantity_string = f"{quantity_sig:#.{quantity_prec-1:}f}"
            exp_out = quantity_exp
        # need to add leading zeros AFTER negative signs
        elif exp_diff < 0 and decimal_in_error == False and quantity_prec > 0:
            if quantity_sig < 0:
                quantity_string = (
                    "-"
                    + "0" * (-exp_diff)
                    + f"{abs(quantity_sig):#.{quantity_prec-1}f}"
                )
            else:
                quantity_string = (
                    "0" * (-exp_diff) + f"{quantity_sig:#.{quantity_prec-1}f}"
                )
            exp_out = quantity_exp
        # signal supressed, create zero significand
        else:
            quantity_string = "0." + "0" * (error_prec - 1)
            exp_out = error_exp

        # exponent
        if texify == True:
            exp_string = r"$\times 10^" + f"{{{exp_out:}}}$"
        else:
            exp_string = f"e{exp_out:0=+3d}"

    # print(quantity_string+error_string+exp_string)
    return quantity_string + error_string + exp_string


def frexp10(float_in):
    """Returns the mantissa and exponent in base 10 of input float."""
    # ADDED: abs()
    exponent = math.floor(math.log10(abs(float_in)))
    significand = float_in / (10**exponent)

    return tuple([significand, exponent])
