import pandas as pd
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import seaborn as sns


def read_data(file_path):
    """
    Read data from CSV file and identify DMUs, inputs, and outputs.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        df (DataFrame): Pandas DataFrame containing the data
        dmu_col (str): Column name for DMUs
        input_cols (list): List of input column names
        output_cols (list): List of output column names
    """
    try:
        df = pd.read_csv(file_path)
        print(
            f"Successfully loaded data with {df.shape[0]} DMUs and {df.shape[1]} columns"
        )

        # Assume first column is DMU name
        dmu_col = df.columns[0]

        # Ask user to identify input and output columns
        print("\nAvailable columns:", ", ".join(df.columns[1:]))

        input_cols = input("Enter input column names (comma-separated): ").split(",")
        input_cols = [col.strip() for col in input_cols]

        output_cols = input("Enter output column names (comma-separated): ").split(",")
        output_cols = [col.strip() for col in output_cols]

        # Validate that specified columns exist
        for col in input_cols + output_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in the dataset")

        return df, dmu_col, input_cols, output_cols

    except Exception as e:
        print(f"Error reading data: {e}")
        return None, None, None, None


def run_ccr_model(df, dmu_col, input_cols, output_cols, orientation="input"):
    """
    Run CCR model (constant returns to scale) DEA analysis.

    Args:
        df (DataFrame): Pandas DataFrame containing the data
        dmu_col (str): Column name for DMUs
        input_cols (list): List of input column names
        output_cols (list): List of output column names
        orientation (str): 'input' for input-oriented or 'output' for output-oriented

    Returns:
        results_df (DataFrame): DataFrame containing efficiency scores and reference sets
    """
    n_dmus = df.shape[0]
    n_inputs = len(input_cols)
    n_outputs = len(output_cols)

    # Normalize data to handle different scales
    input_data = df[input_cols].values
    output_data = df[output_cols].values

    # Create results DataFrame
    results = {
        "DMU": df[dmu_col].tolist(),
        "Efficiency": np.zeros(n_dmus),
        "Reference Set": [[] for _ in range(n_dmus)],
        "Input Slacks": [np.zeros(n_inputs) for _ in range(n_dmus)],
        "Output Slacks": [np.zeros(n_outputs) for _ in range(n_dmus)],
    }

    # For each DMU, solve a linear programming problem
    for i in range(n_dmus):
        if orientation == "input":
            efficiency, ref_set, input_slacks, output_slacks = solve_input_oriented_lp(
                i, input_data, output_data, n_dmus, n_inputs, n_outputs
            )
        else:  # output-oriented
            efficiency, ref_set, input_slacks, output_slacks = solve_output_oriented_lp(
                i, input_data, output_data, n_dmus, n_inputs, n_outputs
            )

        results["Efficiency"][i] = efficiency

        # Record reference DMUs (indices where lambda > 0)
        dmu_names = df[dmu_col].tolist()
        results["Reference Set"][i] = [dmu_names[j] for j in ref_set]
        results["Input Slacks"][i] = input_slacks
        results["Output Slacks"][i] = output_slacks

    # Create results DataFrame
    results_df = pd.DataFrame(
        {
            "DMU": results["DMU"],
            "Efficiency Score": results["Efficiency"],
            "Reference Set": results["Reference Set"],
        }
    )

    # Add slack columns
    for i, col in enumerate(input_cols):
        results_df[f"Slack_{col}"] = [slacks[i] for slacks in results["Input Slacks"]]

    for i, col in enumerate(output_cols):
        results_df[f"Slack_{col}"] = [slacks[i] for slacks in results["Output Slacks"]]

    return results_df


def solve_input_oriented_lp(
    dmu_idx, input_data, output_data, n_dmus, n_inputs, n_outputs
):
    """
    Solve input-oriented DEA linear programming problem for a specific DMU.

    Args:
        dmu_idx (int): Index of the DMU being evaluated
        input_data (ndarray): Array of input values
        output_data (ndarray): Array of output values
        n_dmus (int): Number of DMUs
        n_inputs (int): Number of inputs
        n_outputs (int): Number of outputs

    Returns:
        efficiency (float): Efficiency score
        ref_set (list): Indices of reference DMUs
        input_slacks (ndarray): Input slack values
        output_slacks (ndarray): Output slack values
    """
    # The objective is to minimize the efficiency score theta
    # Decision variables: [theta, lambda_1, lambda_2, ..., lambda_n]
    c = np.zeros(1 + n_dmus)
    c[0] = 1  # Minimize theta

    # Constraints for inputs: theta*x_i0 - sum(lambda_j * x_ij) >= 0
    A_ub_inputs = np.zeros((n_inputs, 1 + n_dmus))
    b_ub_inputs = np.zeros(n_inputs)

    for i in range(n_inputs):
        A_ub_inputs[i, 0] = input_data[dmu_idx, i]
        A_ub_inputs[i, 1:] = -input_data[:, i]

    # Constraints for outputs: sum(lambda_j * y_rj) - y_r0 >= 0
    A_ub_outputs = np.zeros((n_outputs, 1 + n_dmus))
    b_ub_outputs = np.zeros(n_outputs)

    for i in range(n_outputs):
        A_ub_outputs[i, 0] = 0
        A_ub_outputs[i, 1:] = -output_data[:, i]
        b_ub_outputs[i] = -output_data[dmu_idx, i]

    # Combine constraints
    A_ub = np.vstack((A_ub_inputs, A_ub_outputs))
    b_ub = np.concatenate((b_ub_inputs, b_ub_outputs))

    # Lambda variables must be non-negative
    bounds = [(0, None)] * (1 + n_dmus)

    # Solve the linear programming problem
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if result.success:
        theta = result.x[0]
        lambdas = result.x[1:]

        # Calculate slacks in a second phase
        # Decision variables are slack variables for inputs and outputs
        c_slacks = np.ones(n_inputs + n_outputs)

        # Constraints for inputs: theta*x_i0 - sum(lambda_j * x_ij) - s_i^- = 0
        A_eq_inputs = np.zeros((n_inputs, n_inputs + n_outputs))
        b_eq_inputs = np.zeros(n_inputs)

        for i in range(n_inputs):
            A_eq_inputs[i, i] = 1
            b_eq_inputs[i] = theta * input_data[dmu_idx, i] - np.sum(
                lambdas * input_data[:, i]
            )

        # Constraints for outputs: sum(lambda_j * y_rj) - y_r0 - s_r^+ = 0
        A_eq_outputs = np.zeros((n_outputs, n_inputs + n_outputs))
        b_eq_outputs = np.zeros(n_outputs)

        for i in range(n_outputs):
            A_eq_outputs[i, n_inputs + i] = 1
            b_eq_outputs[i] = (
                np.sum(lambdas * output_data[:, i]) - output_data[dmu_idx, i]
            )

        # Combine constraints
        A_eq = np.vstack((A_eq_inputs, A_eq_outputs))
        b_eq = np.concatenate((b_eq_inputs, b_eq_outputs))

        # All slack variables must be non-negative
        bounds_slacks = [(0, None)] * (n_inputs + n_outputs)

        # Solve for slack variables
        result_slacks = linprog(
            c_slacks, A_eq=A_eq, b_eq=b_eq, bounds=bounds_slacks, method="highs"
        )

        if result_slacks.success:
            input_slacks = result_slacks.x[:n_inputs]
            output_slacks = result_slacks.x[n_inputs:]
        else:
            input_slacks = np.zeros(n_inputs)
            output_slacks = np.zeros(n_outputs)

        # Find reference set (DMUs with non-zero lambda values)
        epsilon = 1e-6  # Threshold for considering lambda as non-zero
        ref_set = [j for j in range(n_dmus) if lambdas[j] > epsilon]

        return theta, ref_set, input_slacks, output_slacks
    else:
        print(f"Warning: Optimization for DMU {dmu_idx} failed")
        return 1.0, [], np.zeros(n_inputs), np.zeros(n_outputs)


def solve_output_oriented_lp(
    dmu_idx, input_data, output_data, n_dmus, n_inputs, n_outputs
):
    """
    Solve output-oriented DEA linear programming problem for a specific DMU.

    Args:
        dmu_idx (int): Index of the DMU being evaluated
        input_data (ndarray): Array of input values
        output_data (ndarray): Array of output values
        n_dmus (int): Number of DMUs
        n_inputs (int): Number of inputs
        n_outputs (int): Number of outputs

    Returns:
        efficiency (float): Efficiency score (1/phi)
        ref_set (list): Indices of reference DMUs
        input_slacks (ndarray): Input slack values
        output_slacks (ndarray): Output slack values
    """
    # The objective is to maximize the efficiency score phi
    # Decision variables: [phi, lambda_1, lambda_2, ..., lambda_n]
    # For linprog (minimization), use -phi
    c = np.zeros(1 + n_dmus)
    c[0] = -1  # Maximize phi (minimize -phi)

    # Constraints for inputs: sum(lambda_j * x_ij) - x_i0 <= 0
    A_ub_inputs = np.zeros((n_inputs, 1 + n_dmus))
    b_ub_inputs = np.zeros(n_inputs)

    for i in range(n_inputs):
        A_ub_inputs[i, 0] = 0
        A_ub_inputs[i, 1:] = input_data[:, i]
        b_ub_inputs[i] = input_data[dmu_idx, i]

    # Constraints for outputs: phi*y_r0 - sum(lambda_j * y_rj) <= 0
    A_ub_outputs = np.zeros((n_outputs, 1 + n_dmus))
    b_ub_outputs = np.zeros(n_outputs)

    for i in range(n_outputs):
        A_ub_outputs[i, 0] = output_data[dmu_idx, i]
        A_ub_outputs[i, 1:] = -output_data[:, i]

    # Combine constraints
    A_ub = np.vstack((A_ub_inputs, A_ub_outputs))
    b_ub = np.concatenate((b_ub_inputs, b_ub_outputs))

    # Phi must be >= 1, lambda variables must be non-negative
    bounds = [(1, None)] + [(0, None)] * n_dmus

    # Solve the linear programming problem
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if result.success:
        phi = result.x[0]
        lambdas = result.x[1:]

        # Calculate slacks in a second phase
        # Decision variables are slack variables for inputs and outputs
        c_slacks = np.ones(n_inputs + n_outputs)

        # Constraints for inputs: sum(lambda_j * x_ij) - x_i0 + s_i^- = 0
        A_eq_inputs = np.zeros((n_inputs, n_inputs + n_outputs))
        b_eq_inputs = np.zeros(n_inputs)

        for i in range(n_inputs):
            A_eq_inputs[i, i] = 1
            b_eq_inputs[i] = np.sum(lambdas * input_data[:, i]) - input_data[dmu_idx, i]

        # Constraints for outputs: phi*y_r0 - sum(lambda_j * y_rj) + s_r^+ = 0
        A_eq_outputs = np.zeros((n_outputs, n_inputs + n_outputs))
        b_eq_outputs = np.zeros(n_outputs)

        for i in range(n_outputs):
            A_eq_outputs[i, n_inputs + i] = 1
            b_eq_outputs[i] = phi * output_data[dmu_idx, i] - np.sum(
                lambdas * output_data[:, i]
            )

        # Combine constraints
        A_eq = np.vstack((A_eq_inputs, A_eq_outputs))
        b_eq = np.concatenate((b_eq_inputs, b_eq_outputs))

        # All slack variables must be non-negative
        bounds_slacks = [(0, None)] * (n_inputs + n_outputs)

        # Solve for slack variables
        result_slacks = linprog(
            c_slacks, A_eq=A_eq, b_eq=b_eq, bounds=bounds_slacks, method="highs"
        )

        if result_slacks.success:
            input_slacks = result_slacks.x[:n_inputs]
            output_slacks = result_slacks.x[n_inputs:]
        else:
            input_slacks = np.zeros(n_inputs)
            output_slacks = np.zeros(n_outputs)

        # Find reference set (DMUs with non-zero lambda values)
        epsilon = 1e-6  # Threshold for considering lambda as non-zero
        ref_set = [j for j in range(n_dmus) if lambdas[j] > epsilon]

        # Efficiency = 1/phi for output-oriented model
        return 1.0 / phi, ref_set, input_slacks, output_slacks
    else:
        print(f"Warning: Optimization for DMU {dmu_idx} failed")
        return 1.0, [], np.zeros(n_inputs), np.zeros(n_outputs)


def run_bcc_model(df, dmu_col, input_cols, output_cols, orientation="input"):
    """
    Run BCC model (variable returns to scale) DEA analysis.

    Args:
        df (DataFrame): Pandas DataFrame containing the data
        dmu_col (str): Column name for DMUs
        input_cols (list): List of input column names
        output_cols (list): List of output column names
        orientation (str): 'input' for input-oriented or 'output' for output-oriented

    Returns:
        results_df (DataFrame): DataFrame containing efficiency scores and reference sets
    """
    n_dmus = df.shape[0]
    n_inputs = len(input_cols)
    n_outputs = len(output_cols)

    # Normalize data to handle different scales
    input_data = df[input_cols].values
    output_data = df[output_cols].values

    # Create results DataFrame
    results = {
        "DMU": df[dmu_col].tolist(),
        "Efficiency": np.zeros(n_dmus),
        "Reference Set": [[] for _ in range(n_dmus)],
        "Input Slacks": [np.zeros(n_inputs) for _ in range(n_dmus)],
        "Output Slacks": [np.zeros(n_outputs) for _ in range(n_dmus)],
        "Returns to Scale": [""] * n_dmus,
    }

    # For each DMU, solve a linear programming problem
    for i in range(n_dmus):
        if orientation == "input":
            efficiency, ref_set, input_slacks, output_slacks, rts = (
                solve_bcc_input_oriented_lp(
                    i, input_data, output_data, n_dmus, n_inputs, n_outputs
                )
            )
        else:  # output-oriented
            efficiency, ref_set, input_slacks, output_slacks, rts = (
                solve_bcc_output_oriented_lp(
                    i, input_data, output_data, n_dmus, n_inputs, n_outputs
                )
            )

        results["Efficiency"][i] = efficiency

        # Record reference DMUs (indices where lambda > 0)
        dmu_names = df[dmu_col].tolist()
        results["Reference Set"][i] = [dmu_names[j] for j in ref_set]
        results["Input Slacks"][i] = input_slacks
        results["Output Slacks"][i] = output_slacks
        results["Returns to Scale"][i] = rts

    # Create results DataFrame
    results_df = pd.DataFrame(
        {
            "DMU": results["DMU"],
            "Efficiency Score": results["Efficiency"],
            "Reference Set": results["Reference Set"],
            "Returns to Scale": results["Returns to Scale"],
        }
    )

    # Add slack columns
    for i, col in enumerate(input_cols):
        results_df[f"Slack_{col}"] = [slacks[i] for slacks in results["Input Slacks"]]

    for i, col in enumerate(output_cols):
        results_df[f"Slack_{col}"] = [slacks[i] for slacks in results["Output Slacks"]]

    return results_df


def solve_bcc_input_oriented_lp(
    dmu_idx, input_data, output_data, n_dmus, n_inputs, n_outputs
):
    """
    Solve BCC input-oriented DEA linear programming problem for a specific DMU.

    Args:
        dmu_idx (int): Index of the DMU being evaluated
        input_data (ndarray): Array of input values
        output_data (ndarray): Array of output values
        n_dmus (int): Number of DMUs
        n_inputs (int): Number of inputs
        n_outputs (int): Number of outputs

    Returns:
        efficiency (float): Efficiency score
        ref_set (list): Indices of reference DMUs
        input_slacks (ndarray): Input slack values
        output_slacks (ndarray): Output slack values
        rts (str): Returns to scale
    """
    # The objective is to minimize the efficiency score theta
    # Decision variables: [theta, lambda_1, lambda_2, ..., lambda_n]
    c = np.zeros(1 + n_dmus)
    c[0] = 1  # Minimize theta

    # Constraints for inputs: theta*x_i0 - sum(lambda_j * x_ij) >= 0
    A_ub_inputs = np.zeros((n_inputs, 1 + n_dmus))
    b_ub_inputs = np.zeros(n_inputs)

    for i in range(n_inputs):
        A_ub_inputs[i, 0] = input_data[dmu_idx, i]
        A_ub_inputs[i, 1:] = -input_data[:, i]

    # Constraints for outputs: sum(lambda_j * y_rj) - y_r0 >= 0
    A_ub_outputs = np.zeros((n_outputs, 1 + n_dmus))
    b_ub_outputs = np.zeros(n_outputs)

    for i in range(n_outputs):
        A_ub_outputs[i, 0] = 0
        A_ub_outputs[i, 1:] = -output_data[:, i]
        b_ub_outputs[i] = -output_data[dmu_idx, i]

    # Combine constraints
    A_ub = np.vstack((A_ub_inputs, A_ub_outputs))
    b_ub = np.concatenate((b_ub_inputs, b_ub_outputs))

    # Convexity constraint: sum(lambda_j) = 1
    A_eq = np.zeros((1, 1 + n_dmus))
    A_eq[0, 1:] = 1
    b_eq = np.array([1])

    # Theta and lambda variables must be non-negative
    bounds = [(0, None)] * (1 + n_dmus)

    # Solve the linear programming problem
    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )

    if result.success:
        theta = result.x[0]
        lambdas = result.x[1:]

        # Calculate slacks in a second phase
        # Decision variables are slack variables for inputs and outputs
        c_slacks = np.ones(n_inputs + n_outputs)

        # Constraints for inputs: theta*x_i0 - sum(lambda_j * x_ij) - s_i^- = 0
        A_eq_inputs = np.zeros((n_inputs, n_inputs + n_outputs))
        b_eq_inputs = np.zeros(n_inputs)

        for i in range(n_inputs):
            A_eq_inputs[i, i] = 1
            b_eq_inputs[i] = theta * input_data[dmu_idx, i] - np.sum(
                lambdas * input_data[:, i]
            )

        # Constraints for outputs: sum(lambda_j * y_rj) - y_r0 - s_r^+ = 0
        A_eq_outputs = np.zeros((n_outputs, n_inputs + n_outputs))
        b_eq_outputs = np.zeros(n_outputs)

        for i in range(n_outputs):
            A_eq_outputs[i, n_inputs + i] = 1
            b_eq_outputs[i] = (
                np.sum(lambdas * output_data[:, i]) - output_data[dmu_idx, i]
            )

        # Combine constraints
        A_eq_slacks = np.vstack((A_eq_inputs, A_eq_outputs))
        b_eq_slacks = np.concatenate((b_eq_inputs, b_eq_outputs))

        # All slack variables must be non-negative
        bounds_slacks = [(0, None)] * (n_inputs + n_outputs)

        # Solve for slack variables
        result_slacks = linprog(
            c_slacks,
            A_eq=A_eq_slacks,
            b_eq=b_eq_slacks,
            bounds=bounds_slacks,
            method="highs",
        )

        if result_slacks.success:
            input_slacks = result_slacks.x[:n_inputs]
            output_slacks = result_slacks.x[n_inputs:]
        else:
            input_slacks = np.zeros(n_inputs)
            output_slacks = np.zeros(n_outputs)

        # Find reference set (DMUs with non-zero lambda values)
        epsilon = 1e-6  # Threshold for considering lambda as non-zero
        ref_set = [j for j in range(n_dmus) if lambdas[j] > epsilon]

        # Determine returns to scale
        # Add a small positive/negative value to the RHS of the convexity constraint
        # and see how the objective function (efficiency) changes

        # Test for increasing returns to scale (sum lambda <= 1)
        A_eq_inc = A_eq.copy()
        b_eq_inc = np.array([1 - 1e-6])

        result_inc = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq_inc,
            b_eq=b_eq_inc,
            bounds=bounds,
            method="highs",
        )

        # Test for decreasing returns to scale (sum lambda >= 1)
        A_ub_dec = np.zeros((1, 1 + n_dmus))
        A_ub_dec[0, 1:] = -1
        b_ub_dec = np.array([-1 - 1e-6])

        A_ub_combined = np.vstack((A_ub, A_ub_dec))
        b_ub_combined = np.concatenate((b_ub, b_ub_dec))

        result_dec = linprog(
            c, A_ub=A_ub_combined, b_ub=b_ub_combined, bounds=bounds, method="highs"
        )

        if result_inc.success and abs(result_inc.fun - result.fun) < 1e-6:
            rts = "Increasing"
        elif result_dec.success and abs(result_dec.fun - result.fun) < 1e-6:
            rts = "Decreasing"
        else:
            rts = "Constant"

        return theta, ref_set, input_slacks, output_slacks, rts
    else:
        print(f"Warning: Optimization for DMU {dmu_idx} failed")
        return 1.0, [], np.zeros(n_inputs), np.zeros(n_outputs), "Unknown"


def solve_bcc_output_oriented_lp(
    dmu_idx, input_data, output_data, n_dmus, n_inputs, n_outputs
):
    """
    Solve BCC output-oriented DEA linear programming problem for a specific DMU.

    Args:
        dmu_idx (int): Index of the DMU being evaluated
        input_data (ndarray): Array of input values
        output_data (ndarray): Array of output values
        n_dmus (int): Number of DMUs
        n_inputs (int): Number of inputs
        n_outputs (int): Number of outputs

    Returns:
        efficiency (float): Efficiency score
        ref_set (list): Indices of reference DMUs
        input_slacks (ndarray): Input slack values
        output_slacks (ndarray): Output slack values
        rts (str): Returns to scale
    """
    # The objective is to maximize the efficiency score phi
    # Decision variables: [phi, lambda_1, lambda_2, ..., lambda_n]
    # For linprog (minimization), use -phi
    c = np.zeros(1 + n_dmus)
    c[0] = -1  # Maximize phi (minimize -phi)

    # Constraints for inputs: sum(lambda_j * x_ij) - x_i0 <= 0
    A_ub_inputs = np.zeros((n_inputs, 1 + n_dmus))
    b_ub_inputs = np.zeros(n_inputs)

    for i in range(n_inputs):
        A_ub_inputs[i, 0] = 0
        A_ub_inputs[i, 1:] = input_data[:, i]
        b_ub_inputs[i] = input_data[dmu_idx, i]

    # Constraints for outputs: phi*y_r0 - sum(lambda_j * y_rj) <= 0
    A_ub_outputs = np.zeros((n_outputs, 1 + n_dmus))
    b_ub_outputs = np.zeros(n_outputs)

    for i in range(n_outputs):
        A_ub_outputs[i, 0] = output_data[dmu_idx, i]
        A_ub_outputs[i, 1:] = -output_data[:, i]

    # Combine constraints
    A_ub = np.vstack((A_ub_inputs, A_ub_outputs))
    b_ub = np.concatenate((b_ub_inputs, b_ub_outputs))

    # Convexity constraint: sum(lambda_j) = 1
    A_eq = np.zeros((1, 1 + n_dmus))
    A_eq[0, 1:] = 1
    b_eq = np.array([1])

    # Phi must be >= 1, lambda variables must be non-negative
    bounds = [(1, None)] + [(0, None)] * n_dmus

    # Solve the linear programming problem
    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )

    if result.success:
        phi = result.x[0]
        lambdas = result.x[1:]

        # Calculate slacks in a second phase
        # Decision variables are slack variables for inputs and outputs
        c_slacks = np.ones(n_inputs + n_outputs)

        # Constraints for inputs: sum(lambda_j * x_ij) - x_i0 + s_i^- = 0
        A_eq_inputs = np.zeros((n_inputs, n_inputs + n_outputs))
        b_eq_inputs = np.zeros(n_inputs)

        for i in range(n_inputs):
            A_eq_inputs[i, i] = 1
            b_eq_inputs[i] = np.sum(lambdas * input_data[:, i]) - input_data[dmu_idx, i]

        # Constraints for outputs: phi*y_r0 - sum(lambda_j * y_rj) + s_r^+ = 0
        A_eq_outputs = np.zeros((n_outputs, n_inputs + n_outputs))
        b_eq_outputs = np.zeros(n_outputs)

        for i in range(n_outputs):
            A_eq_outputs[i, n_inputs + i] = 1
            b_eq_outputs[i] = phi * output_data[dmu_idx, i] - np.sum(
                lambdas * output_data[:, i]
            )
