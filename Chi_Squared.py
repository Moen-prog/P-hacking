import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import pandas as pd
from scipy.stats import expon


def chi_square_test(observed, expected):
    """A function that calculates the Chi square value given the observed and expected values
    
    Args:
        observed: A list of observed values
        expected: A list of expected values
        
    Returns:
        chi_square: A float representing the Chi square value"""
    chi_square = 0
    for o, e in zip(observed, expected):
        if e != 0:
            chi_square += (o - e) ** 2 / e
        else:
            chi_square += o ** 2
    return chi_square


def populate_points():
    """A function that generates a data structure containing the points from
    the work that Kristian did
    
    Args:
        None
    
    Returns:
        points: A list of points"""
    
    observed_frequencies = [185, 663, 29, 13, 7, 388, 0, 7, 0, 0, 161, 0, 0, 0, 0, 
                            6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 6, 6, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 16, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 7, 0, 0, 6]
    
    # making an assumption that that the observed values fall right in the middle of their range of frequency
    # thus we have 185 occurances of 0.005, 663 occurances of 0.015, 29 occurances of 0.025, etc.
    
    mid_box_value = 0.005
    points = []
    for frequency in observed_frequencies:
        print("Frequency: ", frequency, "Mid Box Value: ", mid_box_value)

        for i in range(frequency):
            points.append(mid_box_value)

        mid_box_value += 0.01

    print("Generated Points: ", points)

    return points



def main():
    #main funciton

    ### Part 1: Find the theoretical / expected values based on beta distribution. 
    # Define the parameters for the Beta distribution
    alpha = 1  # Shape parameter alpha
    beta_param = 17  # Shape parameter beta

    # Define the range
    start = 0
    end = 1

    # Define the number of equal parts
    num_parts = 100

    # Generate the points
    # points = np.linspace(start, end, num_parts + 1)
    points = np.linspace(start, end, num_parts + 1)

    # Generate the Beta distribution within the scaled range
    scaled_points = points / end
    beta_values = beta.pdf(scaled_points, alpha, beta_param)

    # Scale the beta values to match the original range
    beta_values = beta_values / np.sum(beta_values) * num_parts / (end - start)

    # Plot the Beta distribution
    plt.plot(points, beta_values, label=f'Beta({alpha}, {beta_param})')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.title('Beta Distribution')
    plt.legend()
    plt.grid(True)
    plt.show()


    ### Part 2: Take the observed values and put them into generated "boxes" for the range [0, 1] with 100 equal parts
    # Define the range
    start = 0
    end = 1

    # Define the number of equal parts
    num_parts = 100

    # Generate the "boxes"
    Box_range = np.linspace(start, end, num_parts + 1)

    # Print the points
    for boxes in Box_range:
        print(boxes)

    observed_values = populate_points()
    expected_values = beta_values

    # Perform the Chi square test:
    chi_square = chi_square_test(observed_values, expected_values)

    print(f'Chi square value: {chi_square}')


if __name__ == "__main__":
    # Main function only called when this program is ran directly
    main()


## Exponential graph (With some help from ChatGPT)

# Get your data here
# observed_values_2 = np.array([populate_points()])

# # Fit the exponential distribution to your observed data
# loc, fitted_scale = expon.fit(observed_values_2)

# # Plot the original data and fitted distribution
# plt.figure(figsize=(8,6))
# plt.hist(observed_values_2, bins=50, density=True, alpha=0.1, label='Observed Data')

# # Plot the fitted exponential distribution (PDF)
# x = np.linspace(0, np.max(observed_values_2), 1000)
# pdf_fitted = expon.pdf(x, loc=loc, scale=fitted_scale)
# plt.plot(x, pdf_fitted, 'r-', lw=2, label='Fitted Exponential Distribution')

# plt.xlabel('Value')
# plt.ylabel('Density')
# plt.title('Observed Data vs Fitted Exponential Distribution')
# plt.legend()

# plt.show()