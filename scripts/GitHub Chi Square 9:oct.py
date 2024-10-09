import numpy as np
from scipy.stats import chisquare, beta, expon
import matplotlib.pyplot as plt
import pandas as pd


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


def generate_histogram(graph_name, data):
    """A function that generates a histogram of the data
    
    Args:
        graph_name: A string representing the name of the graph
        data: A list of data points
        
    Returns:
        None"""
    
    # Create a histogram
    plt.hist(data, bins=10, alpha=0.5, color='b', edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(graph_name)
    plt.show()


def main():
    #main funciton

    ### Part 1: Find the theoretical / expected values based on beta distribution. 
    # Define the parameters for the Beta distribution
    alpha = 1  # Shape parameter alpha
    beta_param = 40  # Shape parameter beta

    observed = populate_points()

    # Generate the expected frequencies using a Beta distribution
    np.random.seed(42)  # Seed the random number generator for reproducibility
    expected = beta.rvs(alpha, beta_param, size=len(observed))  # Random sample from beta distribution
    #expected = expected / np.sum(expected) * np.sum(observed)  # Scale expected to match the total count of observed
       #Took out the line above on 10/9, is it necessary?

    # Plotting the Beta distribution
    x = np.linspace(0, 1, 100)  # x values for the plot
    y = beta.pdf(x, alpha, beta_param)  # Beta PDF values
    expected = expected * (1 / 100) * 100  # Assuming total counts = 100

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=f'Beta Distribution (a={alpha}, b={beta_param})', color='blue')
    plt.title('Beta Distribution')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.legend()
    plt.grid()
    plt.show()

    # Perform the Chi square test:
    chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)

    # Calculate degrees of freedom
    n_bins = 100
    degrees_of_freedom = n_bins - 1 

    # Print the results
    print(f"Chi-square statistic: {chi2_stat}")
    print(f"P-value: {p_value}")
    print(f"Degrees of Freedom: {degrees_of_freedom}")

    #chi2_stat, p_value = chi_square_test(observed, expected)
    print(f"Chi-Square Statistic: {chi2_stat}, P-value: {p_value}")

    # Generate the histogram for the observed and expected values
    generate_histogram('Expected Values', expected)
    generate_histogram('Observed Values', observed)

    return chi2_stat, p_value, degrees_of_freedom


if __name__ == "__main__":
    # Main function only called when this program is ran directly
    main()