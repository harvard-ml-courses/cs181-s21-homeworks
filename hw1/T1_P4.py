#####################
# CS 181, Spring 2021
# Homework 1, Problem 4
# Start Code
##################

import csv
import numpy as np
import matplotlib.pyplot as plt
import math

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

# Plot the data.
plt.figure(1)
plt.plot(years, republican_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figure(2)
plt.plot(years, sunspot_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Sunspots")
plt.figure(3)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()

# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(years.shape), years)).T

# TODO: basis functions
# Based on the letter input for part ('a','b','c','d'), output numpy arrays for the bases.
# The shape of arrays you return should be: (a) 24x6, (b) 24x12, (c) 24x6, (c) 24x26
# xx is the input of years (or any variable you want to turn into the appropriate basis).
# is_years is a Boolean variable which indicates whether or not the input variable is
# years; if so, is_years should be True, and if the input varible is sunspots, is_years
# should be false
def make_basis(xx,part='a',is_years=True):
    X = np.ones(xx.shape) # initialize matrix we're building with ones for bias term
    if part == 'a':
        xx = (xx - np.array([1960]*len(xx)))/40 if is_years else xx/20
        for i in range(1, 6):
            X = np.vstack((X, np.power(xx, i)))
        return X.T
    if part == 'b':
        if not is_years:
            return None
        for y in range(1960, 2010, 5):
            X = np.vstack((X, np.exp(-((xx - y) ** 2 / 25))))
        return X.T
    stop = 6 if part == 'c' else 26
    for i in range(1, stop):
        X = np.vstack((X, np.cos(xx / i)))
    return X.T

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X,Y):
    w = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    return w

# Plot the data and the regression lines for each basis on a grid of inputs
for is_years in [True, False]:
    if is_years:
        xlabel = "Year"
        orig_X = years
        orig_grid_X = np.linspace(1960, 2005, 200)
    else:
        xlabel = "Number of Sunspots"
        orig_X = sunspot_counts[years<last_year]
        orig_grid_X = np.linspace(10, 155, 200)
        Y = republican_counts[years<last_year]
    print(xlabel)
    for part in ['a', 'b', 'c', 'd']:
        if not is_years and part == 'b':
            continue
        X = make_basis(orig_X, part, is_years) # get basis transformations for actual data
        Yhat = np.dot(X, find_weights(X, Y)) # get predictions and calculate loss
        print('L2:', sum((Y - Yhat) ** 2))

        grid_X = make_basis(orig_grid_X, part, is_years) # get basis transformations for grid data
        grid_Yhat = np.dot(grid_X, find_weights(X, Y)) # get predictions and plot

        plt.plot(orig_X, Y, 'o', orig_grid_X, grid_Yhat, '-')
        plt.xlabel(xlabel)
        plt.ylabel("Republicans in Congress")
        plt.show()
