# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:40:27 2024

@author: stefa
"""
from sklearn.neighbors import BallTree
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.optimize import minimize
from scipy.spatial import cKDTree


def parabola(x, a, b, c):
    #if a<-0.2 or a>0.2:
    #    return np.repeat(1e10, len(x))
    x = np.array(x)
    return a * np.power(x,2) + b * x + c



def generate_random_parabola(num_parabolas, a_range=(-0.2, 0.2), b_range=(-3, 3), c_range=(-10, 10), rotation_range=(0, np.pi)):
    parabolas = []
    for _ in range(num_parabolas):
        a = np.random.uniform(*a_range)
        b = np.random.uniform(*b_range)
        c = np.random.uniform(*c_range)
        rotation = np.random.uniform(*rotation_range)
        parabolas.append((a, b, c, rotation))
    return parabolas



def add_noise(points, noise_scale=0.05):
    noise_x = np.random.normal(scale=noise_scale, size=points.shape[0])
    noise_y = np.random.normal(scale=noise_scale, size=points.shape[0])
    noisy_points = points + np.column_stack((noise_x, noise_y))
    return noisy_points

def plot_parabolas(parabolas, num_points=500):
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_facecolor('black')
    all_points = []
    x = np.linspace(-5, 5, num_points)
    for a, b, c, rotation in parabolas:
        y = a * x**2 + b * x + c
        x_rotated = x * np.cos(rotation) - y * np.sin(rotation)
        y_rotated = x * np.sin(rotation) + y * np.cos(rotation)
        points = np.column_stack((x_rotated, y_rotated))
        noisy_points = add_noise(points)
        all_points.append(noisy_points)  # Accumulate noisy points
        plt.scatter(noisy_points[:, 0], noisy_points[:, 1], s=1, color='white')
    
    all_points = np.vstack(all_points)
    filtered_array = all_points[
    (all_points[:, 0] >= -5) & (all_points[:, 0] <= 5) &
    (all_points[:, 1] >= -5) & (all_points[:, 1] <= 5)
    ]


    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.show()
    
    return filtered_array  # Convert list of arrays to a single array

def calculate_regression(points):
    """
    Calculate the regression line for a set of points and return the equation
    and the coefficient of determination (r-squared).
    
    Args:
    - points: numpy array of shape (N, 2) where N is the number of points
    
    Returns:
    - regression_eq: tuple containing (slope, intercept) of the regression line
    - r_squared: coefficient of determination (r-squared) of the regression
    """
    x = points[:, 0]
    y = points[:, 1]
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r_squared = r_value ** 2
    regression_eq = (slope, intercept)
    return regression_eq, r_squared


def plot_regression_lines(points, radius):
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_facecolor('black')

    all_segments = []
    for point, data in points.items():
        slope, intercept = data['regression_eq']
        x0, y0 = point
        x1 = x0 + radius / np.sqrt(1 + slope**2)
        y1 = slope * x1 + intercept
        x2 = x0 - radius / np.sqrt(1 + slope**2)
        y2 = slope * x2 + intercept
        all_segments.append([(x1, y1), (x2, y2)])
    
    for segment in all_segments:
        plt.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], color='red')

    #plt.xlim(-5, 5)
    #plt.ylim(-5, 5)
    #plt.show()

def fit_parabola(x_data, y_data):
    popt, _ = curve_fit(parabola, x_data, y_data)
    a_fit, b_fit, c_fit = popt

    # Calculate R^2
    y_pred = parabola(x_data, *popt)
    r_squared = r2_score(y_data, y_pred)

    return a_fit, b_fit, c_fit, r_squared

def generate_parabola_points(a, b, c, x_min, x_max, num_points):
    """Generate points on a parabola."""
    x_values = np.linspace(x_min, x_max, num_points)
    y_values = a * x_values**2 + b * x_values + c
    return np.column_stack((x_values, y_values))

def point_within_distance_of_parabola(point, parabola_points, distance):
    """Check if a given point is within a certain distance of a parabola."""
    kdtree = cKDTree(parabola_points)
    distance_to_nearest_point, _ = kdtree.query(point)
    return distance_to_nearest_point <= distance


def distance_to_parabola(x, y, a, b, c):
    # Define function to minimize: squared distance from point to parabola
    def distance_func(x0):
        return (x0 - x)**2 + (parabola(x0, a, b, c) - y)**2
    
    # Initial guess for x-coordinate of nearest point on the parabola
    x0_initial_guess = x
    
    # Minimize the distance function to find nearest point on the parabola
    result = minimize(distance_func, x0_initial_guess)
    x_nearest = result.x[0]
    y_nearest = parabola(x_nearest, a, b, c)
    
    # Calculate distance from point to nearest point on the parabola
    distance = np.sqrt((x - x_nearest)**2 + (y - y_nearest)**2)
    return distance



def points_within_radius(points, radius):
    """
    Generate a dictionary where each point is a key, and the corresponding value
    is a list of all points within the specified radius, along with the regression
    line equation and r-squared value. Additionally, add a line segment centered
    at each point with length twice the radius and slope as the regression line.
    
    Args:
    - points: numpy array of shape (N, 2) where N is the number of points
    - radius: the radius within which points are considered neighbors
    
    Returns:
    - point_dict: dictionary containing points and their neighbors within the radius
      along with the regression line equation, r-squared value, and line segment
    """
    tree = BallTree(points)  # Build Ball tree from points
    point_dict = {}
    processed_points = set()  # Set to track processed points
    fit_parabolas= []
    for point in points:
        if tuple(point) in processed_points:
            continue  # Skip processed points
        neighbors_indices = tree.query_radius([point], r=radius)[0]
        neighbors = points[neighbors_indices]
        x = [point[0] for point in neighbors]
        y = [point[1] for point in neighbors]
        
        a,b,c, r_squared = fit_parabola(x,y)
        if r_squared>0.8:
            fit_parabolas.append({'a':a,'b':b,'c':c,'r_squared':r_squared})
            data = []
            parabola_points = generate_parabola_points(a, b, c, -6, 6, 2000)
            
            for point in points:
                if tuple(point) in processed_points:
                    continue  # Skip processed points

                within_distance = point_within_distance_of_parabola(point, parabola_points, radius/2)

                if within_distance:
                    data.append(point)
                    processed_points.add(tuple(point))   


        point_dict[tuple(point)] = {
            'point':point,
             'neighbors': neighbors.tolist(),
             'r_squared': r_squared,
         }
        # Skip adding the subject point to processed_points
        #processed_points.update(map(tuple, neighbors))  # Mark neighbors as processed
    return fit_parabolas


def solve_quadratic(a, b, c):
    """Solve the quadratic equation ax^2 + bx + c = 0."""
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return []  # No real roots
    elif discriminant == 0:
        return [-b / (2*a)]  # One real root
    else:
        sqrt_discriminant = discriminant**0.5
        return [(-b + sqrt_discriminant) / (2*a), (-b - sqrt_discriminant) / (2*a)]  # Two real roots

def solve_rotated_quadratic(a, b, c, theta):
    """Solve the quadratic equation ax^2 + bx + c = 0 for a rotated parabola."""
    a_rot = a * np.cos(theta)**2
    b_rot = 2 * a * np.cos(theta) * np.sin(theta) - b
    c_rot = a * np.sin(theta)**2 + b * np.sin(theta) + c

    return solve_quadratic(a_rot, b_rot, c_rot)

def find_intersection_points(parabolas):
    intersection_points = set()

    for i in range(len(parabolas)):
        for j in range(i+1, len(parabolas)):
            a1, b1, c1, theta1 = parabolas[i]
            a2, b2, c2, theta2 = parabolas[j]
            # Calculate the rotated coefficients for both parabolas
            a_rot1 = a1 * np.cos(theta1)**2
            b_rot1 = 2 * a1 * np.cos(theta1) * np.sin(theta1) - b1
            c_rot1 = a1 * np.sin(theta1)**2 + b1 * np.sin(theta1) + c1

            a_rot2 = a2 * np.cos(theta2)**2
            b_rot2 = 2 * a2 * np.cos(theta2) * np.sin(theta2) - b2
            c_rot2 = a2 * np.sin(theta2)**2 + b2 * np.sin(theta2) + c2

            # Solve the quadratic equations simultaneously for rotated parabolas
            intersection_x_values = solve_quadratic(a_rot1 - a_rot2, b_rot1 - b_rot2, c_rot1 - c_rot2)

            # Compute y values for intersection x values
            for x in intersection_x_values:
                y1 = a1 * x**2 + b1 * x + c1
                y2 = a2 * x**2 + b2 * x + c2
                intersection_points.add((x, max(y1, y2)))

    return intersection_points



if __name__ == "__main__":
    for num_parabolas in [20]:
        print("Testing " + str(num_parabolas) + " parabolas:")
        diffs = []
        for _ in range(5):
            parabolas = generate_random_parabola(num_parabolas)
            points = plot_parabolas(parabolas)
            
        
            
            np.random.shuffle(points)
            
            # if num_parabolas == 3:
            #     radius = 2
            # elif num_parabolas == 5:
            #         radius = 1
            # elif num_parabolas == 10:
            #     radius = 1
            # elif num_parabolas == 20:
            #     radius = 0.5
    
            r = points_within_radius(points,1)
            inferred_parabolas = [tuple([x['a'], x['b'], x['c'], 0]) for x in r]
            ints_inferred = find_intersection_points(inferred_parabolas)
            ints_real = find_intersection_points(parabolas)
            print("Difference number intersections")
            print(abs(len(ints_real)-len(ints_inferred)))
            
            print("Difference number parabolas")
            print(abs(len(r)-num_parabolas))
            diff = abs(len(r)-num_parabolas)
            diffs.append(diff)
            x = np.linspace(-6, 6, 100)
        
            # Plot settings
            plt.figure(figsize=(6, 6))
            ax = plt.gca()
            ax.set_facecolor('black')
            plt.xlim(-5, 5)
            plt.ylim(-5, 5)
        
            # Plot each parabola
            for parabol in r:
                a, b, c = parabol['a'], parabol['b'], parabol['c']
                y = a * x**2 + b * x + c
                plt.plot(x, y, color='white')
        
            # Show the plot
            plt.show()
        print("Average number of differences: " + str(sum(diffs)/len(diffs)))
    
        