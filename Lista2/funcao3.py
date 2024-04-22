import math
import numpy as np
import matplotlib.pyplot as plt


def simpson_3_8_rule(a, b, func, _):
    """
    Computes the integral of a function using Simpson's 3/8 rule.

    Parameters:
        a (float): The lower bound of the interval.
        b (float): The upper bound of the interval.
        func (function): The function to be integrated.

    Returns:
        float: The integral of the function.
    """
    h = (b - a) / 3
    return (3 * h / 8) * (func(a) + 3 * func(a + h) + 3 * func(a + 2 * h) + func(b))

def simpson_rule(a, b, func, _):
    """
    Computes the integral of a function using Simpson's 1/3 rule.

    Parameters:
        a (float): The lower bound of the interval.
        b (float): The upper bound of the interval.
        func (function): The function to be integrated.

    Returns:
        float: The integral of the function.
    """
    h = (b - a) / 2
    return (h / 3) * (func(a) + 4 * func(a + h) + func(b))

def gauss_legendre_rule(a, b, func, npoints):
    """
    Computes the integral of a function using Gauss-Legendre quadrature.

    Parameters:
        a (float): The lower bound of the interval.
        b (float): The upper bound of the interval.
        func (function): The function to be integrated.

    Returns:
        float: The integral of the function.
    """
    # Gauss-Legendre quadrature weights and nodes for 1, 2, 3, and 4 points

    if npoints == 1:
        weights = [2]
        nodes = [0]

    elif npoints == 2:
        weights = [1, 1]
        nodes = [-1 / math.sqrt(3), 1 / math.sqrt(3)]

    elif npoints == 3:
        weights = [5/9, 8/9, 5/9]
        nodes = [-math.sqrt(3/5), 0, math.sqrt(3/5)]

    elif npoints == 4:
        weights = [(18-math.sqrt(30))/36, (18+math.sqrt(30))/36, (18+math.sqrt(30))/36, (18-math.sqrt(30))/36]
        nodes = [-math.sqrt(3/7 + 2/7*math.sqrt(6/5)), -math.sqrt(3/7 - 2/7*math.sqrt(6/5)), math.sqrt(3/7 - 2/7*math.sqrt(6/5)), math.sqrt(3/7 + 2/7*math.sqrt(6/5))]
    
    else:
        raise ValueError("Warning: The number of points must be between 1 and 4.")
    
    integral = 0
    for point, weight in zip(nodes, weights):
        x = (b - a) / 2 * point + (b + a) / 2
        integral += weight * func(x)

    return (b - a) / 2 * integral

def trapezoidal_rule(a, b, func, _):
    """
    Computes the integral of a function using the trapezoidal rule.

    Parameters:
        a (float): The lower bound of the interval.
        b (float): The upper bound of the interval.
        func (function): The function to be integrated.

    Returns:
        float: The integral of the function.
    """
    return (b - a)*(func(a) + func(b))/2

class Interval:
    """
    Represents an interval for numerical integration.

    Attributes:
        a (float): The lower bound of the interval.
        b (float): The upper bound of the interval.
        method (str): The method used for numerical integration (default is "Trapezio").
        n_refinements (int): The number of uniform refinements for subdividing the interval (default is 0).
        sub_intervals (list): A list of sub-intervals of the same type as self.
        numerical_integral (float): The integral computed numerically over the interval.
        exact_integral (float): The exact value of the integral over the interval.
        integration_error (float): The integration error.
    """

    def __init__(self, a, b, method, nref=0, ref_level = 0, npoints = 0):
        self.a = a
        self.b = b
        self.method = method
        self.n_refinements = nref
        self.refinement_level = ref_level
        self.sub_intervals = []
        self.numerical_integral = 0
        self.exact_integral = 0
        self.integration_error = 0
        self.npoints = npoints

        if self.n_refinements > 0:
            span = (b - a)
            left_subinterval = Interval(a, a+span/2, method, nref-1, ref_level+1, self.npoints)
            self.sub_intervals.append(left_subinterval)
            right_subinterval = Interval(a+span/2, b, method, nref-1, ref_level+1, self.npoints)
            self.sub_intervals.append(right_subinterval)
    
    def numerical_integrate(self, func, ref_level):
        """
        Numerically integrates the given function using the specified method for a given refinement level.

        Parameters:
            func (function): The function to be integrated.
            ref_level (int): The refinement level for subdividing the interval.
        Returns:
            float: The numerical integral of the function.
        """
        if ref_level < 0:
            raise Exception("Warning: refinement level cannot be negative.")
        
        elif self.refinement_level == 0 and ref_level > self.n_refinements:
            raise Exception("Warning: refinement level cannot be greater than the number of refinements.")
        
        self.numerical_integral = 0
        if self.refinement_level == ref_level:
            self.numerical_integral = self.method(self.a, self.b, func, self.npoints)

        else:
            interval : Interval
            for interval in self.sub_intervals:
                self.numerical_integral += interval.numerical_integrate(func, ref_level)

        return self.numerical_integral

    def exact_integrate(self, integral_func):
        """
        Calculates the exact integral of a given function over the interval [a, b].

        Parameters:
        - integral_func: The primitive function.

        Returns:
        - The exact value of the integral over the interval [a, b].
        """

        self.exact_integral = integral_func(self.b) - integral_func(self.a)
        interval : Interval
        for interval in self.sub_intervals:
            interval.exact_integrate(integral_func)

        return self.exact_integral
    
    def compute_error(self, ref_level):
        if ref_level < 0:
            raise Exception("Warning: refinement level cannot be negative.")
            
        elif self.refinement_level == 0 and ref_level > self.n_refinements:
            raise Exception("Warning: refinement level cannot be greater than the number of refinements.")
            
        if self.refinement_level == ref_level:
            self.integration_error = abs(self.numerical_integral - self.exact_integral) 
        else:
            # Calcula as integrais em cada sub-intervalo para conseguir calcular o erro em cada intervalo (função recursiva)
            for interval in self.sub_intervals:
               self.integration_error += interval.compute_error(ref_level)    

        return self.integration_error               
    
    def Print(self):
        print("Interval: [", self.a, ", ", self.b, "]")
        print("Method: ", self.method)
        print("Number of divisions: ", self.n_refinements)
        print("numerical integral: ", self.numerical_integral)
        print("Exact integral: ", self.exact_integral)
        print("Integration error: ", self.integration_error)
        print("Number of sub intervals: ", len(self.sub_intervals))
        print()                                                       

piecewise_function = lambda x: (
    0 if x <= 0 else
    5*x + x**2 if 0 < x <= 1/math.pi else
    (math.pi**-2 + 5/math.pi - (5 + math.pi**-2 + 10/(3*math.pi) + 25*math.pi)/(1 + 5*math.pi**2) +
    (25*math.pi**2*x + x**2 + 5*math.pi*x**2 + 5*math.pi**2*x**2 - (5*math.pi**2*x**3)/3)/(1 + 5*math.pi**2))
    if 1/math.pi < x <= 2/math.pi else
    (math.pi**-2 + 5/math.pi - (5 + math.pi**-2 + 10/(3*math.pi) + 25*math.pi)/(1 + 5*math.pi**2) +
    (20 + 4/math.pi**2 + 20/(3*math.pi) + 50*math.pi)/(1 + 5*math.pi**2) -
    (8/math.pi + 40*math.pi - (math.pi + 5*math.pi**3)*math.cos(4/math.pi) - 4*math.sin(4/math.pi) - 10*math.pi**2*(-5 + 2*math.sin(4/math.pi)))/(math.pi + 5*math.pi**3) +
    (4*x + 20*math.pi**2*x - (math.pi + 5*math.pi**3)*math.cos(2*x) - 2*math.pi*x*math.sin(4/math.pi) - 5*math.pi**3*x*(-5 + 2*math.sin(4/math.pi)))/(math.pi + 5*math.pi**3))
    if 2/math.pi < x <= 8/math.pi else
    (math.pi**-2 + 5/math.pi - (5 + math.pi**-2 + 10/(3*math.pi) + 25*math.pi)/(1 + 5*math.pi**2) +
    (20 + 4/math.pi**2 + 20/(3*math.pi) + 50*math.pi)/(1 + 5*math.pi**2) +
    (32/math.pi + 160*math.pi - (math.pi + 5*math.pi**3)*math.cos(16/math.pi) - 16*math.sin(4/math.pi) - 40*math.pi**2*(-5 + 2*math.sin(4/math.pi)))/(math.pi + 5*math.pi**3) -
    (8/math.pi + 40*math.pi - (math.pi + 5*math.pi**3)*math.cos(4/math.pi) - 4*math.sin(4/math.pi) - 10*math.pi**2*(-5 + 2*math.sin(4/math.pi)))/(math.pi + 5*math.pi**3))
)

func = lambda x: (
    5 + 2*x if 0 <= x <= 1/math.pi else
    (2*x + 10*math.pi*x - 5*math.pi**2*(-5 - 2*x + x**2))/(1 + 5*math.pi**2) if 1/math.pi < x <= 2/math.pi else
    (4 + 20*math.pi**2 + 25*math.pi**3)/(math.pi + 5*math.pi**3) - 2*math.sin(4/math.pi) + 2*math.sin(2*x) if 2/math.pi < x <= 8/math.pi else
    0)

func_intr = piecewise_function
# interval.Print()

n_refinements = 10

refinement_levels = range(n_refinements)
exact_integrals = []
numerical_integrals_trap = []
integration_errors_trap = []

for n in refinement_levels:
    interval = Interval(0, 8/math.pi, trapezoidal_rule, n)
    
    exact = interval.exact_integrate(func_intr)
    exact_integrals.append(exact)
    
    val = interval.numerical_integrate(func, n)
    numerical_integrals_trap.append(val)
    
    error = interval.compute_error(n)
    integration_errors_trap.append(error)
    


numerical_integrals_simp = []
integration_errors_simp = []

for n in refinement_levels:
    interval = Interval(0, 8/math.pi, simpson_rule, n)
    
    exact = interval.exact_integrate(func_intr)
    exact_integrals.append(exact)
    
    val = interval.numerical_integrate(func, n)
    numerical_integrals_simp.append(val)
    
    error = interval.compute_error(n)
    integration_errors_simp.append(error)
    

numerical_integrals_simp_3_8 = []
integration_errors_simp_3_8 = []

for n in refinement_levels:
    interval = Interval(0, 8/math.pi, simpson_3_8_rule, n)
    
    exact = interval.exact_integrate(func_intr)
    exact_integrals.append(exact)
    
    val = interval.numerical_integrate(func, n)
    numerical_integrals_simp_3_8.append(val)
    
    error = interval.compute_error(n)
    integration_errors_simp_3_8.append(error)
    

h = [8/math.pi / 2**n for n in refinement_levels]

print("Trapezio")
for error, step in zip(integration_errors_trap, h):
    print("(",step,",",error,")")

print("Simpson")
for error, step in zip(integration_errors_simp, h):
    print("(",step,",",error,")")

print("Simpson_3_8")
for error, step in zip(integration_errors_simp_3_8, h):
    print("(",step,",",error,")")


# convergence_rate_trap = (math.log(integration_errors_trap[-1]) - math.log(integration_errors_trap[-2])) / (math.log(h[-1]) - math.log(h[-2]))
# convergence_rate_simp = (math.log(integration_errors_simp[-1]) - math.log(integration_errors_simp[-2])) / (math.log(h[-1]) - math.log(h[-2]))
# convergence_rate_simp_3_8 = (math.log(integration_errors_simp_3_8[-1]) - math.log(integration_errors_simp_3_8[-2])) / (math.log(h[-1]) - math.log(h[-2]))

# print(numerical_integrals_trap)
# print(numerical_integrals_simp)
# print(numerical_integrals_simp_3_8)

# print(exact_integrals)

# print("Convergence rate (Trapezoidal Rule):", convergence_rate_trap)
# print("Convergence rate (Simpson's 1/3 Rule):", convergence_rate_simp)
# print("Convergence rate (Simpson's 3/8 Rule):", convergence_rate_simp_3_8)

# convergence_rates_trap = ['-']
# for i in range(1, len(integration_errors_trap)):
#     rate = (math.log(integration_errors_trap[i]) - math.log(integration_errors_trap[i-1])) / (math.log(h[i]) - math.log(h[i-1]))
#     convergence_rates_trap.append(rate)

# print("Trapézio")
# print("N° Ref.\t\th\t\tNumerical Integration\t\tIntegration Errors\t\tConvergence Rate")
# for refinement, step, numerical_integrals_trap, integration_errors_trap, convergence_rates_trap in zip(refinement_levels, h, numerical_integrals_trap, integration_errors_trap, convergence_rates_trap):
#     if type(convergence_rates_trap) == str:
#         print(f"{refinement} & {step:.5f} & {numerical_integrals_trap:.5f} & {integration_errors_trap:.2E} & {convergence_rates_trap}{chr(92)*2}")
#     else:
#         print(f"{refinement} & {step:.5f} & {numerical_integrals_trap:.5f} & {integration_errors_trap:.2E} & {convergence_rates_trap:.2f} {chr(92)*2}")


# convergence_rates_simp = ['-']
# for i in range(1, len(integration_errors_simp)):
#     rate = (math.log(integration_errors_simp[i]) - math.log(integration_errors_simp[i-1])) / (math.log(h[i]) - math.log(h[i-1]))
#     convergence_rates_simp.append(rate)

# print("Trapézio")
# print("N° Ref.\t\th\t\tNumerical Integration\t\tIntegration Errors\t\tConvergence Rate")
# for refinement, step, numerical_integrals_simp, integration_errors_simp, convergence_rates_simp in zip(refinement_levels, h, numerical_integrals_simp, integration_errors_simp, convergence_rates_simp):
#     if type(convergence_rates_simp) == str:
#         print(f"{refinement} & {step:.5f} & {numerical_integrals_simp:.5f} & {integration_errors_simp:.2E} & {convergence_rates_simp}{chr(92)*2}")
#     else:
#         print(f"{refinement} & {step:.5f} & {numerical_integrals_simp:.5f} & {integration_errors_simp:.2E} & {convergence_rates_simp:.2f} {chr(92)*2}")

convergence_rates_simp_3_8 = ['-']
for i in range(1, len(integration_errors_simp_3_8)):
    rate = (math.log(integration_errors_simp_3_8[i]) - math.log(integration_errors_simp_3_8[i-1])) / (math.log(h[i]) - math.log(h[i-1]))
    convergence_rates_simp_3_8.append(rate)

print("Trapézio")
print("N° Ref.\t\th\t\tNumerical Integration\t\tIntegration Errors\t\tConvergence Rate")
for refinement, step, numerical_integrals_simp_3_8, integration_errors_simp_3_8, convergence_rates_simp_3_8 in zip(refinement_levels, h, numerical_integrals_simp_3_8, integration_errors_simp_3_8, convergence_rates_simp_3_8):
    if type(convergence_rates_simp_3_8) == str:
        print(f"{refinement} & {step:.5f} & {numerical_integrals_simp_3_8:.5f} & {integration_errors_simp_3_8:.2E} & {convergence_rates_simp_3_8}{chr(92)*2}")
    else:
        print(f"{refinement} & {step:.5f} & {numerical_integrals_simp_3_8:.5f} & {integration_errors_simp_3_8:.2E} & {convergence_rates_simp_3_8:.2f} {chr(92)*2}")



