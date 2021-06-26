import math as m
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import pylab
import pandas as pd
import scipy.linalg as la
import traceback
import inspect

'''
Algorithms are based off of 
A friendly Introduction to Numerical Analysis by Brian Bradie 
'''





'''
        Ctrl+Shift+[ Fold (collapse) region
        Ctrl+Shift+] Unfold (uncollapse) region
        Ctrl+K Ctrl+[ Fold (collapse) all subregions
        Ctrl+K Ctrl+] Unfold (uncollapse) all subregions
# Ctrl+K Ctrl+0 Fold (collapse) all regions
# Ctrl+K Ctrl+J Unfold (uncollapse) all regions
'''





def diagonal_form(a, lower=1, upper=1):
    """
    a is a numpy square matrix
    this function converts a square matrix to diagonal ordered form
    returned matrix in ab shape which can be used directly for scipy.linalg.solve_banded
    """
    n = a.shape[1]
    assert(np.all(a.shape == (n, n)))
    ab = np.zeros((2*n-1, n))
    for i in range(n):
        ab[i, (n-1)-i:] = np.diagonal(a, (n-1)-i)
    for i in range(n-1):
        ab[(2*n-2)-i, :i+1] = np.diagonal(a, i-(n-1))
    mid_row_inx = int(ab.shape[0]/2)
    upper_rows = [mid_row_inx - i for i in range(1, upper+1)]
    upper_rows.reverse()
    upper_rows.append(mid_row_inx)
    lower_rows = [mid_row_inx + i for i in range(1, lower+1)]
    keep_rows = upper_rows+lower_rows
    ab = ab[keep_rows, :]
    return ab


def find_i(x_data, x):
    if x > x_data[-1] or x < x_data[0]:
        print('x value is not in the interval')
    index = np.searchsorted(x_data, x, side='right')
    if index == len(x_data):
        index = index - 1
    return index - 1


def LagrangeCoefficient(x_data, j):
    '''
    data is a 1 by n array consisting of x_i for all i from 0 to n
    j marks the index associated with the Lagrange poly
    x_temp removes the j index from consideration
    numerator creates an expression for the numerator
    as the product (handeled by the reduce function) 
    of all differences(handeld by the map function)
    return is a math expression for the Lagrange polynomial for any x.
    '''
    x_temp = np.delete(x_data, [j])
    def numerator(x): return reduce(lambda a, b: a*b,
                                    list(map(lambda d: x - d, x_temp)))
    # The denominator is simply the numerator evaluated at x_j
    def lagrange(x): return numerator(x) / numerator(x_data[j])
    s = ""
    for x in x_temp:
        s += "(x - " + str(x)+")"
    print("\\frac{", s, "}{", str(
        numerator(x_data[j])), "}", "f(", str(x_data[j]), ")", "+")
    return lagrange


def Nevilles(x_bar, data):
    '''
    x_bar is the point we wish to evaluate the polynomial at
    data is a 2 by n array of points consisting of x_i for all i from 0 to n and the associated f_i
    latex_tabbing and latex_newline are functions for formating the output
    return None, prints a table to the console
    '''
    def latex_tabbing(x): return "{:,.3f}{}".format(
        x, '&') if isinstance(x, float) else "{}{}".format(x, '&')

    def latex_newline(x): return "{:,.3f}{}".format(
        x, '\\\\') if isinstance(x, float) else "{}{}".format(x, '\\\\')
    dim_r, dim_c = np.size(data, axis=1), np.size(
        data, axis=1)  # matrix dimension (dim_r x dim_c)
    M = np.empty((dim_r, dim_c))
    M[:] = np.nan
    M[:, 0] = data[1, :]
    i = 1
    while i < dim_r:
        j = 1
        while j < dim_c:
            M[i, j] = ((x_bar - data[0, i-j])*M[i, j-1] - (x_bar -
                                                           data[0, i])*M[i-1, j-1])/(data[0, i] - data[0, i-j])
            j += 1
        i += 1
    # Formating table
    df = pd.DataFrame.from_records(M).replace(np.nan, '', regex=True)
    df.update(df.iloc[:, :-1].applymap(latex_tabbing))
    df.update(df.iloc[:, -1].map(latex_newline))
    print(df.to_string(index=False, header=False))


def Newton_form(data):
    '''
    data is a 2 by n array of points consisting of x_i for all i from 0 to n and the associated f_i.
    latex_tabbing and latex_newline are functions for formating the output.
    return Coefficients, prints divided difference table to the console.
    '''
    def latex_tabbing(x): return "{:,.3f}{}".format(
        x, '&') if isinstance(x, float) else "{}{}".format(x, '&')

    def latex_newline(x): return "{:,.3f}{}".format(
        x, '\\\\') if isinstance(x, float) else "{}{}".format(x, '\\\\')
    dim_r, dim_c = np.size(data, axis=1), np.size(
        data, axis=1)  # matrix dimension (dim_r x dim_c)
    M = np.empty((dim_r, dim_c))
    M[:] = np.nan
    M[:, 0] = data[1, :]
    i = 1
    while i < dim_r:
        j = 1
        while j < dim_c:
            M[i, j] = (M[i, j-1] - M[i-1, j-1])/(data[0, i] - data[0, i-j])
            j += 1
        i += 1
    # Formating table
    df = pd.DataFrame.from_records(M).replace(np.nan, '', regex=True)
    df.update(df.iloc[:, :-1].applymap(latex_tabbing))
    df.update(df.iloc[:, -1].map(latex_newline))
    print(df.to_string(index=False, header=False))
    return np.diag(M)


def bisection(func, x_values, y_values, slopes, v_shift=0, Interval=[0, 1], epsilon=10**(-6)):
    def f(x): return func(x_values, y_values, slopes, x) + \
        v_shift  # func is made by piecewise linear interpolation
    start = Interval[0]
    end = Interval[1]
    n = 0
    while(n < m.ceil(np.log2((end - start)/epsilon))):
        p_1 = (start+end)/2
        if f(start)*f(p_1) < 0:
            end = p_1
        else:
            start = p_1
        n += 1
    return p_1


def LatexTable(df, index_bool=False, header_bool=False):
    def latex_tabbing(x): return "{:,.4e}{}".format(
        x, '&') if isinstance(x, float) else "{}{}".format(x, '&')
    def latex_newline(x): return "{:,.4e}{}".format(
        x, '\\\\') if isinstance(x, float) else "{}{}".format(x, '\\\\')
    df_temp = df.copy()
    df_temp.update(df.iloc[:, :-1].applymap(latex_tabbing))
    df_temp.update(df.iloc[:, -1].map(latex_newline))
    print(df_temp.to_string(index=index_bool, header=header_bool))


def load_data(filePath):
    return pd.read_csv(filePath, sep=',')


def regression(xy_data=[[], []]):
    def linear_regression(a, b, x): return a + b*x
    # under edit  this has not been verified yet
    n = np.size(xy_data, axis=1)
    ones = np.ones(n)
    x_array = np.array(xy_data[0])
    y_array = np.array(xy_data[1])
    b = (n * np.dot(x_array, y_array) - np.dot(x_array, ones) * np.dot(y_array,
                                                                       ones)) / (n*np.dot(x_array, x_array) - np.dot(x_array, ones)**2)
    a = (np.dot(y_array, ones) - b * (np.dot(x_array, ones))) / n
    return a, b, linear_regression


def Piecwise_Linear(xy_data=[]):
    def eval_line(x_data, y_data, slope_data, x): return y_data[find_i(
        x_data, x)] + slope_data[find_i(x_data, x)] * (x - x_data[find_i(x_data, x)])
    # eval_line is used like the function f(x_i,a_i,m,x) = S_i(x) = a_i + m*(x-x_i)
    slopes = []
    y_0 = xy_data[1][0]
    x_0 = xy_data[0][0]
    i = 1
    while i < np.size(xy_data, axis=1):
        y_1 = xy_data[1][i]
        x_1 = xy_data[0][i]
        m = (y_1 - y_0) / (x_1 - x_0)
        slopes.append(m)
        x_0, y_0 = x_1, y_1
        i += 1
    return xy_data[0], xy_data[1],  slopes, eval_line


def hermite_poly(xyyprime_data, func=None, f_prime=None):
    '''
    xyyprime_data is a 3 by n array of points consisting of x_i for all i from 0 to n,
    the associated f_i, and measures of the derivative at x_i.
    latex_tabbing and latex_newline are functions for formating the output.
    return Coefficients, prints divided difference table to the console.
    '''
    def calulate_Hermite_poly(coefficient, z_data, x):
        k = 0
        i = 0
        sol = 0
        prod = 1
        while k < len(coefficient):
            if i == k-1:
                prod *= x - z_data[i]
                i += 1
            sol += coefficient[k]*prod
            k += 1
        return sol
    xyyprime_data = np.atleast_2d(xyyprime_data)
    dim_c = np.size(xyyprime_data, axis=0)
    if dim_c == 3:
        xy_data = [xyyprime_data[0], xyyprime_data[1]]
        prime_data = xyyprime_data[2]
    elif dim_c == 2:
        xy_data = [xyyprime_data[0], xyyprime_data[1]]
        prime_data = [f_prime(x) for x in xyyprime_data[0]]
    elif dim_c == 1:
        xy_data = [xyyprime_data[0], [func(x) for x in xyyprime_data[0]]]
        prime_data = [f_prime(x) for x in xyyprime_data[0]]
    else:
        print('error not enough data to proceed')
        return

    def latex_tabbing(x): return "{:,.3f}{}".format(
        x, '&') if isinstance(x, float) else "{}{}".format(x, '&')

    def latex_newline(x): return "{:,.3f}{}".format(
        x, '\\\\') if isinstance(x, float) else "{}{}".format(x, '\\\\')
    dim_r, dim_c = np.size(xy_data, axis=1), np.size(
        xy_data, axis=1)  # matrix dimension (dim_r x dim_c)
    M = np.empty((2*dim_r, 2*dim_c))
    M[:] = np.nan
    M[:, 0] = [xy_data[1][int(j / 2)] for j in range(2*dim_r)]
    M[1:, 1] = [prime_data[int(j / 2)] for j in range(2*dim_r)][1:]
    z_data = [xy_data[0][int(j / 2)] for j in range(2*dim_r)]
    i = 1
    while i < 2*dim_r:
        j = 1
        while j < 2*dim_c:
            if (z_data[i] - z_data[i-j]) == 0:
                j += 1
                continue
            M[i, j] = float(M[i, j-1] - M[i-1, j-1])/(z_data[i] - z_data[i-j])
            j += 1
        i += 1
    # Formating table
    df = pd.DataFrame.from_records(M).replace(np.nan, '', regex=True)
    df.update(df.iloc[:, :-1].applymap(latex_tabbing))
    df.update(df.iloc[:, -1].map(latex_newline))
    print(df.to_string(index=False, header=False))

    return np.diag(M), z_data, calulate_Hermite_poly


def cubic_spline_clamped():
    print('under construction')


def cubic_spline_NotaKnot(xy_data=[[], []]):
    n = np.size(xy_data, axis=1) - 1  # n+1 data points has n intervals
    h = np.zeros(n, np.float64)  # h values for the length of each interval
    # c is the coefficient matrix n unique c values for each spline
    c = np.zeros([n-1, n-1])
    gamma = np.zeros([n-1], np.float64)     # gamma is the solution vector
    # calculating the length of the inetervals
    i = 1
    while i <= n:
        h[i-1] = xy_data[0][i] - xy_data[0][i-1]
        i += 1
    i = 1
    while i < n:
        gamma[i-1] = 3*(xy_data[1][i+1] - xy_data[1][i])/h[i] - \
            3*(xy_data[1][i] - xy_data[1][i-1])/h[i-1]
        i += 1
    # calculating the middle block of the tridiagonal system (exclude first and last row)
    i = 1
    while i < n - 2:
        c[i, i-1] = h[i-1]
        c[i, i] = 2*(h[i-1] + h[i])
        c[i, i+1] = h[i]
        i += 1
    # applying boundary conditions and setting the first and last row
    c[0, 0] = 3 * h[0] + 2 * h[1] + h[0]**2/float(h[1])
    c[0, 1] = h[1] - (h[0]**2)/h[1]
    c[-1, -1] = 3*h[n-1] + 2*h[n-2] + (h[n-1])**2/h[n-2]
    c[-1, -2] = h[n-2] - h[n-1]**2 / h[n-2]
    # calculating our solution vector
    # c_sol = np.linalg.solve(c,gamma) # this function should have more risidual then solve_banded
    c_diag_orderd = diagonal_form(c, lower=1, upper=1)
    c_sol = la.solve_banded((1, 1), c_diag_orderd, gamma)

    c_0 = (1 + float(h[0])/h[1])*c_sol[0] - c_sol[1] * h[0]/h[1]
    c_sol = np.insert(c_sol, 0, c_0)  # adding c_0 solution... equation 8
    c_n = (-float(h[-1])/h[-2])*c_sol[-2] + \
        (1 + float(h[-1])/h[-2]) * c_sol[-1]
    c_sol = np.append(c_sol, c_n)  # adding c_n solution... equation 9

    i = 0
    b_sol = []
    d_sol = []
    while i < n:
        b = float(xy_data[1][i+1] - xy_data[1][i])/h[i] - \
            float(h[i])*(2*c_sol[i] + c_sol[i+1])/3
        d = (c_sol[i+1] - c_sol[i])/(3*h[i])
        b_sol.append(b)
        d_sol.append(d)
        i += 1
    return xy_data[1], b_sol, c_sol, d_sol, xy_data[0]


def Newton_cotes(a, b, n=0, open=False):
    # n is the subscript of nodes in the interval (a,b) for open [a,b] for closed

    if open:
        print("error open is malfunctioning.... check both open and close")
        n = n + 2  # this adjusts the formula to refrence the book values
        begin, end = 0, n
        b_sol = [(pow(n-1, i+1) - 1) / (i+1) for i in range(begin, end)]
        Vander_matrix = np.vander(
            [i for i in range(begin, end)], increasing=True)

    else:
        print("error open is malfunctioning.... check both open and close")
        begin, end = 0, n + 1
        b_sol = [pow(n, i+1)/(i + 1) for i in range(begin, end)]
        Vander_matrix = np.vander(
            [i for i in range(begin, end)], increasing=True)
    print(Vander_matrix)
    wieghts = np.matmul(np.linalg.inv(Vander_matrix.T), b_sol)/n
    return wieghts


def composite_trapazoidal(a, b, function, n_sub_interval=100, tol=pow(10, -4), max_iteration=500):
    n = n_sub_interval
    h = (b-a)/n
    area = (h/2) * (function(a) + function(b))
    area += (h) * sum([function(a + i*h) for i in range(1, n)])
    return area


def composite_simpsons(a, b, function, n_sub_interval=100, tol=pow(10, -4), max_iteration=500):
    n = n_sub_interval
    h = (b-a)/n
    if n % 2 == 0:
        m = n // 2
    else:
        print("sub_interval is not even returning...")
        return
    area = function(a) + function(b)
    area += 4*sum([function(a + h*(2*i - 1)) for i in range(1, m + 1)]) # sum 1 to m
    area += 2*sum([function(a + h*(2*i)) for i in range(1, m )]) # sum 1 to m - 1
    area *= h/3
    return area


def composite_gausian(a, b, function, n_sub_interval=100, tol=pow(10, -4), max_iteration=500):
    n = n_sub_interval
    h = (b-a)/n
    area = (h/2) * sum([function((a + i*h) - (h/2)*(1 + np.sqrt(1/3)))
                        + function((a + i*h) - (h/2)*(1 - np.sqrt(1/3))) for i in range(1, n + 1)])
    return area


def romberg_integration(a, b, function, tol=pow(10, -4), max_iteration=500, real_sol=None):
    n = max_iteration
    romberg_matrix = np.empty([n, n])
    romberg_matrix[:][:] = np.nan
    h = (b - a)
    romberg_matrix[0][0] = (0.5)*h*(function(a) + function(b))
    approx_error = np.empty(n)
    approx_error[:] = np.nan
    i = 1
    while i < n:
        j = 1
        h = h/2
        romberg_matrix[i][0] = romberg_matrix[i-1][0]/2 + h * \
            sum([function(a + h*(2*i - 1)) for i in range(1, pow(2, i-1) + 1)])
        while j <= i:
            romberg_matrix[i][j] = (
                pow(4, j) * romberg_matrix[i][j-1] - romberg_matrix[i-1][j-1])/(pow(4, j) - 1)
            # Alt version romberg_matrix[i][j-1]/2 + (romberg_matrix[i][j-1] - romberg_matrix[i-1][j-1])/(pow(4,j) - 1)
            j += 1
        error = abs(
            (romberg_matrix[i][i] - romberg_matrix[i-1][i-1])/pow(2, i))
        approx_error[i] = error
        if error < tol:
            approx_error = approx_error[0:i+1]
            romberg_matrix = romberg_matrix[0:i+1, 0:i+1]
            break
        i += 1
    df = pd.DataFrame(romberg_matrix)
    df['error bound'] = approx_error
    df = df.replace(np.nan, '', regex=True)

    LatexTable(df)
    if real_sol is not None:
        df_error = df.applymap(lambda x: abs(
            real_sol - x) if not isinstance(x, str) else x)
        df_error["Error bound"] = approx_error
        df_error.fillna('', inplace=True)
        LatexTable(df_error, header_bool=True)
    else:
        df_error = pd.DataFrame(approx_error, columns=["Error bound"])
        df_error.fillna('', inplace=True)
        LatexTable(df_error)


def trapazoidal(a, b, f):
    return (b-a)*(f(a) + f(b))/2


def simpsons(a, b, f):
    c = (a + b)/2
    return (b-a)*(f(a) + 4 * f(c) + f(b))/6


def gausian(a, b, f):
    return (b - a)*(f((a+b)/2 - (b - a)/(2 * np.sqrt(3))) + f((a+b)/2 - (b-a)/(2 * np.sqrt(3)))) / 2


def adaptive_quadrature(function, a, b, tol, s_type, data=[]):
    # hardcoded for simpsons to reduce function calls...
    def I_approx(a, b, function, s_type):
        if s_type == "simpsons":
            sol = simpsons(a, b, function)
        elif s_type == "gausian":
            sol = gausian(a, b, function)
        elif s_type == "trapazoidal":
            sol = trapazoidal(a, b, function)
        return sol

    def adapt1(sab, fa, fc, fb, function, a, b, tol):
        c = (a+b)/2
        fd = function((a+c)/2)
        fe = function((c+b)/2)
        sac = (c-a)*(fa + 4 * fd + fc)/6
        scb = (b-c)*(fc + 4 * fe + fb)/6
        #sac = I_approx( a, c, function , s_type)
        #scb = I_approx( c, b, function, s_type )
        if abs(sac + scb - sab) < 10*tol:
            data.append(["\\int_" + str(a) + "^" + str(b) + "f(x) dx",
                         sab, sac, scb, sac + scb, abs(sac + scb - sab)/10, True])
            return sac + scb
        else:
            data.append(["\\int_" + str(a) + "^" + str(b) + "f(x) dx",
                         sab, sac, scb, sac + scb, abs(sac + scb - sab)/10, False])
            return adapt1(sac, fa, fd, fc, function, a, c, tol/2) + adapt1(scb, fc, fe, fb, function, c, b, tol/2)
    fa = function(a)
    fb = function(b)
    fc = function((a+b)/2)
    sab = I_approx(a, b, function, s_type)
    return data, adapt1(sab, fa, fc, fb, function, a, b, tol)


def euler_method(function, a, b, n_steps, t0, y0):
    data = []
    t = t0
    y = y0
    h = (b-a)/n_steps
    data.append([h, 0, t, y])
    for j in range(1, n_steps+1):
        k1 = function(t, y)
        y += k1*h
        t += h
        data.append([h, j, t, y])
   
    df = pd.DataFrame(data, columns=["h", "steps", "t", "sol"])
    print(df)
    return df

def taylor_method(function, a, b, n_steps, t0, y0, order=4, true_sol=None, ft=None, ftt=None, fttt=None):
    data = []
    k1, k2, k3, k4 = 0, 0, 0, 0
    t = t0
    y = y0
    h = (b-a)/n_steps
    data.append([h, 0, t, y])
    for j in range(1, n_steps+1):
        if order >= 1: 
            k1 = function(t, y)
        if order >= 2 and ft is not None:
            k2 = ft(t, y)
        if order >= 3 and ftt is not None:
            k3 = ftt(t, y)
        if order == 4 and fttt is not None:
            k4 = fttt(t, y)
        y += k1*h + k2*h*h/2 + k3*pow(h,3)/6 + k4*pow(h, 4)/24 
        t += h
        error = abs(true_sol(t) - y)
        data.append([h, j, t, y, error])  
    df = pd.DataFrame(data, columns=["h", "steps", "t", "sol Order_"+str(order), "abs error Order_"+str(order)])
    print(df)
    return df


def example_method():
    a, b = 1, 6
    t0, y0 = 1, 1
    H = [0.125, 0.25, 0.5]
    order = [1, 2, 3]
    def true_sol(t): return t*(1 + np.log(t))
    def f(t, x): return 1 + x/t
    def ft(t,x): return 1/t
    def ftt(t,x): return -1/(t*t)
    def fttt(t,x): return 2/pow(t,3)
    ax = None
    for i in range(len(H)):
        n_steps = int((b - a)/H[i])
        df = taylor_method(f, a, b, n_steps, t0, y0, order=order[i], true_sol=true_sol, ft=ft, ftt=ftt, fttt=fttt)
        ax = df.plot(x="t", y="sol Order_"+str(order[i]), ax=ax)
        #ax = df.plot(x="t", y="abs error Order_"+str(order[i]),logy=True, ax=ax)
    plt.show()


def runge_kutta(function, a, b, h, t0, y0, x=None):
    # classical_weights = np.array([1/6, 1/3, 1/3, 1/6 ])
    data = [["t_i", "w_i", "x(t_i)", " |x(t_i) - w_i |" ]]
    xtrue = None
    if x is not None:
        data.append([t0, y0, x(t0), None ])
    else:
        data.append([t0, y0, None, None ])
    error = None
    for i in range(int((b-a)/ h)):
        k1 = h*function(t0, y0)
        k2 = h*function(t0 + h/2 , y0 + k1/2)
        k3 = h*function(t0 + h/2 , y0 + k2/2)
        k4 = h*function(t0 + h , y0 + k3)
        y0 = y0 + (k1 + 2*k2 + 2*k3 + k4)/6
        #y0 = y0 + k1/6 + k2/3 + k3/3 + k4/6
        if x is not None:
            error = abs(x(t0 + h) - y0)
            xtrue = x(t0 + h)
        t0 += h    
        data.append([t0, y0, xtrue, error])
        
    
    df = pd.DataFrame(data)
    #print(df.to_string(header=False))
    return df

def optimal_runge_kutta_2(function, a, b, h, t0, y0, x=None):
    data = [["t_i", "w_i", "x(t_i)", " |x(t_i) - w_i |" ]]
    if x is not None:
        data.append([t0, y0, x(t0), None ])
    else:
        data.append([t0, y0, None, None ])
    error = None
    for i in range(int((b-a)/ h)):
        w_approx = y0 + 2*h/3*f(t0,y0)
        y0 = y0 + (h/4) * f(t0, y0) + (3*h/4) *f(t0 + 2*h/3, w_approx)
        if x is not None:
            error = abs(x(t0) - y0)
        t0 += h
        data.append([t0, y0, x, error])

    
    df = pd.DataFrame(data)
    #print(df.to_string(header=False))
    return df

def finite_diffence_method(p, q, r, alpha=[1,1,1,1], beta=[1,1,1,1], h=2,  N=6, boundary_cond=["dirchlet","dirchlet"]):
    A = np.zeros([N+1, N+1])
    B = np.zeros(N+1)
    d = lambda i: 2 + h**2 * q(i)
    u = lambda i: -1 + p(i) * h/2
    l = lambda i: -1 - p(i) * h/2
    # setting boundary condition
    if boundary_cond[0].lower() == "dirchlet":
        # A[0, 1] = 0        # dirchlet BC at x = a (initialized at 0)
        A[0, 0] = 1          # dirchlet BC at x = a
        B[0] = alpha[0]

    if boundary_cond[1] == "dirchlet":
        A[-1, -1] = 1        # dirchlet BC at x = b 
        # A[-1, -2] = 0      # dirchlet BC at x = b  (initialized at 0)
        B[-1] = beta[0]  
    else:
        A[-1, -2] = - 2
        A[0, 1] = -2
    if boundary_cond[0].lower() == "neumann":   
        A[0, 0] = d(0)                        # neuman BC at x = a
        B[0] = -h^2 * r(0) + 2*h*l(0)*alpha[0]   # neuman BC at x = a
        
    if boundary_cond[1].lower() == "nuemann":
        A[-1, -1] = d(N)                      # neuman BC at x = b
        B[-1] = -h^2 * r(N) - 2*h*u(N)*beta[0]   # neuman BC at x = b

    if boundary_cond[0].lower() == "robin":
        A[0, 0] =   d(0) + 2*h*l(0)*alpha[1]/alpha[2]
        B[0]  = -h^2 * r(0) + 2*h*l(0)*alpha[3]/alpha[2]
        
    if boundary_cond[1].lower() == "robin":
        A[-1, -1] = d(N) - 2*h*u(N)*beta[1]*beta[2]
        B[-1] = -h^2 * r(N) - 2*h*u(N)*beta[3]/beta[2]

    for i in range(1, N):
        B[i] = -h**2 * r(i)
        A[i, i-1] = l(i)
        A[i, i] = d(i)
        A[i, i + 1] = u(i)
    sol = np.linalg.solve(A ,B)
    return A, B, sol





function = lambda t,y: y*(4 - t*y )
a=0 
b = 3
n_steps = 60
y0 = 1.3
t0 = 0
df = euler_method(function, a, b, n_steps, t0, y0)
df1 = df.loc[20*np.array( [0.5, 1, 1.5, 2, 2.5, 3] ) ] 
print(df1)