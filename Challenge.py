#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 17:40:28 2017

@author: chidam
"""
from numpy import *

def get_error_rate(points, b, m):
    """
    This method returns the error rate using (y- (mx + b) **2)
    """
    totError = 0
    for i in range(0, len(points)):
        X = points[i, 1]/float(1000)
        y = points[i, 0]/float(1000)
        totError += (y - ((m * X) + b)) ** 2
    return totError/ float(len(points))

def step_gradient(points, b, m, learning_rate):
    """ 
    This method calculates the gradient steps
    """
    b_current = b
    m_current = m
    N = float(len(points))
    for i in range(0, len(points)):
        X = points[i, 1]/float(1000)
        y = points[i, 0]/float(1000)
        #print("X= " , X, " & y = " , y)
        b_gradient = - ((2/N) * (y - ((m_current * X)+ b_current)))
        m_gradient = - ((2/N) * X * (y - ((m_current * X) + b_current)))
    b_new = b_current - (learning_rate * b_gradient)
    m_new = m_current - (learning_rate * m_gradient)
    return [b_new, m_new]
    

def gradient_descent_runner(points, b, m, learning_rate, num_iteration):
    """
    This method runs the gradient descent process for the specified # of iterations
    """
    b_run = b
    m_run = m
    for i in range(0, num_iteration):
        [b_run, m_run] = step_gradient(points, b_run, m_run, learning_rate)
    return [b_run, m_run]

def run():
    """ 
    This method reads the data and runs the gradient descent algorithm to predict
    housing price based on squarefeet
    """ 
    points = genfromtxt("kc_house_data.csv",delimiter=",", usecols = (2,5), skip_header=1,max_rows=10)
    b_init = 0
    m_init = 0
    learning_rate = 0.01
    num_iteration = 2000
    print("Error before learning" , get_error_rate(points, b_init, m_init)) 
    [b_optim, m_optim] = gradient_descent_runner(points, b_init, m_init, learning_rate, num_iteration)
    print("Error after learning", get_error_rate(points, b_optim, m_optim), " & b = " , b_optim, " & m = " , m_optim)
    
if __name__ == '__main__':
    run()