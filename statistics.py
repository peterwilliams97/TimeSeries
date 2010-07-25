"""
Created on 25/07/2010

@author: peter
"""

def getMean(sequence):
    """ Returns arithmetic mean of a list """
    n = len(sequence)
    return sum(sequence)/n if n > 0 else 0
