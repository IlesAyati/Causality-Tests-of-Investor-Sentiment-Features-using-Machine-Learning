#!/usr/bin/env python
# coding: utf-8

# # Functions and Methods 
# 
# Often our Python program involves multiple independent tasks. It is good pratice to organise different tasks in separate functions. 
# 
# We have seen several Python built-in functions like `type`, `print` and `range`. It's straightforward to define our own functions:

# In[2]:


def f(x):
    if x < 0:
        return 'negative'
    return 'nonnegative'


# The keyword `def` lets us define a function. 
# 
# The function name comes after `def`, in the case above is `f`. 
# 
# Inside the parentheses is the *parameter variable*, `(x)`. 
# 
# The last three lines form the *function body*. 
# 
# The `return` statement marks the end of the function, at which point control is transfered back to the point in the program where the function is called. The function returns the result of the computation, the *return value*. 
# 
# A function can return any kind of object, including other functions. 
# 
# It can have one or more `return` statements, or none. The function will return the *return* value that corresponds to the first *return statement* that is met during each execution. If the function does not contain a `return` value, Python returns the object `None` by default. 

# In[3]:


f(-1)


# In[4]:


f(1)


# A function can have multiple arguments, and a Python program can have multiple functions. 
# 
# What matters is that all functions are defined in the global code before they are called. 
# 
# This is the reason why usually a Python program contains (1) `import` statements (more on these later), (2) function definitions, and (3) arbitrary global code, in this order. 

# In[5]:


def square(x):
    return x * x

def hypot(a, b):
    return (square(a) + square(b)) ** 0.50


# In[6]:


hypot(1,1)


# We can also use optional function arguments by assigning a *default value* for that argument:

# In[29]:


def g(x, a=1, b=1):
    return a + b * x


# In[34]:


g(2)


# In[33]:


g(2, a=4, b=5)


# ### Docstrings
# 
# Python has a system for adding comments to functions.

# In[12]:


def square(x):
    """ 
    This function squares its argument
    """
    return x*x


# In[13]:


get_ipython().run_line_magic('pinfo', 'square')


# This system allows to document any function, by describing its purpose, parameter values and return objects.

# ## Variable scope 
# 
# The scope of a variable is the universe of statements in the program that can refer to that variable directly. 
# 
# The scope of a function's *parameter variables* and *local variables* (variables declared inside the function's body) is limited to *that same function*. 
# 
# The scope of a variable defined in global code (*global variable*) is limited to the `.py` file containing that variable. 
# 
# When a function defines a local or parameter variable with the same name as a global variable, the variable name in the function refers to the local or parameter variable, not to the global variable. 
# 
# A good principle to follow when designing Python programs (or other software) is to define each variable so that its scope is *as small as possible*.

# ## Passing arguments and returning values 
# 
# We have seen the use of parameter variables above. Other variables that *live* inside a function are called *local variables* 

# In[35]:


def s(a, b):
    total = a + b
    return total


# In[38]:


i = 1
j = 2

s(i,j)


# `a` and `b` are parameter variables, `total` is a local variable.
# 
# The scope of both `a`, `b` and `total` is the function `s`.
# 
# The only difference between parameter variables and local variables is that Python initializes the parameter variable with the corresponding argument provided by the calling statement (above, with `i` and `j`). 
# 
# One consequence of this is that if a parameter variable refers to a *mutable* object and we change that object's value inside the function, then this also changes the object's value in the calling code. 
# 
# **When we pass arguments to a function, the arguments and the function's parameter variables become aliases**
# 
# We should remember the implications of this and the fact that, in Python: 
# 
# - `int`, `float`, `bool` and `str`are all *immutable* data-types 
# 
# - `list` (that we have used to represent arrays of data) are *mutable*

# Note: to see this clearly, we can use the `id` built-in function, that returns the identifier of an object. Each object has a unique identifier. The identity of an object is the address of the object in memory.

# In[1]:


i = 1.0
id(i)


# In[2]:


j = i
id(j)


# At this point, `i` and `j` are *aliases*, they point to the same object in memory. 
# 
# But this object is of data-type `float`, which is *immutable*: 

# In[41]:


type(j)


# This implies that if we add 1 to `j`, we are not changing that object's value, but we are in fact creating a different object to which `j` becomes bound to.

# In[3]:


j += 1


# In[4]:


j


# In[46]:


id(i)


# In[47]:


id(j)


# Let's see the implication of this to passing arguments to functions. 
# 
# Suppose we need a function that increments an integer by 1. 

# In[5]:


def inc(j):
    j += 1


# In[6]:


i = 99

id(i)


# In[52]:


inc(i) # function call with `i` as an argument


# In[11]:


i


# In[56]:


id(i)


# As expected, this code does not increment the value of the global variable `i`. 
# 
# The first statement, `i = 99`, assigns to global variable `i` a reference to the integer 99. 
# 
# Then the statement `inc(i)` passes `i`, an object reference, to the `inc` function. 
# 
# That object reference is assigned to the *parameter variable* `j`. 
# 
# At this point, `i` and `j` are aliases. 
# 
# As above, `j += 1` does not increment the value 99, but instead creates a new object of data-type `int` and value 100 and assigns a reference to that object to `j`. 
# 
# But since the scope of `j` is the function `inc`, when the function returns to its caller, its parameter variable goes out of scope, and the variable `i` still points to the same `int` object with value 99. 
# 
# To objtain the desired effect, we could use:

# In[8]:


def inc(j):
    j += 1
    return j


# In[12]:


inc(i)


# In[14]:


inc(j)


# As we have seen, arrays are *mutable* objects. Therefore, by passing a reference to an object array, we can change the values of that object since a function. 

# In[15]:


def exchange(a, i, j):
    temp = a[i]
    a[i] = a[j]
    a[j] = temp


# In[16]:


x = [1, 2, 3]


# In[20]:


exchange(x, 2, 1)


# In[21]:


x


# ## Modules 
# 
# For large projects that involve many functions, we can organize them in different files instead of having all the function `def` on the same `.py` file. 
# 
# This is done with *modules*. 
# 
# It has many advantages, such as the possibility to use the same function modules across different projects. 
# 
# We can write our own *modules*, or we can use any of the modules available. 
# 
# We only need to `import` them before we call a function that is defined in that *module*

# In[73]:


import math


# In[74]:


get_ipython().run_line_magic('pinfo', 'math')


# In[76]:


math.sqrt(2)


# Above, `math` is a module (which contains useful math function) and `sqrt` is the square root function. We call it by placing a dot in front of the module name, followed by the function name. 
# 
# Alternatively, we can import it and given it a different name. 

# In[77]:


import math as mt


# In[78]:


mt.sqrt(2)


# And we can import only a specific function from that module:

# In[79]:


from math import sqrt

sqrt(2)


# ## Libraries
# 
# Modules can be organized in Libraries. 
# 
# [Numpy](https://en.wikipedia.org/wiki/NumPy) is a widely used library for scientific programming. 
# 
# It is useful due to its fast array processing and many mathematical functions available to operate on those arrays. 

# In[80]:


from numpy import random # `random` is a module with functions that generate r.v. from the `numpy` library


# In[83]:


get_ipython().run_line_magic('pinfo', 'random')


# In[89]:


import numpy as np

x = np.random.uniform(0, 1, size=100) # produces an array with 100 U(0,1) random draws
x.mean()


# We have just used a *method*, `mean`. It implementes something that we could also do using a *function*, except that a *method* is explicitly associated with an *object* (hence the object, `x`, followed by `.` and the method name `mean`).

# ## Methods
# 
# A *method* is a function associated with a specified object (hence, with the type of that object). 
# 
# Methods act on the data contained in the object they belong to. 

# In[90]:


x = ['a', 'b']
x.append('c')


# In[91]:


x


# In[94]:


s = 'this is a string'
s.upper()


# In[95]:


s.replace('this','that')


# Many operations we have been using so far are actually organized as methods.

# In[97]:


x = ['a', 'b']
x[0] = 'aa'
x


# It doens't look like we are using any methods here, but the `[i]` assignment notation is just an interface to a method call: 

# In[98]:


x = ['a', 'b']
x.__setitem__(0, 'aa') # Equivalent to x[0] = 'aa'
x


# Python automatically maps built-in operators and functions to *methods*, using the convention that those methods  have double underscores before and after their names.

# In[110]:


s = 'a' 
t = 'b'
s + t


# In[111]:


s.__add__(t)


# In[114]:


len(s)


# In[115]:


s.__len__()


# In Python, everything in memory is treated as an object. 
# 
# This includes integers, floats, lists, strings, functions, modules, ...

# In[116]:


def f(x): return x**0.50
f


# The function `f` is just another object in memory.

# In[120]:


id(f)


# As an object, it also has methods, for example `__call__` method, that evaluates the function

# In[121]:


f.__call__(2)


# *Distinction between methods and functions*. They key difference is that a *method* is associated with a specified object. 
# 
# - a function call typically uses a *module* name (like `np.random.uniform`, where `random` is a module and `uniform` a function)
# 
# - a method call uses a object name (like `x.mean()`, where `x` is the random variable array and `mean` is the method).
# 
# Check the [Python documentation](https://docs.python.org/3/library/stdtypes.html) to see more modules associated with each data type.

# ### Exercise 1
# 
# Consider the polynomial
# 
# 
# <a id='equation-polynom0'></a>
# $$
# p(x)
# = a_0 + a_1 x + a_2 x^2 + \cdots a_n x^n
# = \sum_{i=0}^n a_i x^i \tag{1}
# $$
# 
# Write a function `p` such that `p(x, coeff)` computes the value in [(1)](#equation-polynom0) given a point `x` and a list of coefficients `coeff`.
# 
# Try to use `enumerate()` in your loop.

# This is probably not the best way to do this, but my approach is creating 
# one vector for betas, in this case = [1,2,..., n ], and one matrix x with ones in 
# column one and the polynomes on column 2. Then I matrix multiply the arrays. 

import numpy as np
n = 5
i = 1
mat = np.empty((n,1,))
mat[:] = np.nan
betas = mat[:]
x = np.ones((n,2,))
x[:,1] = 2                # Enter point here ex. x = 0.3
for i in range(len(mat)):
    betas[i] = i + 1      # Here I use integers as betas = 1:n
    x[i][1] = x[i,1]**(i)
betas = np.transpose(betas)
def p(x,betas):
    p = np.matmul(betas,x)
    return betas[0][0] + p[0][1]   # Returns p(x) =  a_0 + a_1 x + a_2 x^2 + \cdots a_n x^n = \sum_{i=0}^n a_i x^i 
p(x,betas)       # In this case n = 5


# ### Exercise 2
# 
# Write a function that takes a string as an argument and returns the number of capital letters in the string
# 
# Hint: `'foo'.upper()` returns `'FOO'`

def capcount(string):
    return sum(map(str.isupper, string))
# This one directly retunrs the number of capital letters in the string
 

# ### Exercise 3
# 
    ## I WAS NOT ABLE TO TO THIS ONE.
# Let’s write our own function approximation routine as an exercise.
# 
# #### 3.1 
# Without using any imports, write a function `linapprox` that takes as arguments:
# 
# - A function `f` mapping some interval $ [a, b] $ into $ \mathbb R $  
# - two scalars `a` and `b` providing the limits of this interval  
# - An integer `n` determining the number of grid points  
# - A number `x` satisfying `a <= x <= b`  
# 
# 
# and returns the [piecewise linear interpolation](https://en.wikipedia.org/wiki/Linear_interpolation) of `f` at `x`, based on `n` evenly spaced grid points `a = point[0] < point[1] < ... < point[n-1] = b`
def linapprox(f(x),a,b,n):
    
# #### 3.2 
# Test your approximation routine using `f` as a linear function.
# 
# #### 3.3
# Test your approximation routine using an higher order polynomial given by the function created in [Exercise 1](#pyfunc-ex1).
