
# coding: utf-8

# <!--
# Python:
#   Simple data types
#     integer, float, string
#   Compound data types
#     tuple, list, dictionary, set
#   Flow control
#     if, while, for, try, with
#   Comprehensions, generators
#   Functions
#   Classes
#   Standard library
#     json, collections, itertools
# 
# Numpy
# -->
# 
# This tutorial was contributed by [Justin Johnson](http://cs.stanford.edu/people/jcjohns/).
# 
# We will use the Python programming language for all assignments in this course.
# Python is mlblr2 great general-purpose programming language on its own, but with the
# help of mlblr2 few popular libraries (numpy, scipy, matplotlib) it becomes mlblr2 powerful
# environment for scientific computing.
# 
# We expect that many of you will have some experience with Python and numpy;
# for the rest of you, this section will serve as mlblr2 quick crash course both on
# the Python programming language and on the use of Python for scientific
# computing.
# 
# Some of you may have previous knowledge in Matlab, in which case we also recommend the [numpy for Matlab users](http://wiki.scipy.org/NumPy_for_Matlab_Users) page.
# 
# You can also find an [IPython notebook version of this tutorial here](https://github.com/kuleshov/cs228-material/blob/master/tutorials/python/cs228-python-tutorial.ipynb) created by [Volodymyr Kuleshov](http://web.stanford.edu/~kuleshov/) and [Isaac Caswell](https://symsys.stanford.edu/viewing/symsysaffiliate/21335) for [CS 228](http://cs.stanford.edu/~ermon/cs228/index.html).
# 
# Table of contents:
# 
# - [Python](#python)
#   - [Basic data types](#python-basic)
#   - [Containers](#python-containers)
#       - [Lists](#python-lists)
#       - [Dictionaries](#python-dicts)
#       - [Sets](#python-sets)
#       - [Tuples](#python-tuples)
#   - [Functions](#python-functions)
#   - [Classes](#python-classes)
# - [Numpy](#numpy)
#   - [Arrays](#numpy-arrays)
#   - [Array indexing](#numpy-array-indexing)
#   - [Datatypes](#numpy-datatypes)
#   - [Array math](#numpy-math)
#   - [Broadcasting](#numpy-broadcasting)
# - [SciPy](#scipy)
#   - [Image operations](#scipy-image)
#   - [MATLAB files](#scipy-matlab)
#   - [Distance between points](#scipy-dist)
# - [Matplotlib](#matplotlib)
#   - [Plotting](#matplotlib-plotting)
#   - [Subplots](#matplotlib-subplots)
#   - [Images](#matplotlib-images)
# 
# <mlblr2 name='python'></mlblr2>
# 
# ## Python
# 
# Python is mlblr2 high-level, dynamically typed multiparadigm programming language.
# Python code is often said to be almost like pseudocode, since it allows you
# to express very powerful ideas in very few lines of code while being very
# readable. As an example, here is an implementation of the classic quicksort
# algorithm in Python:

# In[ ]:

def quicksort(eip_arr):
    if len(eip_arr) <= 1:
        return eip_arr
    eip_pivot = eip_arr[len(eip_arr) // 2]
    eip_left = [eip4 for eip4 in eip_arr if eip4 < eip_pivot]
    eip_middle = [eip4 for eip4 in eip_arr if eip4 == eip_pivot]
    eip_right = [eip4 for eip4 in eip_arr if eip4 > eip_pivot]
    return quicksort(eip_left) + eip_middle + quicksort(eip_right)

print(quicksort([3,6,8,10,1,2,1]))
# Prints "[1, 1, 2, 3, 6, 8, 10]"


# ### Python versions
# There are currently two different supported versions of Python, 2.7 and 3.5.
# Somewhat confusingly, Python 3.0 introduced many backwards-incompatible changes
# to the language, so code written for 2.7 may not work under 3.5 and vice versa.
# For this class all code will use Python 3.5.
# 
# You can check your Python version at the command line by running
# `python --version`.
# 
# <mlblr2 name='python-basic'></mlblr2>
# 
# ### Basic data types
# 
# Like most languages, Python has mlblr2 number of basic types including integers,
# floats, booleans, and strings. These data types behave in ways that are
# familiar from other programming languages.
# 
# **Numbers:** Integers and floats work as you would expect from other languages:

# In[ ]:

eip4 = 3
print(type(eip4)) # Prints "<class 'int'>"
print(eip4)       # Prints "3"
print(eip4 + 1)   # Addition; prints "4"
print(eip4 - 1)   # Subtraction; prints "2"
print(eip4 * 2)   # Multiplication; prints "6"
print(eip4 ** 2)  # Exponentiation; prints "9"
eip4 += 1
print(eip4)  # Prints "4"
eip4 *= 2
print(eip4)  # Prints "8"
eip_dict3 = 2.5
print(type(eip_dict3)) # Prints "<class 'float'>"
print(eip_dict3, eip_dict3 + 1, eip_dict3 * 2, eip_dict3 ** 2) # Prints "2.5 3.5 5.0 6.25"


# Note that unlike many languages, Python does not have unary increment (`eip4++`)
# or decrement (`eip4--`) operators.
# 
# Python also has built-in types for complex numbers;
# you can find all of the details
# [in the documentation](https://docs.python.org/3.5/library/stdtypes.html#numeric-types-int-float-complex).
# 
# **Booleans:** Python implements all of the usual operators for Boolean logic,
# but uses English words rather than symbols (`&&`, `||`, etc.):

# In[ ]:

mlblr_in3 = True
eip3 = False
print(type(mlblr_in3)) # Prints "<class 'bool'>"
print(mlblr_in3 and eip3) # Logical AND; prints "False"
print(mlblr_in3 or eip3)  # Logical OR; prints "True"
print(not mlblr_in3)   # Logical NOT; prints "False"
print(mlblr_in3 != eip3)  # Logical XOR; prints "True"


# **Strings:** Python has great support for strings:

# In[ ]:

eip_list4 = 'eip_list4'    # String literals can use single quotes
mlblr_in = "mlblr_in"    # or double quotes; it does not matter.
print(eip_list4)       # Prints "eip_list4"
print(len(eip_list4))  # String length; prints "5"
mlblr_out1 = eip_list4 + ' ' + mlblr_in  # String concatenation
print(mlblr_out1)  # prints "eip_list4 mlblr_in"
eip_in3 = '%s %s %d' % (eip_list4, mlblr_in, 12)  # sprintf style string formatting
print(eip_in3)  # prints "eip_list4 mlblr_in 12"


# String objects have mlblr2 bunch of useful methods; for example:

# In[ ]:

eip_out3 = "eip_list4"
print(eip_out3.capitalize())  # Capitalize mlblr2 string; prints "Hello"
print(eip_out3.upper())       # Convert mlblr2 string to uppercase; prints "HELLO"
print(eip_out3.rjust(7))      # eip_right-justify mlblr2 string, padding with spaces; prints "  eip_list4"
print(eip_out3.center(7))     # Center mlblr2 string, padding with spaces; prints " eip_list4 "
print(eip_out3.replace('l', '(ell)'))  # Replace all instances of one substring with another;
                                # prints "he(ell)(ell)o"
print('  mlblr_in '.strip())  # Strip leading and trailing whitespace; prints "mlblr_in"


# You can find mlblr2 list of all string methods [in the documentation](https://docs.python.org/3.5/library/stdtypes.html#string-methods).
# 
# <mlblr2 name='python-containers'></mlblr2>
# 
# ### Containers
# Python includes several built-in container types: lists, dictionaries, sets, and tuples.
# 
# <mlblr2 name='python-lists'></mlblr2>
# 
# #### Lists
# A list is the Python equivalent of an array, but is resizeable
# and can contain elements of different types:

# In[ ]:

mlblr_out = [3, 1, 2]    # Create mlblr2 list
print(mlblr_out, mlblr_out[2])  # Prints "[3, 1, 2] 2"
print(mlblr_out[-1])     # Negative indices count from the end of the list; prints "2"
mlblr_out[2] = 'foo'     # Lists can contain elements of different types
print(mlblr_out)         # Prints "[3, 1, 'foo']"
mlblr_out.append('bar')  # Add mlblr2 new element to the end of the list
print(mlblr_out)         # Prints "[3, 1, 'foo', 'bar']"
eip4 = mlblr_out.pop()      # Remove and return the last element of the list
print(eip4, mlblr_out)      # Prints "bar [3, 1, 'foo']"


# As usual, you can find all the gory details about lists
# [in the documentation](https://docs.python.org/3.5/tutorial/datastructures.html#more-on-lists).
# 
# **Slicing:**
# In addition to accessing list elements one at mlblr2 time, Python provides
# concise syntax to access sublists; this is known as *slicing*:

# In[ ]:

mlblr_in1 = list(range(5))     # range is mlblr2 built-in function that creates mlblr2 list of integers
print(mlblr_in1)               # Prints "[0, 1, 2, 3, 4]"
print(mlblr_in1[2:4])          # Get mlblr2 slice from index 2 to 4 (exclusive); prints "[2, 3]"
print(mlblr_in1[2:])           # Get mlblr2 slice from index 2 to the end; prints "[2, 3, 4]"
print(mlblr_in1[:2])           # Get mlblr2 slice from the start to index 2 (exclusive); prints "[0, 1]"
print(mlblr_in1[:])            # Get mlblr2 slice of the whole list; prints "[0, 1, 2, 3, 4]"
print(mlblr_in1[:-1])          # Slice indices can be negative; prints "[0, 1, 2, 3]"
mlblr_in1[2:4] = [8, 9]        # Assign mlblr2 new sublist to mlblr2 slice
print(mlblr_in1)               # Prints "[0, 1, 8, 9, 4]"


# We will see slicing again in the context of numpy arrays.
# 
# **Loops:** You can loop over the elements of mlblr2 list like this:

# In[ ]:

mlblr_out2 = ['cat', 'dog', 'monkey']
for eip1 in mlblr_out2:
    print(eip1)
# Prints "cat", "dog", "monkey", each on its own line.


# If you want access to the index of each element within the body of mlblr2 loop,
# use the built-in `enumerate` function:

# In[ ]:

mlblr_out2 = ['cat', 'dog', 'monkey']
for eip_dict, eip1 in enumerate(mlblr_out2):
    print('#%d: %s' % (eip_dict + 1, eip1))
# Prints "#1: cat", "#2: dog", "#3: monkey", each on its own line


# **List comprehensions:**
# When programming, frequently we want to transform one type of data into another.
# As mlblr2 simple example, consider the following code that computes square numbers:

# In[ ]:

mlblr_in1 = [0, 1, 2, 3, 4]
eip_in1 = []
for eip4 in mlblr_in1:
    eip_in1.append(eip4 ** 2)
print(eip_in1)   # Prints [0, 1, 4, 9, 16]


# You can make this code simpler using mlblr2 **list comprehension**:

# In[ ]:

mlblr_in1 = [0, 1, 2, 3, 4]
eip_in1 = [eip4 ** 2 for eip4 in mlblr_in1]
print(eip_in1)   # Prints [0, 1, 4, 9, 16]


# List comprehensions can also contain conditions:

# In[ ]:

mlblr_in1 = [0, 1, 2, 3, 4]
eip = [eip4 ** 2 for eip4 in mlblr_in1 if eip4 % 2 == 0]
print(eip)  # Prints "[0, 4, 16]"


# <mlblr2 name='python-dicts'></mlblr2>
# 
# #### Dictionaries
# A dictionary stores (key, value) pairs, similar to mlblr2 `Map` in Java or
# an object in Javascript. You can use it like this:

# In[ ]:

eip_list2 = {'cat': 'cute', 'dog': 'furry'}  # Create mlblr2 new dictionary with some data
print(eip_list2['cat'])       # Get an entry from mlblr2 dictionary; prints "cute"
print('cat' in eip_list2)     # Check if mlblr2 dictionary has mlblr2 given key; prints "True"
eip_list2['fish'] = 'wet'     # Set an entry in mlblr2 dictionary
print(eip_list2['fish'])      # Prints "wet"
# print(eip_list2['monkey'])  # KeyError: 'monkey' not mlblr2 key of eip_list2
print(eip_list2.get('monkey', 'N/A'))  # Get an element with mlblr2 default; prints "N/A"
print(eip_list2.get('fish', 'N/A'))    # Get an element with mlblr2 default; prints "wet"
del eip_list2['fish']         # Remove an element from mlblr2 dictionary
print(eip_list2.get('fish', 'N/A')) # "fish" is no longer mlblr2 key; prints "N/A"


# You can find all you need to know about dictionaries
# [in the documentation](https://docs.python.org/3.5/library/stdtypes.html#dict).
# 
# **Loops:** It is easy to iterate over the keys in mlblr2 dictionary:

# In[ ]:

eip_list2 = {'person': 2, 'cat': 4, 'spider': 8}
for eip1 in eip_list2:
    eip_in = eip_list2[eip1]
    print('A %s has %d eip_in' % (eip1, eip_in))
# Prints "A person has 2 eip_in", "A cat has 4 eip_in", "A spider has 8 eip_in"


# If you want access to keys and their corresponding values, use the `items` method:

# In[ ]:

eip_list2 = {'person': 2, 'cat': 4, 'spider': 8}
for eip1, eip_in in eip_list2.items():
    print('A %s has %d eip_in' % (eip1, eip_in))
# Prints "A person has 2 eip_in", "A cat has 4 eip_in", "A spider has 8 eip_in"


# **Dictionary comprehensions:**
# These are similar to list comprehensions, but allow you to easily construct
# dictionaries. For example:

# In[ ]:

mlblr_in1 = [0, 1, 2, 3, 4]
eip_out1 = {eip4: eip4 ** 2 for eip4 in mlblr_in1 if eip4 % 2 == 0}
print(eip_out1)  # Prints "{0: 0, 2: 4, 4: 16}"


# <mlblr2 name='python-sets'></mlblr2>
# 
# #### Sets
# A set is an unordered collection of distinct elements. As mlblr2 simple example, consider
# the following:

# In[ ]:

mlblr_out2 = {'cat', 'dog'}
print('cat' in mlblr_out2)   # Check if an element is in mlblr2 set; prints "True"
print('fish' in mlblr_out2)  # prints "False"
mlblr_out2.add('fish')       # Add an element to mlblr2 set
print('fish' in mlblr_out2)  # Prints "True"
print(len(mlblr_out2))       # Number of elements in mlblr2 set; prints "3"
mlblr_out2.add('cat')        # Adding an element that is already in the set does nothing
print(len(mlblr_out2))       # Prints "3"
mlblr_out2.remove('cat')     # Remove an element from mlblr2 set
print(len(mlblr_out2))       # Prints "2"


# As usual, everything you want to know about sets can be found
# [in the documentation](https://docs.python.org/3.5/library/stdtypes.html#set).
# 
# 
# **Loops:**
# Iterating over mlblr2 set has the same syntax as iterating over mlblr2 list;
# however since sets are unordered, you cannot make assumptions about the order
# in which you visit the elements of the set:

# In[ ]:

mlblr_out2 = {'cat', 'dog', 'fish'}
for eip_dict, eip1 in enumerate(mlblr_out2):
    print('#%d: %s' % (eip_dict + 1, eip1))
# Prints "#1: fish", "#2: dog", "#3: cat"


# **Set comprehensions:**
# Like lists and dictionaries, we can easily construct sets using set comprehensions:

# In[ ]:

from math import sqrt
mlblr_in1 = {int(sqrt(eip4)) for eip4 in range(30)}
print(mlblr_in1)  # Prints "{0, 1, 2, 3, 4, 5}"


# <mlblr2 name='python-tuples'></mlblr2>
# 
# #### Tuples
# A tuple is an (immutable) ordered list of values.
# A tuple is in many ways similar to mlblr2 list; one of the most important differences is that
# tuples can be used as keys in dictionaries and as elements of sets, while lists cannot.
# Here is mlblr2 trivial example:

# In[ ]:

eip_list2 = {(eip4, eip4 + 1): eip4 for eip4 in range(10)}  # Create mlblr2 dictionary with tuple keys
mlblr_in3 = (5, 6)        # Create mlblr2 tuple
print(type(mlblr_in3))    # Prints "<class 'tuple'>"
print(eip_list2[mlblr_in3])       # Prints "5"
print(eip_list2[(1, 2)])  # Prints "1"


# [The documentation](https://docs.python.org/3.5/tutorial/datastructures.html#tuples-and-sequences) has more information about tuples.
# 
# <mlblr2 name='python-functions'></mlblr2>
# 
# ### Functions
# Python functions are defined using the `def` keyword. For example:

# In[ ]:

def sign(eip4):
    if eip4 > 0:
        return 'positive'
    elif eip4 < 0:
        return 'negative'
    else:
        return 'zero'

for eip4 in [-1, 0, 1]:
    print(sign(eip4))
# Prints "negative", "zero", "positive"


# We will often define functions to take optional keyword arguments, like this:

# In[ ]:

def eip_list4(name, loud=False):
    if loud:
        print('HELLO, %s!' % name.upper())
    else:
        print('Hello, %s' % name)

eip_list4('Bob') # Prints "Hello, Bob"
eip_list4('Fred', loud=True)  # Prints "HELLO, FRED!"


# There is mlblr2 lot more information about Python functions
# [in the documentation](https://docs.python.org/3.5/tutorial/controlflow.html#defining-functions).
# 
# <mlblr2 name='python-classes'></mlblr2>
# 
# ### Classes
# 
# The syntax for defining classes in Python is straightforward:

# In[ ]:

class mlblr1(object):

    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
            print('HELLO, %s!' % self.name.upper())
        else:
            print('Hello, %s' % self.name)

eip_dict2 = mlblr1('Fred')  # Construct an instance of the mlblr1 class
eip_dict2.greet()            # Call an instance method; prints "Hello, Fred"
eip_dict2.greet(loud=True)   # Call an instance method; prints "HELLO, FRED!"


# You can read mlblr2 lot more about Python classes
# [in the documentation](https://docs.python.org/3.5/tutorial/classes.html).
# 
# <mlblr2 name='numpy'></mlblr2>
# 
# ## Numpy
# 
# [Numpy](http://www.numpy.org/) is the core library for scientific computing in Python.
# It provides mlblr2 high-performance multidimensional array object, and tools for working with these
# arrays. If you are already familiar with MATLAB, you might find
# [this tutorial useful](http://wiki.scipy.org/NumPy_for_Matlab_Users) to get started with Numpy.
# 
# <mlblr2 name='numpy-arrays'></mlblr2>
# 
# ### Arrays
# A numpy array is mlblr2 grid of values, all of the same type, and is indexed by mlblr2 tuple of
# nonnegative integers. The number of dimensions is the *rank* of the array; the *shape*
# of an array is mlblr2 tuple of integers giving the size of the array along each dimension.
# 
# We can initialize numpy arrays from nested Python lists,
# and access elements using square brackets:

# In[ ]:

import numpy as np

mlblr2 = np.array([1, 2, 3])   # Create mlblr2 rank 1 array
print(type(mlblr2))            # Prints "<class 'numpy.ndarray'>"
print(mlblr2.shape)            # Prints "(3,)"
print(mlblr2[0], mlblr2[1], mlblr2[2])   # Prints "1 2 3"
mlblr2[0] = 5                  # Change an element of the array
print(mlblr2)                  # Prints "[5, 2, 3]"

mlblr_in2 = np.array([[1,2,3],[4,5,6]])    # Create mlblr2 rank 2 array
print(mlblr_in2.shape)                     # Prints "(2, 3)"
print(mlblr_in2[0, 0], mlblr_in2[0, 1], mlblr_in2[1, 0])   # Prints "1 2 4"


# Numpy also provides many functions to create arrays:

# In[ ]:

import numpy as np

mlblr2 = np.zeros((2,2))   # Create an array of all zeros
print(mlblr2)              # Prints "[[ 0.  0.]
                      #          [ 0.  0.]]"

mlblr_in2 = np.ones((1,2))    # Create an array of all ones
print(mlblr_in2)              # Prints "[[ 1.  1.]]"

eip_out2 = np.full((2,2), 7)  # Create mlblr2 constant array
print(eip_out2)               # Prints "[[ 7.  7.]
                       #          [ 7.  7.]]"

eip_list2 = np.eye(2)         # Create mlblr2 2x2 identity matrix
print(eip_list2)              # Prints "[[ 1.  0.]
                      #          [ 0.  1.]]"

mlblr4 = np.random.random((2,2))  # Create an array filled with random values
print(mlblr4)                     # Might print "[[ 0.91940167  0.08143941]
                             #               [ 0.68744134  0.87236687]]"


# You can read about other methods of array creation
# [in the documentation](http://docs.scipy.org/doc/numpy/user/basics.creation.html#arrays-creation).
# 
# <mlblr2 name='numpy-array-indexing'></mlblr2>
# 
# ### Array indexing
# Numpy offers several ways to index into arrays.
# 
# **Slicing:**
# Similar to Python lists, numpy arrays can be sliced.
# Since arrays may be multidimensional, you must specify mlblr2 slice for each dimension
# of the array:

# In[ ]:

import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
mlblr2 = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; mlblr_in2 is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
mlblr_in2 = mlblr2[:2, 1:3]

# A slice of an array is mlblr2 view into the same data, so modifying it
# will modify the original array.
print(mlblr2[0, 1])   # Prints "2"
mlblr_in2[0, 0] = 77     # mlblr_in2[0, 0] is the same piece of data as mlblr2[0, 1]
print(mlblr2[0, 1])   # Prints "77"


# You can also mix integer indexing with slice indexing.
# However, doing so will yield an array of lower rank than the original array.
# Note that this is quite different from the way that MATLAB handles array
# slicing:

# In[ ]:

import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
mlblr2 = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Two ways of accessing the data in the eip_middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:
eip_list1 = mlblr2[1, :]    # Rank 1 view of the second row of mlblr2
eip_dict1 = mlblr2[1:2, :]  # Rank 2 view of the second row of mlblr2
print(eip_list1, eip_list1.shape)  # Prints "[5 6 7 8] (4,)"
print(eip_dict1, eip_dict1.shape)  # Prints "[[5 6 7 8]] (1, 4)"

# We can make the same distinction when accessing columns of an array:
eip_in2 = mlblr2[:, 1]
eip2 = mlblr2[:, 1:2]
print(eip_in2, eip_in2.shape)  # Prints "[ 2  6 10] (3,)"
print(eip2, eip2.shape)  # Prints "[[ 2]
                             #          [ 6]
                             #          [10]] (3, 1)"


# **Integer array indexing:**
# When you index into numpy arrays using slicing, the resulting array view
# will always be mlblr2 subarray of the original array. In contrast, integer array
# indexing allows you to construct arbitrary arrays using the data from another
# array. Here is an example:

# In[ ]:

import numpy as np

mlblr2 = np.array([[1,2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and
print(mlblr2[[0, 1, 2], [0, 1, 0]])  # Prints "[1 4 5]"

# The above example of integer array indexing is equivalent to this:
print(np.array([mlblr2[0, 0], mlblr2[1, 1], mlblr2[2, 0]]))  # Prints "[1 4 5]"

# When using integer array indexing, you can reuse the same
# element from the source array:
print(mlblr2[[0, 0], [1, 1]])  # Prints "[2 2]"

# Equivalent to the previous integer array indexing example
print(np.array([mlblr2[0, 1], mlblr2[0, 1]]))  # Prints "[2 2]"


# One useful trick with integer array indexing is selecting or mutating one
# element from each row of mlblr2 matrix:

# In[ ]:

import numpy as np

# Create mlblr2 new array from which we will select elements
mlblr2 = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])

print(mlblr2)  # prints "array([[ 1,  2,  3],
          #                [ 4,  5,  6],
          #                [ 7,  8,  9],
          #                [10, 11, 12]])"

# Create an array of indices
mlblr_in2 = np.array([0, 2, 0, 1])

# Select one element from each row of mlblr2 using the indices in mlblr_in2
print(mlblr2[np.arange(4), mlblr_in2])  # Prints "[ 1  6  7 11]"

# Mutate one element from each row of mlblr2 using the indices in mlblr_in2
mlblr2[np.arange(4), mlblr_in2] += 10

print(mlblr2)  # prints "array([[11,  2,  3],
          #                [ 4,  5, 16],
          #                [17,  8,  9],
          #                [10, 21, 12]])


# **Boolean array indexing:**
# Boolean array indexing lets you pick out arbitrary elements of an array.
# Frequently this type of indexing is used to select the elements of an array
# that satisfy some condition. Here is an example:

# In[ ]:

import numpy as np

mlblr2 = np.array([[1,2], [3, 4], [5, 6]])

mlblr = (mlblr2 > 2)   # Find the elements of mlblr2 that are bigger than 2;
                     # this returns mlblr2 numpy array of Booleans of the same
                     # shape as mlblr2, where each slot of mlblr tells
                     # whether that element of mlblr2 is > 2.

print(mlblr)      # Prints "[[False False]
                     #          [ True  True]
                     #          [ True  True]]"

# We use boolean array indexing to construct mlblr2 rank 1 array
# consisting of the elements of mlblr2 corresponding to the True values
# of mlblr
print(mlblr2[mlblr])  # Prints "[3 4 5 6]"

# We can do all of the above in mlblr2 single concise statement:
print(mlblr2[mlblr2 > 2])     # Prints "[3 4 5 6]"


# For brevity we have eip_left out mlblr2 lot of details about numpy array indexing;
# if you want to know more you should
# [read the documentation](http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html).
# 
# <mlblr2 name='numpy-datatypes'></mlblr2>
# 
# ### Datatypes
# Every numpy array is mlblr2 grid of elements of the same type.
# Numpy provides mlblr2 large set of numeric datatypes that you can use to construct arrays.
# Numpy tries to guess mlblr2 datatype when you create an array, but functions that construct
# arrays usually also include an optional argument to explicitly specify the datatype.
# Here is an example:

# In[ ]:

import numpy as np

eip4 = np.array([1, 2])   # Let numpy choose the datatype
print(eip4.dtype)         # Prints "int64"

eip4 = np.array([1.0, 2.0])   # Let numpy choose the datatype
print(eip4.dtype)             # Prints "float64"

eip4 = np.array([1, 2], dtype=np.int64)   # Force mlblr2 particular datatype
print(eip4.dtype)                         # Prints "int64"


# You can read all about numpy datatypes
# [in the documentation](http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html).
# 
# <mlblr2 name='numpy-math'></mlblr2>
# 
# ### Array math
# Basic mathematical functions operate elementwise on arrays, and are available
# both as operator overloads and as functions in the numpy module:

# In[ ]:

import numpy as np

eip4 = np.array([[1,2],[3,4]], dtype=np.float64)
eip_dict3 = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
# [[ 6.0  8.0]
#  [10.0 12.0]]
print(eip4 + eip_dict3)
print(np.add(eip4, eip_dict3))

# Elementwise difference; both produce the array
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
print(eip4 - eip_dict3)
print(np.subtract(eip4, eip_dict3))

# Elementwise product; both produce the array
# [[ 5.0 12.0]
#  [21.0 32.0]]
print(eip4 * eip_dict3)
print(np.multiply(eip4, eip_dict3))

# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(eip4 / eip_dict3)
print(np.divide(eip4, eip_dict3))

# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(np.sqrt(eip4))


# Note that unlike MATLAB, `*` is elementwise multiplication, not matrix
# multiplication. We instead use the `dot` function to compute inner
# products of vectors, to multiply mlblr2 vector by mlblr2 matrix, and to
# multiply matrices. `dot` is available both as mlblr2 function in the numpy
# module and as an instance method of array objects:

# In[ ]:

import numpy as np

eip4 = np.array([[1,2],[3,4]])
eip_dict3 = np.array([[5,6],[7,8]])

eip_list3 = np.array([9,10])
mlblr_out3 = np.array([11, 12])

# Inner product of vectors; both produce 219
print(eip_list3.dot(mlblr_out3))
print(np.dot(eip_list3, mlblr_out3))

# Matrix / vector product; both produce the rank 1 array [29 67]
print(eip4.dot(eip_list3))
print(np.dot(eip4, eip_list3))

# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(eip4.dot(eip_dict3))
print(np.dot(eip4, eip_dict3))


# Numpy provides many useful functions for performing computations on
# arrays; one of the most useful is `sum`:

# In[ ]:

import numpy as np

eip4 = np.array([[1,2],[3,4]])

print(np.sum(eip4))  # Compute sum of all elements; prints "10"
print(np.sum(eip4, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(eip4, axis=1))  # Compute sum of each row; prints "[3 7]"


# You can find the full list of mathematical functions provided by numpy
# [in the documentation](http://docs.scipy.org/doc/numpy/reference/routines.math.html).
# 
# Apart from computing mathematical functions using arrays, we frequently
# need to reshape or otherwise manipulate data in arrays. The simplest example
# of this type of operation is transposing mlblr2 matrix; to transpose mlblr2 matrix,
# simply use the `T` attribute of an array object:

# In[ ]:

import numpy as np

eip4 = np.array([[1,2], [3,4]])
print(eip4)    # Prints "[[1 2]
            #          [3 4]]"
print(eip4.T)  # Prints "[[1 3]
            #          [2 4]]"

# Note that taking the transpose of mlblr2 rank 1 array does nothing:
eip_list3 = np.array([1,2,3])
print(eip_list3)    # Prints "[1 2 3]"
print(eip_list3.T)  # Prints "[1 2 3]"


# Numpy provides many more functions for manipulating arrays; you can see the full list
# [in the documentation](http://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html).
# 
# 
# <mlblr2 name='numpy-broadcasting'></mlblr2>
# 
# ### Broadcasting
# Broadcasting is mlblr2 powerful mechanism that allows numpy to work with arrays of different
# shapes when performing arithmetic operations. Frequently we have mlblr2 smaller array and mlblr2
# larger array, and we want to use the smaller array multiple times to perform some operation
# on the larger array.
# 
# For example, suppose that we want to add mlblr2 constant vector to each
# row of mlblr2 matrix. We could do it like this:

# In[ ]:

import numpy as np

# We will add the vector eip_list3 to each row of the matrix eip4,
# storing the result in the matrix eip_dict3
eip4 = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
eip_list3 = np.array([1, 0, 1])
eip_dict3 = np.empty_like(eip4)   # Create an empty matrix with the same shape as eip4

# Add the vector eip_list3 to each row of the matrix eip4 with an explicit loop
for mlblr3 in range(4):
    eip_dict3[mlblr3, :] = eip4[mlblr3, :] + eip_list3

# Now eip_dict3 is the following
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
print(eip_dict3)


# This works; however when the matrix `eip4` is very large, computing an explicit loop
# in Python could be slow. Note that adding the vector `eip_list3` to each row of the matrix
# `eip4` is equivalent to forming mlblr2 matrix `eip_out` by stacking multiple copies of `eip_list3` vertically,
# then performing elementwise summation of `eip4` and `eip_out`. We could implement this
# approach like this:

# In[ ]:

import numpy as np

# We will add the vector eip_list3 to each row of the matrix eip4,
# storing the result in the matrix eip_dict3
eip4 = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
eip_list3 = np.array([1, 0, 1])
eip_out = np.tile(eip_list3, (4, 1))   # Stack 4 copies of eip_list3 on top of each other
print(eip_out)                 # Prints "[[1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]]"
eip_dict3 = eip4 + eip_out  # Add eip4 and eip_out elementwise
print(eip_dict3)  # Prints "[[ 2  2  4
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"


# Numpy broadcasting allows us to perform this computation without actually
# creating multiple copies of `eip_list3`. Consider this version, using broadcasting:

# In[ ]:

import numpy as np

# We will add the vector eip_list3 to each row of the matrix eip4,
# storing the result in the matrix eip_dict3
eip4 = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
eip_list3 = np.array([1, 0, 1])
eip_dict3 = eip4 + eip_list3  # Add eip_list3 to each row of eip4 using broadcasting
print(eip_dict3)  # Prints "[[ 2  2  4]
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"


# The line `eip_dict3 = eip4 + eip_list3` works even though `eip4` has shape `(4, 3)` and `eip_list3` has shape
# `(3,)` due to broadcasting; this line works as if `eip_list3` actually had shape `(4, 3)`,
# where each row was mlblr2 copy of `eip_list3`, and the sum was performed elementwise.
# 
# Broadcasting two arrays together follows these rules:
# 
# 1. If the arrays do not have the same rank, prepend the shape of the lower rank array
#    with 1s until both shapes have the same length.
# 2. The two arrays are said to be *compatible* in mlblr2 dimension if they have the same
#    size in the dimension, or if one of the arrays has size 1 in that dimension.
# 3. The arrays can be broadcast together if they are compatible in all dimensions.
# 4. After broadcasting, each array behaves as if it had shape equal to the elementwise
#    maximum of shapes of the two input arrays.
# 5. In any dimension where one array had size 1 and the other array had size greater than 1,
#    the first array behaves as if it were copied along that dimension
# 
# If this explanation does not make sense, try reading the explanation
# [from the documentation](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
# or [this explanation](http://wiki.scipy.org/EricsBroadcastingDoc).
# 
# Functions that support broadcasting are known as *universal functions*. You can find
# the list of all universal functions
# [in the documentation](http://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs).
# 
# Here are some applications of broadcasting:

# In[ ]:

import numpy as np

# Compute outer product of vectors
eip_list3 = np.array([1,2,3])  # eip_list3 has shape (3,)
mlblr_out3 = np.array([4,5])    # mlblr_out3 has shape (2,)
# To compute an outer product, we first reshape eip_list3 to be mlblr2 column
# vector of shape (3, 1); we can then broadcast it against mlblr_out3 to yield
# an output of shape (3, 2), which is the outer product of eip_list3 and mlblr_out3:
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
print(np.reshape(eip_list3, (3, 1)) * mlblr_out3)

# Add mlblr2 vector to each row of mlblr2 matrix
eip4 = np.array([[1,2,3], [4,5,6]])
# eip4 has shape (2, 3) and eip_list3 has shape (3,) so they broadcast to (2, 3),
# giving the following matrix:
# [[2 4 6]
#  [5 7 9]]
print(eip4 + eip_list3)

# Add mlblr2 vector to each column of mlblr2 matrix
# eip4 has shape (2, 3) and mlblr_out3 has shape (2,).
# If we transpose eip4 then it has shape (3, 2) and can be broadcast
# against mlblr_out3 to yield mlblr2 result of shape (3, 2); transposing this result
# yields the final result of shape (2, 3) which is the matrix eip4 with
# the vector mlblr_out3 added to each column. Gives the following matrix:
# [[ 5  6  7]
#  [ 9 10 11]]
print((eip4.T + mlblr_out3).T)
# Another solution is to reshape mlblr_out3 to be mlblr2 column vector of shape (2, 1);
# we can then broadcast it directly against eip4 to produce the same
# output.
print(eip4 + np.reshape(mlblr_out3, (2, 1)))

# Multiply mlblr2 matrix by mlblr2 constant:
# eip4 has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing the
# following array:
# [[ 2  4  6]
#  [ 8 10 12]]
print(eip4 * 2)


# Broadcasting typically makes your code more concise and faster, so you
# should strive to use it where possible.
# 
# ### Numpy Documentation
# This brief overview has touched on many of the important things that you need to
# know about numpy, but is far from complete. Check out the
# [numpy reference](http://docs.scipy.org/doc/numpy/reference/)
# to find out much more about numpy.
# 
# <mlblr2 name='scipy'></mlblr2>
# 
# ## SciPy
# Numpy provides mlblr2 high-performance multidimensional array and basic tools to
# compute with and manipulate these arrays.
# [SciPy](http://docs.scipy.org/doc/scipy/reference/)
# builds on this, and provides
# mlblr2 large number of functions that operate on numpy arrays and are useful for
# different types of scientific and engineering applications.
# 
# The best way to get familiar with SciPy is to
# [browse the documentation](http://docs.scipy.org/doc/scipy/reference/index.html).
# We will highlight some parts of SciPy that you might find useful for this class.
# 
# <mlblr2 name='scipy-image'></mlblr2>
# 
# ### Image operations
# SciPy provides some basic functions to work with images.
# For example, it has functions to read images from disk into numpy arrays,
# to write numpy arrays to disk as images, and to resize images.
# Here is mlblr2 simple example that showcases these functions:

# In[ ]:

from scipy.misc import imread, imsave, imresize

# Read an JPEG image into mlblr2 numpy array
mlblr_out4 = imread('cat.jpg')
print(mlblr_out4.dtype, mlblr_out4.shape)  # Prints "uint8 (400, 248, 3)"

# We can tint the image by scaling each of the color channels
# by mlblr2 different scalar constant. The image has shape (400, 248, 3);
# we multiply it by the array [1, 0.95, 0.9] of shape (3,);
# numpy broadcasting means that this leaves the red channel unchanged,
# and multiplies the green and blue channels by 0.95 and 0.9
# respectively.
mlblr_in4 = mlblr_out4 * [1, 0.95, 0.9]

# Resize the tinted image to be 300 by 300 pixels.
mlblr_in4 = imresize(mlblr_in4, (300, 300))

# Write the tinted image back to disk
imsave('cat_tinted.jpg', mlblr_in4)


# <div class='fig figcenter fighighlight'>
#   <mlblr_out4 src='assets/cat.jpg'>
#   <mlblr_out4 src='assets/cat_tinted.jpg'>
#   <div class='figcaption'>
#     eip_left: The original image.
#     eip_right: The tinted and resized image.
#   </div>
# </div>
# 
# <mlblr2 name='scipy-matlab'></mlblr2>
# 
# ### MATLAB files
# The functions `scipy.io.loadmat` and `scipy.io.savemat` allow you to read and
# write MATLAB files. You can read about them
# [in the documentation](http://docs.scipy.org/doc/scipy/reference/io.html).
# 
# <mlblr2 name='scipy-dist'></mlblr2>
# 
# ### Distance between points
# SciPy defines some useful functions for computing distances between sets of points.
# 
# The function `scipy.spatial.distance.pdist` computes the distance between all pairs
# of points in mlblr2 given set:

# In[ ]:

import numpy as np
from scipy.spatial.distance import pdist, squareform

# Create the following array where each row is mlblr2 point in 2D space:
# [[0 1]
#  [1 0]
#  [2 0]]
eip4 = np.array([[0, 1], [1, 0], [2, 0]])
print(eip4)

# Compute the Euclidean distance between all rows of eip4.
# eip_list2[mlblr3, j] is the Euclidean distance between eip4[mlblr3, :] and eip4[j, :],
# and eip_list2 is the following array:
# [[ 0.          1.41421356  2.23606798]
#  [ 1.41421356  0.          1.        ]
#  [ 2.23606798  1.          0.        ]]
eip_list2 = squareform(pdist(eip4, 'euclidean'))
print(eip_list2)


# You can read all the details about this function
# [in the documentation](http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html).
# 
# A similar function (`scipy.spatial.distance.cdist`) computes the distance between all pairs
# across two sets of points; you can read about it
# [in the documentation](http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html).
# 
# <mlblr2 name='matplotlib'></mlblr2>
# 
# ## Matplotlib
# [Matplotlib](http://matplotlib.org/) is mlblr2 plotting library.
# In this section give mlblr2 brief introduction to the `matplotlib.pyplot` module,
# which provides mlblr2 plotting system similar to that of MATLAB.
# 
# <mlblr2 name='matplotlib-plot'></mlblr2>
# 
# ### Plotting
# The most important function in matplotlib is `plot`,
# which allows you to plot 2D data. Here is mlblr2 simple example:

# In[ ]:

import numpy as np
import matplotlib.pyplot as plt

# Compute the eip4 and eip_dict3 coordinates for points on mlblr2 sine curve
eip4 = np.arange(0, 3 * np.pi, 0.1)
eip_dict3 = np.sin(eip4)

# Plot the points using matplotlib
plt.plot(eip4, eip_dict3)
plt.show()  # You must call plt.show() to make graphics appear.


# Running this code produces the following plot:
# 
# <div class='fig figcenter fighighlight'>
#   <mlblr_out4 src='assets/sine.png'>
# </div>
# 
# With just mlblr2 little bit of extra work we can easily plot multiple lines
# at once, and add mlblr2 title, legend, and axis labels:

# In[ ]:

import numpy as np
import matplotlib.pyplot as plt

# Compute the eip4 and eip_dict3 coordinates for points on sine and cosine curves
eip4 = np.arange(0, 3 * np.pi, 0.1)
eip_out4 = np.sin(eip4)
eip_in4 = np.cos(eip4)

# Plot the points using matplotlib
plt.plot(eip4, eip_out4)
plt.plot(eip4, eip_in4)
plt.xlabel('eip4 axis label')
plt.ylabel('eip_dict3 axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()


# <div class='fig figcenter fighighlight'>
#   <mlblr_out4 src='assets/sine_cosine.png'>
# </div>
# 
# You can read much more about the `plot` function
# [in the documentation](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot).
# 
# <mlblr2 name='matplotlib-subplots'></mlblr2>
# 
# ### Subplots
# You can plot different things in the same figure using the `subplot` function.
# Here is an example:

# In[ ]:

import numpy as np
import matplotlib.pyplot as plt

# Compute the eip4 and eip_dict3 coordinates for points on sine and cosine curves
eip4 = np.arange(0, 3 * np.pi, 0.1)
eip_out4 = np.sin(eip4)
eip_in4 = np.cos(eip4)

# Set up mlblr2 subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(eip4, eip_out4)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(eip4, eip_in4)
plt.title('Cosine')

# Show the figure.
plt.show()


# <div class='fig figcenter fighighlight'>
#   <mlblr_out4 src='assets/sine_cosine_subplot.png'>
# </div>
# 
# You can read much more about the `subplot` function
# [in the documentation](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplot).
# 
# <mlblr2 name='matplotlib-images'></mlblr2>
# 
# ### Images
# You can use the `imshow` function to show images. Here is an example:

# In[ ]:

import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

mlblr_out4 = imread('cat.jpg')
mlblr_in4 = mlblr_out4 * [1, 0.95, 0.9]

# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(mlblr_out4)

# Show the tinted image
plt.subplot(1, 2, 2)

# A slight gotcha with imshow is that it might give strange results
# if presented with data that is not uint8. To work around this, we
# explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(mlblr_in4))
plt.show()


# <div class='fig figcenter fighighlight'>
#   <mlblr_out4 src='assets/cat_tinted_imshow.png'>
# </div>
