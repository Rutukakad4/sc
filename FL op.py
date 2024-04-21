# Example to Demonstrate the
# Union of Two Fuzzy Sets
A = dict()
B = dict()
Y = dict()

A = {"a": 0.2, "b": 0.3, "c": 0.6, "d": 0.6}
B = {"a": 0.9, "b": 0.9, "c": 0.4, "d": 0.5}

print('The First Fuzzy Set is :', A)
print('The Second Fuzzy Set is :', B)


for A_key, B_key in zip(A, B):
	A_value = A[A_key]
	B_value = B[B_key]

	if A_value > B_value:
		Y[A_key] = A_value
	else:
		Y[B_key] = B_value

print('Fuzzy Set Union is :', Y)

#Intersection
for A_key, B_key in zip(A, B):
    A_value = A[A_key]
    B_value = B[B_key]

    if A_value < B_value:
        Y[A_key] = A_value
    else:
        Y[B_key] = B_value
print('Fuzzy Set Intersection is :', Y)

#Complement
for A_key in A:
   Y[A_key]= 1-A[A_key]

print('Fuzzy Set Complement is :', Y)

# Difference
for A_key, B_key in zip(A, B):
    A_value = A[A_key]
    B_value = B[B_key]
    B_value = 1 - B_value

    if A_value < B_value:
        Y[A_key] = A_value
    else:
        Y[B_key] = B_value

print('Fuzzy Set Difference is :', Y)



# Cartesian product
def findCart(arr1, arr2, n, n1):
  for i in range(0, n):
    for j in range(0, n1):
      print("{", arr1[i], ", ", arr2[j], "}, ", sep="", end="")
    print(end='\n')

# Driver code
arr1 = [1, 2, 3]  # first set
arr2 = [4, 5, 6]  # second set

n1 = len(arr1)  # sizeof(arr1[0])
n2 = len(arr2)  # sizeof(arr2[0]);

findCart(arr1, arr2, n1, n2)

# Triangular membership function
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import math
import random
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
a=float(input("Enter a: "))
b=float(input("Enter b: "))
m=(a+b)/2
X=(a,m,b)
Y=(0,1,0)
plt.plot(X,Y)
plt.show()

# Finding membership value of an element  x in the above FS
x=float(input("Enter the element: "))
if x<=a:
    mem = 0
elif x>a and x<=m:
    mem = (x-a)/(m-a)
elif x>m and x<b:
    mem = (b-x)/(b-m)
else:
    mem=0

print("Membership : ",mem)

# Cartesian Product
 n = int(input("\nEnter number of elements in first set (A): "))
 A = []
 B = []
 print("Enter elements for A:")
 for i in range(0, n):
  ele = float(input())
  A.append(ele)
m = int(input("\nEnter number of elements in second set (B): "))
print("Enter elements for B:")
for i in range(0, m):
  ele = float(input())
  B.append(ele)
print("A = {"+str(A)[1:-1]+"}")
print("B = {"+str(B)[1:-1]+"}")
cart_prod = []
cart_prod = [[0 for j in range(m)]for i in range(n)]
for i in range(n):
  for j in range(m):
    cart_prod[i][j] = min(A[i],B[j])
print("A x B = ")
for i in range(n):
  for j in range(m):
    print(cart_prod[i][j],end="  ")
  print("\n")
  
  
