# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 15:11:35 2020

@author: Peng
"""

def pos(x) :
    return True if x > 0 else False
print(pos(1))


#########################################
# List
#########################################
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
#Python uses zero-based indexing, so the first element has index 0
planets[0]
#Elements at the end of the list can be accessed with negative numbers, starting from -1:
planets[-1]    
planets[0:3]
#Lists are "mutable", meaning they can be modified "in place"
planets[3] = 'Malacandra'
planets
# Add Pluto to the end
planets.append('Pluto')
#removes and returns the last element of a list:
planets.pop()
#the length of a list
len(planets)
# The planets sorted in alphabetical order
sorted(planets)
# Is Earth a planet?
"Earth" in planets
#Help on list object
help(planets)

#########################################
# Loops
#########################################

#####for loop
s = 'steganograpHy is the practicE of conceaLing a file, message, image, or video within another fiLe, message, image, Or video.'
msg = ''
# print all the uppercase letters in s, one at a time
for char in s:
    if char.isupper():
        print(char, end='') #end specify how each char is connected
#loop through numbers
for i in range(5):
    print("Doing important work. i =", i)
    
####while loop, which iterates until some condition is met
i = 0
while i < 10:
    print(i, end=' ')
    i += 1 #Equivalent to i = i + 1
    
####List comprehensions
squares = [n**2 for n in range(10)]
squares
#add if
#ex. 1
short_planets = [planet for planet in planets if len(planet) < 6]
short_planets
#ex. 2
loud_short_planets = [planet.upper() + '!' 
                      for planet in planets 
                      if len(planet) < 6]
loud_short_planets # str.upper() returns an all-caps version of a string
#ex. 3: Return the number of negative numbers in the given list
def count_negatives(nums):
    return len([num for num in nums if num < 0])
#ex. 4: Return whether the given list of numbers is lucky. A lucky list contains at least one number divisible by 7.
def has_lucky_number(nums):
    return any([num % 7 == 0 for num in nums])
#ex. 5: Return a list with the same length as L, where the value at index i is  True if L[i] is greater than thresh, and False otherwise.
def elementwise_greater_than(L, thresh):
    return [x > thresh for x in L] # x > thresh return true or false
X=[1,2,3]
elementwise_greater_than(X, 2)
#ex. 6: Given a list of meals served over some period of time, return True if the same meal has ever been served two days in a row, and False otherwise.
def menu_is_boring(meals):
    # Iterate over all indices of the list, except the last one
    for i in range(len(meals)-1):
        if meals[i] == meals[i+1]:
            return True #return ends loop
    return False