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