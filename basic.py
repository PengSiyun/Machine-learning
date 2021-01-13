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

#########################################
#Strings and Dictionaries
#########################################
#####use of \
# \' ->	'	e.g. 'What\'s up?'	What's up?
# \" ->	"	e.g. "That's \"cool\""	That's "cool"
# \\ ->	\	e.g. "Look, a mountain: /\\"	Look, a mountain: /\
# \n  -> start a new line

#triple quote syntax for strings lets us include newlines literally (i.e. by just hitting 'Enter' on our keyboard, rather than using the special '\n' sequence).
triplequoted_hello = """hello
world"""
print(triplequoted_hello)

###### Indexing
planet = 'Pluto'
planet[0]
planet[-3:]
#loop over them
[x+'! ' for x in planet]
# ALL CAPS
planet.upper()
# Searching for the first index of a substring
planet.index('o')

##### Going between strings and lists: .split() and .join()
# str.split() turns a string into a list of smaller strings, breaking on
# whitespace by default. This is super useful for taking you from one 
# big string to a list of words.
claim = "Pluto is a planet!"
words = claim.split()
words
#Occasionally you'll want to split on something other than whitespace:
datestr = '1956-01-31'
year, month, day = datestr.split('-')
# str.join() takes us in the other direction, sewing a list of strings up 
# into one long string, using the string it was called on as a separator.
'/'.join([month, day, year])
position = 9
planet + ", you'll always be the " + position + "th planet to me."
planet + ", you'll always be the " + str(position) + "th planet to me."
#alternative: str.format()
"{}, you'll always be the {}th planet to me.".format(planet, position)
#other function of str.format()
pluto_mass = 1.303 * 10**22
earth_mass = 5.9722 * 10**24
population = 52910390
#2 decimal points 3 decimal points, format as percent     separate with commas
"{} weighs about {:.2} kilograms ({:.3%} of Earth's mass). It is home to {:,} Plutonians.".format(
    planet, pluto_mass, pluto_mass / earth_mass, population,
)
# Referring to format() arguments by index, starting from 0
s = """Pluto's a {0}.
No, it's a {1}.
{0}!
{1}!""".format('planet', 'dwarf planet')
print(s)

#####Dictionaries: for mapping keys to values
numbers = {'one':1, 'two':2, 'three':3}
numbers['one']
# add another key, value pair
numbers['eleven'] = 11
numbers
# get initial of every planet in the list
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
planet_to_initial = {x: x[0] for x in planets}
planet_to_initial
# in operator tells us whether something is a key in the dictionary
'Saturn' in planet_to_initial
# A for loop over a dictionary will loop over its keys
for k in numbers:
    print("{} = {}".format(k, numbers[k]))
#We can access a collection of all the keys or all the values with dict.keys() and dict.values()
# Get all the initials, sort them alphabetically, and put them in a space-separated string.
' '.join(sorted(planet_to_initial.values()))
# The very useful dict.items() method lets us iterate over the keys and values of a dictionary simultaneously. (In Python jargon, an item refers to a key, value pair)
for planet, initial in planet_to_initial.items():
    print("{} begins with \"{}\"".format(planet.rjust(10), initial))

#####practice
#Returns whether the input string is a valid (5 digit) zip code
def is_valid_zip(zip_str):
    return len(zip_str) == 5 and zip_str.isdigit()

def word_search(documents, keyword):
    # list to hold the indices of matching documents
    indices = [] 
    # Iterate through the indices (i) and elements (doc) of documents
    for i, doc in enumerate(documents):
        # Split the string doc into a list of words (according to whitespace)
        tokens = doc.split()
        # Make a transformed list where we 'normalize' each word to facilitate matching.
        # Periods and commas are removed from the end of each word, and it's set to all lowercase.
        normalized = [token.rstrip('.,').lower() for token in tokens]
        # Is there a match? If so, update the list of matching indices.
        if keyword.lower() in normalized:
            indices.append(i)
    return indices

#########################################
#Working with External Libraries
#########################################
#Three tools for understanding strange objects
#1: type() (what is this thing?)
#2: dir() (what can I do with it?)
#3: help() (tell me more)




