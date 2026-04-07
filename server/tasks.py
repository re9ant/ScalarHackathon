"""
Task bank for the CodeDebugger RL Environment.

Each task contains a buggy Python code snippet that the agent must fix.
Tasks are categorized by type and difficulty.
"""

import random
from typing import Optional

TASKS = [
    # ─── EASY: Syntax Errors ────────────────────────────────────────────────
    {
        "id": "syn_001",
        "title": "Missing colon in if statement",
        "difficulty": "easy",
        "category": "syntax",
        "buggy_code": """\
x = 10
if x > 5
    print("x is greater than 5")
""",
        "description": "This code tries to check if x > 5, but there's a syntax error preventing it from running.",
        "expected_output": "x is greater than 5",
        "hint": "Check the end of the if statement line — Python needs a special character there.",
        "solution": """\
x = 10
if x > 5:
    print("x is greater than 5")
""",
    },
    {
        "id": "syn_002",
        "title": "Missing closing parenthesis",
        "difficulty": "easy",
        "category": "syntax",
        "buggy_code": """\
name = "Alice"
print("Hello, " + name
""",
        "description": "This code should print a greeting but has a syntax error.",
        "expected_output": "Hello, Alice",
        "hint": "Count the opening and closing parentheses in the print statement.",
        "solution": """\
name = "Alice"
print("Hello, " + name)
""",
    },
    {
        "id": "syn_003",
        "title": "Wrong indentation",
        "difficulty": "easy",
        "category": "syntax",
        "buggy_code": """\
def greet(name):
print(f"Hello, {name}!")

greet("Bob")
""",
        "description": "This function should greet a person but has an indentation error.",
        "expected_output": "Hello, Bob!",
        "hint": "The body of a function must be indented.",
        "solution": """\
def greet(name):
    print(f"Hello, {name}!")

greet("Bob")
""",
    },
    {
        "id": "syn_004",
        "title": "Missing quotes",
        "difficulty": "easy",
        "category": "syntax",
        "buggy_code": """\
message = Hello World
print(message)
""",
        "description": "This code tries to store and print a string, but fails to run.",
        "expected_output": "Hello World",
        "hint": "String literals need to be enclosed in quotes.",
        "solution": """\
message = "Hello World"
print(message)
""",
    },
    {
        "id": "syn_005",
        "title": "Missing colon in function def",
        "difficulty": "easy",
        "category": "syntax",
        "buggy_code": """\
def add(a, b)
    return a + b

result = add(3, 4)
print(result)
""",
        "description": "This function should add two numbers but has a syntax error in the definition.",
        "expected_output": "7",
        "hint": "Function definitions end with a special character before the body.",
        "solution": """\
def add(a, b):
    return a + b

result = add(3, 4)
print(result)
""",
    },
    # ─── EASY: Runtime Errors ───────────────────────────────────────────────
    {
        "id": "run_001",
        "title": "Division by zero",
        "difficulty": "easy",
        "category": "runtime",
        "buggy_code": """\
total = 100
count = 0
average = total / count
print(f"Average: {average}")
""",
        "description": "This code tries to compute an average, but crashes at runtime.",
        "expected_output": "Average: 0",
        "hint": "Check what happens when count is 0 before dividing.",
        "solution": """\
total = 100
count = 0
if count == 0:
    average = 0
else:
    average = total / count
print(f"Average: {average}")
""",
    },
    {
        "id": "run_002",
        "title": "Index out of range",
        "difficulty": "easy",
        "category": "runtime",
        "buggy_code": """\
fruits = ["apple", "banana", "cherry"]
print(fruits[3])
""",
        "description": "This code should print the last fruit but crashes.",
        "expected_output": "cherry",
        "hint": "List indices start at 0. A list with 3 items has indices 0, 1, 2.",
        "solution": """\
fruits = ["apple", "banana", "cherry"]
print(fruits[2])
""",
    },
    {
        "id": "run_003",
        "title": "Key error in dictionary",
        "difficulty": "easy",
        "category": "runtime",
        "buggy_code": """\
person = {"name": "Alice", "age": 30}
print(person["email"])
""",
        "description": "This code tries to access a person's email but crashes.",
        "expected_output": "not found",
        "hint": "Use .get() to safely access dictionary keys with a default value.",
        "solution": """\
person = {"name": "Alice", "age": 30}
print(person.get("email", "not found"))
""",
    },
    {
        "id": "run_004",
        "title": "Type error: concatenating str and int",
        "difficulty": "easy",
        "category": "runtime",
        "buggy_code": """\
age = 25
message = "I am " + age + " years old"
print(message)
""",
        "description": "This code should print a message about age but crashes at runtime.",
        "expected_output": "I am 25 years old",
        "hint": "You can't concatenate strings and integers directly in Python.",
        "solution": """\
age = 25
message = "I am " + str(age) + " years old"
print(message)
""",
    },
    {
        "id": "run_005",
        "title": "NoneType error",
        "difficulty": "easy",
        "category": "runtime",
        "buggy_code": """\
def get_name():
    name = "Alice"

result = get_name()
print(result.upper())
""",
        "description": "This code should print the name in uppercase, but crashes.",
        "expected_output": "ALICE",
        "hint": "The function doesn't return anything. Functions that don't return explicitly return None.",
        "solution": """\
def get_name():
    name = "Alice"
    return name

result = get_name()
print(result.upper())
""",
    },
    # ─── MEDIUM: Logic Errors ───────────────────────────────────────────────
    {
        "id": "log_001",
        "title": "Off-by-one in range",
        "difficulty": "medium",
        "category": "logic",
        "buggy_code": """\
total = 0
for i in range(1, 10):
    total += i
print(total)
""",
        "description": "This code tries to sum numbers from 1 to 10 (inclusive), but the result is wrong.",
        "expected_output": "55",
        "hint": "Python's range(a, b) goes up to but does NOT include b.",
        "solution": """\
total = 0
for i in range(1, 11):
    total += i
print(total)
""",
    },
    {
        "id": "log_002",
        "title": "Wrong comparison operator",
        "difficulty": "medium",
        "category": "logic",
        "buggy_code": """\
def is_even(n):
    return n % 2 = 0

print(is_even(4))
print(is_even(7))
""",
        "description": "This function should return True for even numbers and False for odd, but has a logic error.",
        "expected_output": "True\nFalse",
        "hint": "Be careful about the difference between = (assignment) and == (comparison).",
        "solution": """\
def is_even(n):
    return n % 2 == 0

print(is_even(4))
print(is_even(7))
""",
    },
    {
        "id": "log_003",
        "title": "Swapped condition in while loop",
        "difficulty": "medium",
        "category": "logic",
        "buggy_code": """\
count = 0
while count >= 5:
    print(count)
    count += 1
""",
        "description": "This code should print numbers from 0 to 4, but prints nothing.",
        "expected_output": "0\n1\n2\n3\n4",
        "hint": "The while condition starts false immedialy. Think about when count=0 should loop.",
        "solution": """\
count = 0
while count < 5:
    print(count)
    count += 1
""",
    },
    {
        "id": "log_004",
        "title": "Wrong variable in return",
        "difficulty": "medium",
        "category": "logic",
        "buggy_code": """\
def celsius_to_fahrenheit(c):
    f = (c * 9/5) + 32
    return c

print(celsius_to_fahrenheit(100))
""",
        "description": "This function converts Celsius to Fahrenheit but returns the wrong value.",
        "expected_output": "212.0",
        "hint": "The function computes the right value but returns the wrong variable.",
        "solution": """\
def celsius_to_fahrenheit(c):
    f = (c * 9/5) + 32
    return f

print(celsius_to_fahrenheit(100))
""",
    },
    {
        "id": "log_005",
        "title": "String comparison case sensitivity",
        "difficulty": "medium",
        "category": "logic",
        "buggy_code": """\
user_input = "YES"
if user_input == "yes":
    print("Confirmed!")
else:
    print("Not confirmed")
""",
        "description": "The code should confirm when the user types YES, yes, or any case variant.",
        "expected_output": "Confirmed!",
        "hint": "String comparison is case-sensitive. Convert both strings to the same case before comparing.",
        "solution": """\
user_input = "YES"
if user_input.lower() == "yes":
    print("Confirmed!")
else:
    print("Not confirmed")
""",
    },
    {
        "id": "log_006",
        "title": "Mutation in loop",
        "difficulty": "medium",
        "category": "logic",
        "buggy_code": """\
numbers = [1, 2, 3, 4, 5]
for num in numbers:
    if num % 2 == 0:
        numbers.remove(num)
print(numbers)
""",
        "description": "This code tries to remove all even numbers from a list, but the result is wrong.",
        "expected_output": "[1, 3, 5]",
        "hint": "Modifying a list while iterating over it causes elements to be skipped. Use a copy or list comprehension.",
        "solution": """\
numbers = [1, 2, 3, 4, 5]
numbers = [num for num in numbers if num % 2 != 0]
print(numbers)
""",
    },
    {
        "id": "log_007",
        "title": "Accumulator not reset",
        "difficulty": "medium",
        "category": "logic",
        "buggy_code": """\
def sum_list(lst):
    for num in lst:
        total = 0
        total += num
    return total

print(sum_list([1, 2, 3, 4, 5]))
""",
        "description": "This function should sum a list of numbers but returns only the last element.",
        "expected_output": "15",
        "hint": "The accumulator variable total is being reset inside the loop.",
        "solution": """\
def sum_list(lst):
    total = 0
    for num in lst:
        total += num
    return total

print(sum_list([1, 2, 3, 4, 5]))
""",
    },
    {
        "id": "log_008",
        "title": "Integer division instead of float",
        "difficulty": "medium",
        "category": "logic",
        "buggy_code": """\
a = 7
b = 2
result = a // b
print(result)
""",
        "description": "This code should compute the exact division 7/2 = 3.5, but gives the wrong answer.",
        "expected_output": "3.5",
        "hint": "// is integer (floor) division. Use / for floating point division.",
        "solution": """\
a = 7
b = 2
result = a / b
print(result)
""",
    },
    # ─── MEDIUM: Algorithm Errors ───────────────────────────────────────────
    {
        "id": "alg_001",
        "title": "Bubble sort direction wrong",
        "difficulty": "medium",
        "category": "algorithm",
        "buggy_code": """\
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] < arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

print(bubble_sort([64, 34, 25, 12, 22, 11, 90]))
""",
        "description": "This bubble sort should sort in ascending order but produces descending order.",
        "expected_output": "[11, 12, 22, 25, 34, 64, 90]",
        "hint": "The comparison operator in the swap condition determines sort direction.",
        "solution": """\
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

print(bubble_sort([64, 34, 25, 12, 22, 11, 90]))
""",
    },
    {
        "id": "alg_002",
        "title": "Fibonacci off by one",
        "difficulty": "medium",
        "category": "algorithm",
        "buggy_code": """\
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

print(fibonacci(8))
""",
        "description": "This function should return the first 8 Fibonacci numbers, but the list has wrong length.",
        "expected_output": "[0, 1, 1, 2, 3, 5, 8, 13]",
        "hint": "The function returns the correct numbers but check whether range(2, n) generates n-2 iterations.",
        "solution": """\
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

print(fibonacci(8))
""",
    },
    {
        "id": "alg_003",
        "title": "Binary search wrong bounds",
        "difficulty": "hard",
        "category": "algorithm",
        "buggy_code": """\
def binary_search(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

arr = [1, 3, 5, 7, 9, 11, 13]
print(binary_search(arr, 7))
""",
        "description": "This binary search should find the index of 7 in the sorted list, but may crash or return wrong value.",
        "expected_output": "3",
        "hint": "The initial right bound is wrong. Also check whether `right = mid - 1` is correct with `left < right` condition.",
        "solution": """\
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

arr = [1, 3, 5, 7, 9, 11, 13]
print(binary_search(arr, 7))
""",
    },
    {
        "id": "alg_004",
        "title": "Factorial uses wrong base case",
        "difficulty": "medium",
        "category": "algorithm",
        "buggy_code": """\
def factorial(n):
    if n == 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))
print(factorial(0))
""",
        "description": "This factorial function works for n>=1 but crashes for n=0.",
        "expected_output": "120\n1",
        "hint": "0! = 1 by mathematical convention. The base case needs to handle n=0.",
        "solution": """\
def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))
print(factorial(0))
""",
    },
    {
        "id": "alg_005",
        "title": "Palindrome check off",
        "difficulty": "medium",
        "category": "algorithm",
        "buggy_code": """\
def is_palindrome(s):
    return s == s[1:-1]

print(is_palindrome("racecar"))
print(is_palindrome("hello"))
""",
        "description": "This function checks if a string is a palindrome but gives wrong results.",
        "expected_output": "True\nFalse",
        "hint": "Reversing a string in Python uses s[::-1], not s[1:-1].",
        "solution": """\
def is_palindrome(s):
    return s == s[::-1]

print(is_palindrome("racecar"))
print(is_palindrome("hello"))
""",
    },
    # ─── HARD: Complex Logic ────────────────────────────────────────────────
    {
        "id": "hard_001",
        "title": "Mutable default argument",
        "difficulty": "hard",
        "category": "logic",
        "buggy_code": """\
def append_item(item, lst=[]):
    lst.append(item)
    return lst

print(append_item(1))
print(append_item(2))
print(append_item(3))
""",
        "description": "This function should return a new list each call, but the lists keep growing.",
        "expected_output": "[1]\n[2]\n[3]",
        "hint": "Never use mutable objects as default argument values in Python. Use None instead.",
        "solution": """\
def append_item(item, lst=None):
    if lst is None:
        lst = []
    lst.append(item)
    return lst

print(append_item(1))
print(append_item(2))
print(append_item(3))
""",
    },
    {
        "id": "hard_002",
        "title": "Closure variable capture bug",
        "difficulty": "hard",
        "category": "logic",
        "buggy_code": """\
funcs = []
for i in range(5):
    funcs.append(lambda: i)

print([f() for f in funcs])
""",
        "description": "This code should create 5 functions that return 0,1,2,3,4 respectively, but they all return the same value.",
        "expected_output": "[0, 1, 2, 3, 4]",
        "hint": "Lambda captures the variable i by reference, not by value. Use a default argument to capture the current value.",
        "solution": """\
funcs = []
for i in range(5):
    funcs.append(lambda x=i: x)

print([f() for f in funcs])
""",
    },
    {
        "id": "hard_003",
        "title": "Shallow copy vs deep copy",
        "difficulty": "hard",
        "category": "logic",
        "buggy_code": """\
original = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
copy = original.copy()
copy[0][0] = 99
print(original[0][0])
""",
        "description": "This code tries to modify a copy without affecting the original, but the original is changed.",
        "expected_output": "1",
        "hint": "list.copy() creates a shallow copy — nested lists are still shared. Use copy.deepcopy().",
        "solution": """\
import copy
original = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
cp = copy.deepcopy(original)
cp[0][0] = 99
print(original[0][0])
""",
    },
    {
        "id": "hard_004",
        "title": "Generator exhaustion",
        "difficulty": "hard",
        "category": "logic",
        "buggy_code": """\
def gen_numbers():
    for i in range(5):
        yield i

nums = gen_numbers()
first_pass = list(nums)
second_pass = list(nums)
print(len(first_pass), len(second_pass))
""",
        "description": "This code should print '5 5' but prints wrong values because generators can only be iterated once.",
        "expected_output": "5 5",
        "hint": "A generator is exhausted after one iteration. You need to recreate it for the second pass.",
        "solution": """\
def gen_numbers():
    for i in range(5):
        yield i

nums = gen_numbers()
first_pass = list(nums)
nums = gen_numbers()
second_pass = list(nums)
print(len(first_pass), len(second_pass))
""",
    },
    {
        "id": "hard_005",
        "title": "Integer identity vs equality",
        "difficulty": "hard",
        "category": "logic",
        "buggy_code": """\
a = 1000
b = 1000
if a is b:
    print("same")
else:
    print("different")
""",
        "description": "This code checks if two variables with value 1000 are the same object. Fix it to check value equality instead.",
        "expected_output": "same",
        "hint": "Python's `is` checks object identity (same memory), not value equality. Use == for value comparison.",
        "solution": """\
a = 1000
b = 1000
if a == b:
    print("same")
else:
    print("different")
""",
    },
    {
        "id": "hard_006",
        "title": "Dict comprehension key collision",
        "difficulty": "hard",
        "category": "algorithm",
        "buggy_code": """\
words = ["apple", "banana", "avocado", "blueberry", "cherry"]
first_letter_map = {word[0]: word for word in words}
print(len(first_letter_map))
""",
        "description": "This code should create a map grouping words by first letter, but words are being overwritten. Fix it to group all words with the same first letter into a list.",
        "expected_output": "3",
        "hint": "When multiple words share the same first letter, the dict comprehension overwrites previous entries. Use a defaultdict or group them into lists.",
        "solution": """\
from collections import defaultdict
words = ["apple", "banana", "avocado", "blueberry", "cherry"]
first_letter_map = defaultdict(list)
for word in words:
    first_letter_map[word[0]].append(word)
print(len(first_letter_map))
""",
    },
    {
        "id": "hard_007",
        "title": "Missing return in recursion",
        "difficulty": "hard",
        "category": "algorithm",
        "buggy_code": """\
def find_max(lst, idx=0, current_max=None):
    if idx == len(lst):
        return current_max
    if current_max is None or lst[idx] > current_max:
        current_max = lst[idx]
    find_max(lst, idx + 1, current_max)

print(find_max([3, 1, 4, 1, 5, 9, 2, 6]))
""",
        "description": "This recursive function should find the maximum value in a list but always returns None.",
        "expected_output": "9",
        "hint": "The recursive call's result is never returned to the caller.",
        "solution": """\
def find_max(lst, idx=0, current_max=None):
    if idx == len(lst):
        return current_max
    if current_max is None or lst[idx] > current_max:
        current_max = lst[idx]
    return find_max(lst, idx + 1, current_max)

print(find_max([3, 1, 4, 1, 5, 9, 2, 6]))
""",
    },
    {
        "id": "hard_008",
        "title": "Thread-unsafe counter",
        "difficulty": "hard",
        "category": "runtime",
        "buggy_code": """\
import threading

counter = 0

def increment(n):
    global counter
    for _ in range(n):
        counter += 1

threads = [threading.Thread(target=increment, args=(1000,)) for _ in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(counter == 5000)
""",
        "description": "This code tries to increment a counter from multiple threads. Fix it to use a thread-safe approach so the result is always 5000.",
        "expected_output": "True",
        "hint": "Use threading.Lock() to protect the counter increment operation.",
        "solution": """\
import threading

counter = 0
lock = threading.Lock()

def increment(n):
    global counter
    for _ in range(n):
        with lock:
            counter += 1

threads = [threading.Thread(target=increment, args=(1000,)) for _ in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(counter == 5000)
""",
    },
    # ─── MORE MEDIUM TASKS ──────────────────────────────────────────────────
    {
        "id": "log_009",
        "title": "Print inside function not returned",
        "difficulty": "medium",
        "category": "logic",
        "buggy_code": """\
def square(n):
    print(n * n)

result = square(6)
print(result)
""",
        "description": "This code should compute and use the square of 6, but result is None.",
        "expected_output": "36\n36",
        "hint": "The function prints the value but doesn't return it.",
        "solution": """\
def square(n):
    return n * n

result = square(6)
print(result)
print(result)
""",
    },
    {
        "id": "log_010",
        "title": "String immutability",
        "difficulty": "medium",
        "category": "logic",
        "buggy_code": """\
s = "hello world"
s[0] = "H"
print(s)
""",
        "description": "This code tries to capitalize the first letter of a string, but strings are immutable in Python.",
        "expected_output": "Hello world",
        "hint": "Strings are immutable in Python. You need to create a new string.",
        "solution": """\
s = "hello world"
s = s[0].upper() + s[1:]
print(s)
""",
    },
    {
        "id": "run_006",
        "title": "Unpack too many values",
        "difficulty": "easy",
        "category": "runtime",
        "buggy_code": """\
data = (10, 20, 30)
a, b = data
print(a, b)
""",
        "description": "This code tries to unpack a tuple but there are too many values.",
        "expected_output": "10 20",
        "hint": "The tuple has 3 items but only 2 variables. Either add a variable or use slicing.",
        "solution": """\
data = (10, 20, 30)
a, b, _ = data
print(a, b)
""",
    },
    {
        "id": "alg_006",
        "title": "FizzBuzz wrong conditions",
        "difficulty": "easy",
        "category": "algorithm",
        "buggy_code": """\
for i in range(1, 16):
    if i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    elif i % 3 == 0 and i % 5 == 0:
        print("FizzBuzz")
    else:
        print(i)
""",
        "description": "This FizzBuzz implementation never prints 'FizzBuzz'. Fix the condition order.",
        "expected_output": "1\n2\nFizz\n4\nBuzz\nFizz\n7\n8\nFizz\nBuzz\n11\nFizz\n13\n14\nFizzBuzz",
        "hint": "The FizzBuzz condition (divisible by both 3 and 5) must be checked first, before the individual checks.",
        "solution": """\
for i in range(1, 16):
    if i % 3 == 0 and i % 5 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)
""",
    },
    {
        "id": "alg_007",
        "title": "Count occurrences wrong",
        "difficulty": "medium",
        "category": "algorithm",
        "buggy_code": """\
def count_char(s, char):
    count = 0
    for c in s:
        if c == char:
            count = 1
    return count

print(count_char("mississippi", "s"))
""",
        "description": "This function should count how many times 's' appears in 'mississippi' (answer: 4), but returns 1.",
        "expected_output": "4",
        "hint": "The counter is being set to 1 instead of incremented.",
        "solution": """\
def count_char(s, char):
    count = 0
    for c in s:
        if c == char:
            count += 1
    return count

print(count_char("mississippi", "s"))
""",
    },
]


def get_task(task_id: Optional[str] = None, difficulty: Optional[str] = None, category: Optional[str] = None) -> dict:
    """
    Get a task by ID, or randomly select one by difficulty/category.
    
    Args:
        task_id: Specific task ID to retrieve
        difficulty: Filter by difficulty ('easy', 'medium', 'hard')
        category: Filter by category ('syntax', 'runtime', 'logic', 'algorithm')
    
    Returns:
        Task dictionary
    """
    if task_id:
        for task in TASKS:
            if task["id"] == task_id:
                return task
        raise ValueError(f"Task '{task_id}' not found")
    
    pool = TASKS
    if difficulty:
        pool = [t for t in pool if t["difficulty"] == difficulty]
    if category:
        pool = [t for t in pool if t["category"] == category]
    
    if not pool:
        raise ValueError(f"No tasks found for difficulty={difficulty}, category={category}")
    
    return random.choice(pool)


def get_all_task_ids() -> list:
    """Return a list of all task IDs."""
    return [t["id"] for t in TASKS]


def get_task_metadata() -> list:
    """Return lightweight metadata for all tasks (no solution/hint)."""
    return [
        {
            "id": t["id"],
            "title": t["title"],
            "difficulty": t["difficulty"],
            "category": t["category"],
        }
        for t in TASKS
    ]
