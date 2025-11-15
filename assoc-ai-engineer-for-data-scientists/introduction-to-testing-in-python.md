# Creating Tests with pytest
```python
import pytest

def squared(number):
    return number * number
def test_squared():
    assert squared(-2) == squared(2)
```
## Context managers
- a python object that is used by declaring a `with` statement
- we use context managers to set up and tear down temporary context
```python
with open("hello_world.txt", 'w') as hello_file:
    hello_file.write("Hello world")
```
## pytest: raises
- when you expect the test to raise an Exception
```python
import pytest

def division(a,b):
    return a/b
def test_raises():
    with pytest.raises(ZeroDivisionError):
        division(a=25, b=0)
```
## Invoking pytest from CLI
`pytest slides.py`
`pytest tests_ex.py -k "squared"`

## Applying test markers
- `Decorator`: a design pattern in Python that allows a user to add new functionality to an existing object without modifying its structure
- Test markers syntax are started with `@pytest.mark` decorator
```python
import pytest

def get_length(string):
    return len(string)

@pytest.mark.skip
def test_get_len():
    assert get_length('123') == 3
```

## Skip and skipif markers
- `@pytest.mark.skip`
- `@pytest.mark.skipif`

```python
import pytest

# skip
@pytest.mark.skip
def test_get_len():
    assert get_length('123') == 3

# skip if
@pytest.mark.skipif('2 * 2 == 5')
def test_get_len():
    assert get_length('abc') == 3
```
## Xfail marker
- use when you expect a test to be failed
```python
import pytest

@pytest.mark.xfail
def test_gen_seq():
    assert gen_sequence(-1)
```

# Pytest Fixtures
## Introduction to fixture
- Fixuture: a prepared environment that can be used for a test execution
- Fixture setup: a process of preparing the environment and setting up resources that are required by one or more tests 
```python
import pytest

@pytest.fixture
def data():
    return [0,1,2,3,4]
```
### How to use fixtures
- prepare software and tests
- find environment preparation
- create a fixture
    - declare the `@pytest.fixture` decorator
    - implement the fixture function
- use the created fixture
    - pass the fixture name to the test function
    - run tests
### Chain fixtures requests
- Chain fixtures requests - a pytest feature, that allows a fixture to use another fixture
- helps
    - establish dependencies between fixtures
    - keep the code modular
```python
@pytest.fixture
def setup_data():
    return "I am a fixture"
@pytest.fixture
def process_data(setup_data):
    return setup_data.upper()

def test_process_data(process)data:
    assert process_data == "I AM A FIXTURE"
```
### How to use chain requests
- prepare the proggram we want to test
- prepare the testing functions
- prepare the pytest fixtures
- pass the fixture name to the other fixture signature

## Fixture autouse
- Autouse argument
    - an optional boolean argument of a fixture
    - can be passed to the fixture decorator
    - when `autouse=True` function is executing regardless of a request
    - helps to reduce the amount of redundant fixture
- When to use
    - in case we need to apply certain environment preparations or modifications for all tests
```python
import pytest
import pandas as pd

@pytest.fixture(autouse=True)
def set_pd_options():
    pd.set_option('display.max_columns',5000)
```
## Fixture teardown
- a process of cleaning up resources that were allocated or created during the setup of a testing environment
Why use
- It is important to clean the environment at the end of a test. If one does not use teardown, itcan lead to significant issues:
    - Memory leaks
    - Low speed of execution and performance issues
    - Invalid test results
    - Pipeline failures and errors

```python
def lazy_increment(n):
    for i in range(n):
        yield i

f = lazy_increment(5)
next(f)
next(f)
next(f)
```
```python
@pytest.fixture
def init_list():
    return []
@pytest.fixture(autouse=True)
def add_numbers_to_list(init_list):
    # Fixture Setup    
    init_list.extend([i for i in range(10)])
    # Fixture output
    yield init_list
    # Teardown statement    
    init_list.clear()

def test_9(init_list):
    assert 9 in init_list

```

# Basic Testing Types
## Unit testing
```python
def sum_of_arr(array:list) -> int:
    return sum(array)

# Test Case 1: regular array
def test_regular():
    assert sum_of_arr([1, 2, 3]) == 6
    assert sum_of_arr([100, 150]) == 250

# Test Case 2: empty List
def test_empty ():
    assert sum_of_arr([]) == 0

# Test Case 3: one number

def test_one_number():
    assert sum_of_arr([10]) == 10
    assert sum_of_arr([0]) == 0
```
## Feature testing
```python
import pandas as pd
import pytest

df = pd.read_csv('laptops.csv')

def filter_data_by_manuf(df, manufacturer_name):
    filtered_df = df[df["Manufacturer"] == manufacturer_name]
    return filtered_df

# Feature test function
def test_unique():
    manuf_name = 'Apple'
    filtered = filter_data_by_manuf(df, manuf_name)
    assert filtered['Manufacturer'].nunique() == 1
    assert filtered['Manufacturer'].unique() == [manuf_name]
```

## Integration test
```python
import pandas as pd
import pytest

# Fixture to read the dataframe
@pytest.fixture
def get_df():
    return pd.read_csv('https://assets.datacamp.com/production/repositories/6253/datasets/757c6cb769f7effc5f5496050ea4d73e4586c2dd/laptops_train.csv')

# Integration test function
def test_get_df(get_df):
    # Check the type
    assert type(get_df) == pd.DataFrame
    # Check the number of rows
    assert get_df.shape[0] > 0
```
# Writing tests with unittest
## Meeting the Unit Test
```python
def func_factorial(number):
    if number < 0:
        raise ValueError('Factorial is not defined for negative values')
    factorial = 1
    while number > 1:
        factorial = factorial * number
        number = number - 1
    return factorial

class TestFactorial(unittest.TestCase):
    def test_positives(self):
        # Add the test for testing positives here
        self.assertEqual(func_factorial(5), 120) 
```
```python
import unittest
def func_factorial(number):
    if number < 0:
        raise ValueError('Factorial is not defined for negative values')
    factorial = 1
    while number > 1:
        factorial = factorial * number
        number = number - 1
    return factorial

class TestFactorial(unittest.TestCase):
    def test_negatives(self):
      	# Add the test for testing negatives here
        with self.assertRaises(ValueError):
            func_factorial(-1)
```
## CLI Interface
- `python3 -m unittest test_sqneg.py`
- `python3 -m unittest -k "SomeStringOrPattern" test_script.py`: Run test cases that match the substring
- `python3 -m unittest -f test_script.py`: stop test on the first error/failure
- `python3 -m unittest -c test_script.py`: lets to interrup the test by pushing Ctrl - C
- `python3 -m unittest -v test_script.py`: Verbose flag, run tests with more detail

## Fixtures in unittest
```python
import unittest

class TestLi(unittest.TestCase):
    # Fixture setup method
    defsetUp(self):        
        self.li = [i for i inrange(100)]
    
    # Fixture teardown method
    deftearDown(self):        
        self.li.clear()
    
    # Test method
    deftest_your_list(self):        
        self.assertIn(99, self.li)        
        self.assertNotIn(100, self.li)
```

## Practical Examples
```python
import pytest
import pandas as pd

DF_PATH = "/usr/local/share/salaries.csv"
@pytest.fixture
def read_df():
    return pd.read_csv(DF_PATH)

def get_grouped(df):
    return df.groupby('work_year').agg({'salary': 'describe'})['salary']

def test_feature_2022(read_df):
    salary_by_year = get_grouped(read_df)
    salary_2022 = salary_by_year.loc[2022, '50%']
    # Check the median type here
    assert isinstance(salary_2022, float)
    # Check the median is greater than zero
    assert 0 < salary_2022

# Use benchmark here
def test_reading_speed(benchmark):
    benchmark(pd.read_csv, DF_PATH)

```

```python
import unittest
import pandas as pd
DF_PATH = 'https://assets.datacamp.com/production/repositories/6253/datasets/f015ac99df614ada3ef5e011c168054ca369d23b/energy_truncated.csv'

def get_data():
    return pd.read_csv(DF_PATH)

def min_country(df):
    return df['VALUE'].idxmin()

class TestDF(unittest.TestCase):
    def setUp(self):
        self.df = get_data()
        self.df.drop('previousYearToDate', axis=1, inplace=True)
        self.df = self.df.groupby('COUNTRY')\
            .agg({'VALUE': 'sum'})

    def test_NAs(self):
        # Check the number of nulls
        self.assertEqual(self.df.isna().sum().sum(), 0)

    def test_argmax(self):
        # Check that min_country returns a string
        self.assertIsInstance(min_country(self.df), str)

    def tearDown(self):
        self.df.drop(self.df.index, inplace=True)

```