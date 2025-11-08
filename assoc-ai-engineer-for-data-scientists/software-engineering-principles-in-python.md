# Software Engineering & Data Science
## Python, data science, & software engineering
- Modularity
    - Improve readability
- Documentation
    - Show users how to use your project
- Automated testing
## Introduction to packages & documentation
PyPi
`pip`
`help()`
    - help(numpy.busday_count)
## Conventions and PEP 8
pycodestyle dict_to_array.py

# Writing a Python Module
## Writing Your First Package
- Directory: `package_name`
    - python file: `__init__.py `
    - functionality file: `utils.py`, `data.py`, etc.

## Adding functionality to packages
`utils.py`: file names should be all lowercase

## Making your package portable
`requirements.txt`
`setup.py`
```python
from setuptools import setup

setup (name='my_package',
    version='0.0.1',
    description='An example package for DataCamp.',
    author='Adam Spannbauer',
    author_email='spannbaueradam@gmail.com',
    packages=['my_package'],
    install_requires=['matplotlib',
        'numpy == 1.15.4',
        'pycodestyle>=2.4.0'])
```

# Utilizing Classes
## Adding classes to a package
```python
class MyClass:
    """
    Documentation here
    """

    def __init__(self, value):
        self.attribute = value

from .my_class import MyClass


import my_package

my_instance = my_package.MyClass(value='attribute')

print(my_instance.attribute)
```
### Self convention

## Leveraing class
```python
from .token_utils import tokenize

class Document:
    def __init__(self, text, token_regex = r'[a-zA-Z]+'):
        self.text = text
        self.tokens = self._tokenize()
    def _tokenize(self):
        return tokenize(self.text)

doc = Document('text')
print(doc.text)
```

## Classes and the DRY principle
```python
from .parent_class import ParentClass

class ChildClass(ParentClass):
    def __init__(self):
        ParentClass.__init__(self)
        self.child_attribute = "I'm a child class attribute"

child_class = ChildClass()
print(child_class.child_attribute)
print(child_class.parent_attribute)
```

### Multilevel inheritance
```python
# Import needed package
import text_analyzer

# Create instance of document
my_doc = text_analyzer.Document(datacamp_tweets)

# Run help on my_doc's plot method
help(my_doc.plot_counts)

# Plot the word_counts of my_doc
my_doc.plot_counts()
```

```python
# Define a Tweet class that inherits from SocialMedia
class Tweets(SocialMedia):
    def __init__(self, text):
        # Call parent's __init__ with super()
        super().__init__(text)
        # Define retweets attribute with non-public method
        self.retweets = self._process_retweets()

    def _process_retweets(self):
        # Filter tweet text to only include retweets
        retweet_text = filter_lines(self.text, first_chars='RT')
        # Return retweet_text as a SocialMedia object
        return SocialMedia(retweet_text)
```

```python
# Import needed package
import text_analyzer

# Create instance of Tweets
my_tweets = text_analyzer.Tweets(datacamp_tweets)

# Plot the most used hashtags in the tweets
my_tweets.plot_counts('hashtag_counts')


# Plot the most used hashtags in the retweets
my_tweets.retweets.plot_counts('hashtag_counts')



```
# Maintainability
## Documentation
- Comments
    - for devs
    - in line
- Docstrings
    - for users
    - this will output if users call `help()`
- Sample proper docstring
```python
# Complete the function's docstring
def tokenize(text, regex=r'[a-zA-z]+'):
  """Split text into tokens using a regular expression

  :param text: text to be tokenized
  :param regex: regular expression used to match tokens using re.findall 
  :return: a list of resulting tokens

  >>> tokenize('the rain in spain')
  ['the', 'rain', 'in', 'spain']
  """
  return re.findall(regex, text, flags=re.IGNORECASE)

# Print the docstring
help(tokenize)
```

## Readability countrs
Zen of Python: `import this`
- Descriptive naming
- Keep it simple
- Know When to refactor
- sample of correct documentation
```python
def hypotenuse_length(leg_a, leg_b):
    """Find the length of a right triangle's hypotenuse

    :param leg_a: length of one leg of triangle
    :param leg_b: length of other leg of triangle
    :return: length of hypotenuse
    
    >>> hypotenuse_length(3, 4)
    5
    """
    return math.sqrt(leg_a**2 + leg_b**2)


# Print the length of the hypotenuse with legs 6 & 8
print(hypotenuse_length(6,8))
```

## Unit Testing
### Using doctest
```python
def square(x):
    """Square the number x

    :param x: number to square
    :return: x squared

    >>> square(3)

    9

    return x ** 3
    """

import doctest
doctest.testmod()
```

### pytest
#### Writing unit tests
```python
from text_analyzer import Document

# Test tokens attribute on Document object
def test_document_tokens():
    doc = Document('a e i o u')

    assert doc. tokens == ['a', 'e', 'i', 'o' , 'u' ]

# Test edge case of blank document
def test_document_empty():
    doc = Document('')

    assert doc. tokens == []
    assert doc.word_counts == Counter()
```

`/work_dir $ pytest`
`/work_dir $ pytest tests/test_document.py`

## Documentation & testing in practice
- Sphix documentation
```python
from text_analyzer import Document

class SocialMedia(Document):
    """Analyze text data from social media
    
    :param text: social media text to analyze

    :ivar hashtag_counts: Counter object containing counts of hashtags used in text
    :ivar mention_counts: Counter object containing counts of @mentions used in text
    """
    def __init__(self, text):
        Document.__init__(self, text)
        self.hashtag_counts = self._count_hashtags()
        self.mention_counts = self._count_mentions()

```