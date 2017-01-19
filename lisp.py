from typing import List as L
from typing import TypeVar, Generic

T = TypeVar('T')


# Lisp datatypes

class LispData(object):
    pass


class Symbol(str, LispData):
    pass


class String(str, LispData):
    pass


class List(list, LispData, Generic[T]):
    pass


class Int(int, LispData):
    pass


class Float(float, LispData):
    pass


def tokenize(chars: str) -> L[str]:
    "Convert a string of characters into a list of tokens."
    return chars.replace('(', ' ( ').replace(')', ' ) ').split()


def read_from_tokens(tokens: L[str]) -> LispData:
    "Read an expression from a sequence of tokens."
    if len(tokens) == 0:
        raise SyntaxError('unexpected EOF while reading')
    token = tokens.pop(0)
    if '(' == token:
        acc: List[LispData] = List()  # flake8: noqa
        while tokens[0] != ')':
            acc.append(read_from_tokens(tokens))
        tokens.pop(0)  # pop off ')'
        return acc
    elif ')' == token:
        raise SyntaxError('unexpected )')
    else:
        return atom(token)


def atom(token: str) -> LispData:
    try:
        return Int(int(token))
    except ValueError:
        try:
            return Float(float(token))
        except ValueError:
            # TODO: chars?
            if token.startswith('"'):
                return String(token)
            else:
                return Symbol(token)
