from typing import List as L
from typing import TypeVar, Generic

T = TypeVar('T')


# Lisp datatypes

class LispData(object):
    pass

class Symbol(str, LispData):
    def __repr__(self):
        return f"Symbol({super().__repr__()})"


class String(str, LispData):
    def __repr__(self):
        return f"String({super().__repr__()})"


class Bool(int, LispData):  # cant inherit from bool
    def __repr__(self):
        to_show = "true" if self else "false"
        return f"Bool({to_show})"


class List(list, LispData, Generic[T]):
    def __repr__(self):
        return f"List({super().__repr__()})"


class Int(int, LispData):
    def __repr__(self):
        return f"Int({super().__repr__()})"


class Float(float, LispData):
    def __repr__(self):
        return f"Float({super().__repr__()})"


def tokenize(chars: str) -> L[str]:
    "Convert a string of characters into a list of tokens."
    return chars.replace('(', ' ( ').replace(')', ' ) ').split()


def parse(program):
    "Read a Scheme expression from a string."
    tokens = tokenize(program)
    parsed = read_from_tokens(tokens)
    if tokens:
        raise SyntaxError(f"unexpected input: {tokens}")
    return parsed


def read_from_tokens(tokens: L[str]) -> LispData:
    "Read an expression from a sequence of tokens."
    print(tokens)
    if not tokens:
        raise SyntaxError('unexpected EOF while reading')
    token = tokens.pop(0)
    if '(' == token:
        acc: List[LispData] = List()
        while tokens and tokens[0] != ')':
            acc.append(read_from_tokens(tokens))
        if not tokens:
            raise SyntaxError("unexpected EOF while reading")
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
            if token == "true":
                return Bool(True)
            elif token == "false":
                return Bool(False)
            elif token.startswith('"'):
                if not token.endswith('"'):
                    raise SyntaxError(f"unclosed string: {token}")
                s = token[1:-1]
                if '"' in s:
                    raise SyntaxError(f"invalid string: {token}")
                return String(s)
            else:
                return Symbol(token)
