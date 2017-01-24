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


LispTrue = Bool(1)
LispFalse = Bool(0)


class List(list, LispData, Generic[T]):
    def __repr__(self):
        return f"List({super().__repr__()})"


class Int(int, LispData):
    def __repr__(self):
        return f"Int({super().__repr__()})"


class Float(float, LispData):
    def __repr__(self):
        return f"Float({super().__repr__()})"


# parser

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


# cars and cdrs, cons

def cons(x, xs):
    return List([x]) + xs


def car(xs):
    return xs[0]


def cdr(xs):
    return xs[1:]

# TODO make more efficient?
def caar(x): return car(car(x))
def cadr(x): return car(cdr(x))
def cdar(x): return cdr(car(x))
def cddr(x): return cdr(cdr(x))
def caaar(x): return car(car(car(x)))
def caadr(x): return car(car(cdr(x)))
def cadar(x): return car(cdr(car(x)))
def caddr(x): return car(cdr(cdr(x)))
def cdaar(x): return cdr(car(car(x)))
def cdadr(x): return cdr(car(cdr(x)))
def cddar(x): return cdr(cdr(car(x)))
def cdddr(x): return cdr(cdr(cdr(x)))
def caaaar(x): return car(car(car(car(x))))
def caaadr(x): return car(car(car(cdr(x))))
def caadar(x): return car(car(cdr(car(x))))
def caaddr(x): return car(car(cdr(cdr(x))))
def cadaar(x): return car(cdr(car(car(x))))
def cadadr(x): return car(cdr(car(cdr(x))))
def caddar(x): return car(cdr(cdr(car(x))))
def cadddr(x): return car(cdr(cdr(cdr(x))))
def cdaaar(x): return cdr(car(car(car(x))))
def cdaadr(x): return cdr(car(car(cdr(x))))
def cdadar(x): return cdr(car(cdr(car(x))))
def cdaddr(x): return cdr(car(cdr(cdr(x))))
def cddaar(x): return cdr(cdr(car(car(x))))
def cddadr(x): return cdr(cdr(car(cdr(x))))
def cdddar(x): return cdr(cdr(cdr(car(x))))
def cddddr(x): return cdr(cdr(cdr(cdr(x))))




# evaluator

python_eval = eval  # why not


# TODO types?
def eval(exp, env):
    if is_self_evaluating(exp):
        return exp
    elif is_variable(exp):
        # TODO make sure environment supports this sort of kv lookup
        return lookup_variable_value(exp, env)
    elif is_quoted(exp):
        return text_of_quotation(exp)
    elif is_assignment(exp):
        return eval_assignment(exp, env)
    elif is_definition(exp):
        return eval_definition(exp, env)
    elif is_if(exp):
        return eval_if(exp, env)
    elif is_lambda(exp):
        return make_procedure(lambda_parameters(exp), lambda_body(exp), env)
    elif is_begin(exp):
        eval_sequence(begin_actions(exp), env)
    elif is_cond(exp):
        eval(cond_to_if(exp), env)
    elif is_application(exp):
        apply(eval(operator(exp), env), list_of_values(operands(exp), env))
    else:
        raise RuntimeError("unknown expression type -- EVAL", exp)


def apply(procedure, arguments):
    if is_primitive_procedure(procedure):
        return apply_primitive_procedure(procedure, arguments)
    elif is_compound_procedure(procedure):
        return eval_sequence(
            procedure_body(procedure),
            extend_environment(procedure_parameters(procedure),
                               arguments, procedure_environment(procedure))
        )
    else:
        raise RuntimeError("Unknown procedure type -- APPLY", procedure)



# TODO this code evaluates operands from left to right. Exercise 4.1
# is to write a version that evaluates from right to left
#
# it would pretty much be the same but recurse from the right
def list_of_values(exps, env):
    if no_operands(exps):
        return []
    else:
        [eval(first_operand(exps), env)] + list_of_values(rest_operands(exps), env)


def eval_if(exp, env):
    # test for equality with the true value in the implemented language
    if eval(if_predicate(exp)) == LispTrue:
        return eval(if_consequent(exp), env)
    else:
        return eval(if_alternative(exp), env)


def eval_sequence(exps, env):
    if is_last_exp(exps):
        return eval(first_exp(exps), env)
    else:
        eval(first_exp(exps), env)
        return eval_sequence(rest_exps(exps), env)


def eval_assignment(exp, env):
    # TODO is [] syntax going to work for this?
    env[assignment_variable(exp)] = eval(assignment_value(exp), env)
    return Symbol("ok")  # XXX


def eval_definition(exp, env):
    env[definition_variable(exp)] = eval(definition_value(exp), env)
    return Symbol("ok")


def self_evaluating(exp):
    # NOTE our lisp has bools and apparently the one in the book
    # doesn't? this is probably ok
    return (isinstance(exp, Float) or isinstance(exp, Int)
            or isinstance(exp, String) or isinstance(exp, Bool))


def is_variable(exp):
    return isinstance(exp, Symbol)


def is_quoted(exp):
    return tagged_list(exp, Symbol("quote"))


def tagged_list(exp, tag):
    return exp and car(exp) == tag


def is_assignment(exp):
    return tagged_list(exp, Symbol("set!"))


def assignment_variable(exp):
    return cadr(exp)


def assignment_value(exp):
    return caddr(exp)


def is_definition(exp):
    return tagged_list(exp, Symbol("define"))


def definition_variable(exp):
    if isinstance(cadr(exp), Symbol):
        return cadr(exp)
    else:
        return caadr(exp)


def definition_value(exp):
    if isinstance(cadr(exp), Symbol):
        return caddr(exp)
    else:
        return make_lambda(
            cdadr(exp),  # formal parameters
            cddr(exp)  # body
        )


def is_lambda(exp):
    return tagged_list(exp, Symbol("lambda"))


def lambda_parameters(exp):
    return cadr(exp)


def lambda_body(exp):
    return cddr(exp)


def make_lambda(parameters, body):
    # TODO it's possible I've implemented cons incorrectly. I'm unsure
    # whether this should give me like [lambda, [x], x] or [lambda,
    # [x], [x]] for the identity function
    return cons(Symbol("lambda"), cons(parameters, body))


def is_if(exp):
    return tagged_list(exp, Symbol("if"))


def if_predicate(exp):
    return cadr(exp)


def if_consequent(exp):
    return caddr(exp)


def if_alternative(exp):
    if cdddr(exp):
        return cadddr(exp)
    else:
        return LispFalse


def make_if(predicate, consequent, alternative):
    List([Symbol("if"), predicate, consequent, alternative])
