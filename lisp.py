from typing import List as L
from typing import TypeVar, Generic

T = TypeVar('T')


class ParseError(Exception):
    pass


class EvalError(Exception):
    def __init__(self, msg, data=None):
        self.msg = msg
        self.data = data


# Lisp datatypes

class LispData(object):
    pass


class Symbol(str, LispData):
    def __repr__(self):
        return self


class String(str, LispData):
    def __repr__(self):
        return super().__repr__()


class Bool(int, LispData):  # cant inherit from bool
    def __repr__(self):
        return "true" if self else "false"


LispTrue = Bool(1)
LispFalse = Bool(0)


class List(list, LispData, Generic[T]):
    def __repr__(self):
        # TODO handle quotes nicely
        contents = " ".join([c.__repr__() for c in self])
        return "({})".format(contents)

    def __getitem__(self, item):
        # make slicing closed
        ret = list.__getitem__(self, item)
        if isinstance(item, slice):
            return List(ret)
        else:
            return ret


# TODO make these closed under math

class Int(int, LispData):
    pass


class Float(float, LispData):
    pass


# parser

def tokenize(chars: str) -> L[str]:
    "Convert a string of characters into a list of tokens."
    return chars.replace('(', ' ( ').replace(')', ' ) ').split()


def parse(program):
    "Read a Scheme expression from a string."
    tokens = tokenize(program)
    parsed = read_from_tokens(tokens)
    if tokens:
        raise ParseError("unexpected input: {}".format(tokens))
    return parsed


def read_from_tokens(tokens: L[str]) -> LispData:
    "Read an expression from a sequence of tokens."
    if not tokens:
        raise ParseError('unexpected EOF while reading')
    token = tokens.pop(0)
    if '(' == token:
        acc: List[LispData] = List()
        while tokens and tokens[0] != ')':
            acc.append(read_from_tokens(tokens))
        if not tokens:
            raise ParseError("unexpected EOF while reading")
        tokens.pop(0)  # pop off ')'
        return acc
    elif ')' == token:
        raise ParseError('unexpected )')
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
                    raise ParseError("unclosed string: {}".format(token))
                s = token[1:-1]
                if '"' in s:
                    raise ParseError("invalid string: {}".format(token))
                return String(s)
            else:
                return Symbol(token)


# cars and cdrs, cons

def cons(x: LispData, xs: List[LispData]):
    return List([x] + xs)


def car(xs: List[LispData]) -> LispData:
    return xs[0]


def cdr(xs: List[LispData]) -> List[LispData]:
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


# eval and apply


def eval(exp: LispData, env: dict) -> LispData:
    if is_self_evaluating(exp):
        return exp
    elif is_variable(exp):
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
        return eval_sequence(begin_actions(exp), env)
    elif is_cond(exp):
        return eval(cond_to_if(exp), env)
    elif is_application(exp):
        return apply(eval(operator(exp), env), list_of_values(operands(exp), env))
    else:
        raise EvalError("unknown expression type -- EVAL", exp)


def apply(procedure: LispData, arguments: List[LispData]) -> LispData:
    if is_primitive_procedure(procedure):
        return apply_primitive_procedure(procedure, arguments)
    elif is_compound_procedure(procedure):
        return eval_sequence(
            procedure_body(procedure),
            extend_environment(procedure_parameters(procedure),
                               arguments, procedure_environment(procedure))
        )
    else:
        raise EvalError("Unknown procedure type -- APPLY", procedure)


# representing expressions

# TODO this code evaluates operands from left to right. Exercise 4.1
# is to write a version that evaluates from right to left
#
# it would pretty much be the same but recurse from the right
def list_of_values(exps: List[LispData], env: dict) -> List[LispData]:
    if no_operands(exps):
        return List([])
    else:
        return List([eval(first_operand(exps), env)]) + list_of_values(rest_operands(exps), env)  # type: ignore


def eval_if(exp: LispData, env: dict):
    # test for equality with the true value in the implemented language
    if eval(if_predicate(exp), env) == LispTrue:
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
    set_variable_value(
        assignment_variable(exp),
        eval(assignment_value(exp), env),
        env
    )
    return Symbol("ok")  # XXX


def eval_definition(exp, env):
    define_variable(
        definition_variable(exp),
        eval(definition_value(exp), env),
        env
    )
    return Symbol("ok")  # XXX


def is_self_evaluating(exp):
    # NOTE our lisp has bools and apparently the one in the book
    # doesn't? this is probably ok
    return (isinstance(exp, Float) or isinstance(exp, Int)
            or isinstance(exp, String) or isinstance(exp, Bool))


def is_variable(exp):
    return isinstance(exp, Symbol)


def is_quoted(exp):
    return tagged_list(exp, Symbol("quote"))


def text_of_quotation(exp):
    n = len(exp)
    if n != 2:
        raise EvalError("quote -- wrong number of arguments {}".format(n))
    return cadr(exp)


def tagged_list(exp, tag):
    return isinstance(exp, List) and car(exp) == tag


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


def is_if(exp: LispData):
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


def is_begin(exp):
    return tagged_list(exp, Symbol("Begin"))


def begin_actions(exp):
    return cdr(exp)


def is_last_exp(seq):
    return not bool(cdr(seq))


def first_exp(seq):
    return car(seq)


def rest_exps(seq):
    return cdr(seq)


def sequence_to_exp(seq):
    if not seq:
        return seq
    elif is_last_exp(seq):
        return first_exp(seq)
    else:
        return make_begin(seq)


def make_begin(seq):
    return cons(Symbol("begin"), seq)


def is_application(seq):
    # TODO in the book this is `pair?`, so this is a bit bogus becuase
    # I don't have pairs but i think it will fine?
    return isinstance(seq, List)


def operator(exp):
    return car(exp)


def operands(exp):
    return cdr(exp)


def no_operands(ops):
    return not bool(ops)


def first_operand(ops):
    return car(ops)


def rest_operands(ops):
    return cdr(ops)


def is_cond(exp):
    return tagged_list(exp, Symbol("cond"))


def cond_clauses(exp):
    return cdr(exp)


def is_cond_else_clause(clause):
    return cond_predicate(clause) == Symbol("else")


def cond_predicate(clause):
    return car(clause)


def cond_actions(clause):
    return cdr(clause)


def cond_to_if(exp):
    return expand_clauses(cond_clauses(exp))


def expand_clauses(clauses):
    if not clauses:
        # TODO the book does not make it clear whether this should be
        # implemented-false or implementor-false, but I think the
        # former
        return LispFalse
    else:
        first = car(clauses)
        rest = cdr(clauses)
        if is_cond_else_clause(first):
            if not rest:
                return sequence_to_exp(cond_actions(first))
            else:
                raise EvalError("ELSE clause isn't last -- COND->IF", clauses)
        else:
            return make_if(
                cond_predicate(first),
                sequence_to_exp(cond_actions(first)),
                expand_clauses(rest)
            )


# evaluator data structures

# procedures


def make_procedure(parameters, body, env):
    return List([Symbol("procedure"), parameters, body, env])


def is_compound_procedure(exp):
    return tagged_list(exp, Symbol("procedure"))


def procedure_parameters(p):
    return cadr(p)


def procedure_body(p):
    return caddr(p)


def procedure_environment(p):
    return cadddr(p)


# environments

# NOTE I represent enviornments differently than the book does (using
# dictionaries, so a bit more natural in Python)

def is_empty(env):
    return not bool(env)


EMPTY_ENV: dict = {}


# A unique value that enviornments use to store the enviornment they
# were extended from (as opposed to bindings)
ENCLOSING = object()


def lookup_variable_value(var, env):
    if is_empty(env):
        raise EvalError("Unbound variable {}".format(var))
    if var in env:
        return env[var]
    else:
        return lookup_variable_value(var, env[ENCLOSING])


def extend_environment(vars, values, env):
    if len(vars) != len(values):
        if len(vars) > len(values):
            raise EvalError("Too few arguments supplied")
        else:
            raise EvalError("Too many arguments supplied")
    ret = dict(zip(vars, values))
    ret[ENCLOSING] = env
    return ret


def set_variable_value(var, value, env):
    if is_empty(env):
        raise EvalError("Unbound variable {}".format(var))
    if var in env:
        env[var] = value
    else:
        set_variable_value(var, value, env[ENCLOSING])


def define_variable(var, value, env):
    env[var] = value


# running the evaluator as a program


PRIMITIVE_PROCEDURES = {
    Symbol("car"): car,
    Symbol("cdr"): cdr,
    Symbol("cons"): cons,
    Symbol("null?"): lambda x: Bool(not bool(x)),
    Symbol("+"): lambda x, y: x + y,
    Symbol("-"): lambda x, y: x - y,
    Symbol("*"): lambda x, y: x * y,
    Symbol("/"): lambda x, y: x / y,
    Symbol("="): lambda x, y: x == y,
}


def setup_environment():
    initial_environment = {k: List([Symbol("primitive"), v])
                           for k, v in PRIMITIVE_PROCEDURES.items()}
    initial_environment[ENCLOSING] = {}
    return initial_environment


def is_primitive_procedure(exp):
    return tagged_list(exp, Symbol("primitive"))


def primitive_implementation(proc):
    return cadr(proc)


def apply_primitive_procedure(proc, args):
    return primitive_implementation(proc)(*args)


# driver loop

INPUT_PROMPT = ";;; M-Eval input: "
OUTPUT_PROMPT = ";;; M-Eval value: "


def driver_loop(global_env):
    while True:
        print()
        print(INPUT_PROMPT)
        # we have to parse, whereas lisp doesn't because of `read`
        user_input = '\n'.join(iter(input, ''))
        try:
            output = eval(parse(user_input), global_env)
            print(OUTPUT_PROMPT)
            user_print(output)
        except (ParseError, EvalError) as e:
            print(e)


def user_print(obj):
    if is_compound_procedure(obj):
        print(List([Symbol("compound-procedure"),
                    procedure_parameters(obj), procedure_body(obj), Symbol("<procedure-env>")]))
    else:
        print(obj)


def main():
    driver_loop(setup_environment())



if __name__ == "__main__":
    main()
