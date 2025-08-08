# dsl/parser.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

import re

# ===========================
# Tokenizer
# ===========================

TokenType = str

@dataclass(frozen=True)
class Token:
    type: TokenType
    value: Any
    pos: int

_WHITESPACE = re.compile(r"\s+")
_NUMBER = re.compile(r"""
    (?:
        \d+\.\d+ |   # float like 12.34
        \d+\.\d* |   # float like 12.
        \.\d+   |    # float like .34
        \d+          # int
    )
""", re.VERBOSE)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_STRING = re.compile(r"""
    (?:'([^'\\]|\\.)*')   # single-quoted
    |                     # or
    (?:"([^"\\]|\\.)*")   # double-quoted
""", re.VERBOSE)

KEYWORDS = {
    "and": "AND",
    "or": "OR",
    "not": "NOT",
    "true": "BOOL",
    "false": "BOOL",
}

OPERATORS = {
    "==": "EQ",
    "!=": "NE",
    ">=": "GE",
    "<=": "LE",
    ">": "GT",
    "<": "LT",
}

PUNCT = {
    "(": "LPAREN",
    ")": "RPAREN",
    ",": "COMMA",
}

@dataclass
class LexError(Exception):
    msg: str
    pos: int

def tokenize(s: str) -> List[Token]:
    i = 0
    n = len(s)
    out: List[Token] = []

    def push(ttype: TokenType, val: Any):
        out.append(Token(ttype, val, i))

    while i < n:
        # whitespace
        m = _WHITESPACE.match(s, i)
        if m:
            i = m.end()
            if i >= n:
                break

        # operators (two-char first)
        op2 = s[i:i+2]
        if op2 in OPERATORS:
            push(OPERATORS[op2], op2)
            i += 2
            continue

        # single-char ops
        op1 = s[i:i+1]
        if op1 in OPERATORS:
            push(OPERATORS[op1], op1)
            i += 1
            continue

        # punctuation
        if op1 in PUNCT:
            push(PUNCT[op1], op1)
            i += 1
            continue

        # number
        m = _NUMBER.match(s, i)
        if m:
            txt = m.group(0)
            val = float(txt) if ('.' in txt) else int(txt)
            push("NUMBER", val)
            i = m.end()
            continue

        # string
        m = _STRING.match(s, i)
        if m:
            raw = m.group(0)
            # strip quotes and unescape basic escapes
            if raw[0] == "'":
                val = bytes(raw[1:-1], "utf-8").decode("unicode_escape")
            else:
                val = bytes(raw[1:-1], "utf-8").decode("unicode_escape")
            push("STRING", val)
            i = m.end()
            continue

        # identifier / keyword / boolean literal
        m = _IDENT.match(s, i)
        if m:
            ident = m.group(0)
            lower = ident.lower()
            if lower in KEYWORDS:
                ttype = KEYWORDS[lower]
                val = True if lower == "true" else False if lower == "false" else lower
                push(ttype, val)
            else:
                push("IDENT", ident)
            i = m.end()
            continue

        raise LexError(f"Unexpected character '{s[i]}'", i)

    return out


# ===========================
# AST Nodes
# ===========================

class Expr:
    def eval(self, env: Dict[str, Any], funcs: Dict[str, Callable[..., Any]]) -> Any:
        raise NotImplementedError

@dataclass
class Literal(Expr):
    value: Any
    def eval(self, env, funcs): return self.value

@dataclass
class Identifier(Expr):
    name: str
    def eval(self, env, funcs):
        if self.name in env:
            return env[self.name]
        # allow nested access a.b if env has dicts (optional)
        if "." in self.name:
            cur: Any = env
            for part in self.name.split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    raise KeyError(f"Unknown identifier '{self.name}'")
            return cur
        raise KeyError(f"Unknown identifier '{self.name}'")

@dataclass
class UnaryOp(Expr):
    op: str  # 'NOT'
    rhs: Expr
    def eval(self, env, funcs):
        v = self.rhs.eval(env, funcs)
        if self.op == "NOT":
            return not bool(v)
        raise ValueError(f"Unknown unary op {self.op}")

@dataclass
class BinaryOp(Expr):
    op: str  # 'AND'/'OR' or comparators 'EQ','NE','GT','LT','GE','LE'
    lhs: Expr
    rhs: Expr
    def eval(self, env, funcs):
        if self.op == "AND":
            return bool(self.lhs.eval(env, funcs)) and bool(self.rhs.eval(env, funcs))
        if self.op == "OR":
            return bool(self.lhs.eval(env, funcs)) or bool(self.rhs.eval(env, funcs))

        lv = self.lhs.eval(env, funcs)
        rv = self.rhs.eval(env, funcs)

        if self.op == "EQ": return lv == rv
        if self.op == "NE": return lv != rv
        if self.op == "GT": return lv > rv
        if self.op == "LT": return lv < rv
        if self.op == "GE": return lv >= rv
        if self.op == "LE": return lv <= rv

        raise ValueError(f"Unknown binary op {self.op}")

@dataclass
class Call(Expr):
    name: str
    args: List[Expr]
    def eval(self, env, funcs):
        if self.name not in funcs:
            raise NameError(f"Unknown function '{self.name}'")
        fn = funcs[self.name]
        vals = [a.eval(env, funcs) for a in self.args]
        return fn(*vals)


# ===========================
# Parser (Shunting-yard to AST)
# ===========================

@dataclass
class ParseError(Exception):
    msg: str
    pos: int

# precedence (higher means tighter binding)
# NOT > comparisons > AND > OR
PREC = {
    "NOT": 4,
    "EQ": 3, "NE": 3, "GT": 3, "LT": 3, "GE": 3, "LE": 3,
    "AND": 2,
    "OR": 1,
}
RIGHT_ASSOC = {"NOT"}  # right-assoc

def _to_ast(output_stack: List[Union[Expr, Token]], op: Token):
    """Pop nodes from output_stack and push AST nodes based on operator token."""
    t = op.type
    if t == "NOT":
        if not output_stack:
            raise ParseError("Missing operand for 'not'", op.pos)
        rhs = output_stack.pop()
        if isinstance(rhs, Token):
            raise ParseError("Invalid operand for 'not'", op.pos)
        output_stack.append(UnaryOp(op="NOT", rhs=rhs))
        return

    # binary
    if len(output_stack) < 2:
        raise ParseError("Binary operator missing operands", op.pos)
    rhs = output_stack.pop()
    lhs = output_stack.pop()
    if isinstance(rhs, Token) or isinstance(lhs, Token):
        raise ParseError("Invalid operands", op.pos)
    output_stack.append(BinaryOp(op=t, lhs=lhs, rhs=rhs))

def parse(tokens: List[Token]) -> Expr:
    output: List[Union[Expr, Token]] = []
    ops: List[Token] = []

    i = 0
    n = len(tokens)

    def push_value(tok: Token):
        if tok.type == "NUMBER":
            output.append(Literal(tok.value))
        elif tok.type == "STRING":
            output.append(Literal(tok.value))
        elif tok.type == "BOOL":
            output.append(Literal(bool(tok.value)))
        elif tok.type == "IDENT":
            output.append(Identifier(tok.value))
        else:
            raise ParseError(f"Unexpected token {tok.type}", tok.pos)

    while i < n:
        tok = tokens[i]

        if tok.type in {"NUMBER", "STRING", "BOOL", "IDENT"}:
            # lookahead for function call IDENT '('
            if tok.type == "IDENT" and (i + 1 < n and tokens[i+1].type == "LPAREN"):
                # parse function call
                fname = tok.value
                i += 2  # skip IDENT and '('
                args: List[Expr] = []
                expect_expr = True
                paren = 1
                # simple arg parser until matching ')'
                start_pos = tok.pos
                while i < n and paren > 0:
                    tt = tokens[i]
                    if tt.type == "RPAREN":
                        paren -= 1
                        i += 1
                        if paren == 0:
                            break
                        else:
                            # inside nested calls
                            args.append(Identifier(")"))  # impossible branch, but keep sane
                    elif tt.type == "LPAREN":
                        # nested expression: we delegate to a mini-parse
                        # We'll do a recursive descent by collecting sub-tokens until balanced paren
                        j = i + 1
                        depth = 1
                        while j < n and depth > 0:
                            if tokens[j].type == "LPAREN": depth += 1
                            elif tokens[j].type == "RPAREN": depth -= 1
                            j += 1
                        if depth != 0:
                            raise ParseError("Unbalanced parentheses in function args", tt.pos)
                        subexpr = parse(tokens[i+1:j-1])
                        args.append(subexpr)
                        i = j  # at ')' already consumed by loop
                    elif tt.type == "COMMA":
                        i += 1
                        expect_expr = True
                    else:
                        # parse a *single* expression until comma or ')'
                        # We use a small lookahead slice and re-parse with our main parse()
                        j = i
                        depth = 0
                        while j < n:
                            if tokens[j].type == "LPAREN":
                                depth += 1
                            elif tokens[j].type == "RPAREN":
                                if depth == 0:
                                    break
                                depth -= 1
                            elif tokens[j].type == "COMMA" and depth == 0:
                                break
                            j += 1
                        subexpr = parse(tokens[i:j])
                        args.append(subexpr)
                        i = j
                output.append(Call(name=fname, args=args))
                continue
            else:
                push_value(tok)

        elif tok.type in PREC:
            while ops and ops[-1].type in PREC:
                top = ops[-1]
                if (PREC[top.type] > PREC[tok.type]) or (
                    PREC[top.type] == PREC[tok.type] and tok.type not in RIGHT_ASSOC
                ):
                    ops.pop()
                    _to_ast(output, top)
                else:
                    break
            ops.append(tok)

        elif tok.type == "LPAREN":
            ops.append(tok)

        elif tok.type == "RPAREN":
            while ops and ops[-1].type != "LPAREN":
                _to_ast(output, ops.pop())
            if not ops:
                raise ParseError("Mismatched ')'", tok.pos)
            ops.pop()  # pop '('

        elif tok.type == "COMMA":
            # commas are only valid inside function arg parsing handled above
            raise ParseError("Unexpected comma", tok.pos)
        else:
            raise ParseError(f"Unexpected token {tok.type}", tok.pos)

        i += 1

    while ops:
        top = ops.pop()
        if top.type in {"LPAREN", "RPAREN"}:
            raise ParseError("Mismatched parentheses", top.pos)
        _to_ast(output, top)

    if len(output) != 1 or isinstance(output[0], Token):
        raise ParseError("Invalid expression", tokens[-1].pos if tokens else 0)
    return output[0]  # Expr


# ===========================
# ConditionEngine
# ===========================

class ConditionEngine:
    """
    Tiny safe evaluator for your DSL conditions.

    Usage:
        eng = ConditionEngine(contains_pii_hook=lambda text: True/False)
        fn = eng.compile("privacy_score > 0.8 and contains_pii(user_input)")
        result = fn(env_dict)  # True/False
    """

    def __init__(
        self,
        *,
        contains_pii_hook: Optional[Callable[[str], bool]] = None,
        extra_funcs: Optional[Dict[str, Callable[..., Any]]] = None
    ) -> None:
        # built-ins available to expressions
        funcs: Dict[str, Callable[..., Any]] = {
            "len": lambda x: len(x) if x is not None else 0,
            "exists": lambda x: x is not None,
        }

        # domain-specific: contains_pii(user_input)
        if contains_pii_hook is not None:
            def _contains_pii(x: Any) -> bool:
                # if x is not str, try to get 'user_input' from env by name
                if isinstance(x, str):
                    return bool(contains_pii_hook(x))
                return bool(contains_pii_hook(str(x)))
            funcs["contains_pii"] = _contains_pii
        else:
            # default stub: always False
            funcs["contains_pii"] = lambda *_: False

        if extra_funcs:
            funcs.update(extra_funcs)

        self._funcs = funcs

    def compile(self, expression: str) -> Callable[[Dict[str, Any]], bool]:
        tokens = tokenize(expression)
        ast = parse(tokens)

        def _runner(env: Dict[str, Any]) -> bool:
            val = ast.eval(env, self._funcs)
            return bool(val)

        return _runner

    def evaluate(self, expression: str, env: Dict[str, Any]) -> bool:
        """One-shot convenience."""
        return self.compile(expression)(env)


# ===========================
# Quick self-test
# ===========================
if __name__ == "__main__":
    # Demo env – mirror your EvaluationContext fields
    env = {
        "privacy_score": 0.85,
        "session_duration": 42.0,
        "emotional_intensity": 0.3,
        "user_input": "Email me at jane.doe@company.com please!",
    }

    # Very naive PII detector for demo
    email_pat = re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", re.I)
    def contains_pii_hook(text: str) -> bool:
        return bool(email_pat.search(text))

    eng = ConditionEngine(contains_pii_hook=contains_pii_hook)

    tests = [
        "privacy_score > 0.8",
        "session_duration > 60",
        "privacy_score > 0.8 and contains_pii(user_input)",
        "not (emotional_intensity >= 0.9) and privacy_score >= 0.8",
        "exists(user_input) and len(user_input) > 5",
        "privacy_score == 0.85",
        "privacy_score != 0.2 or session_duration < 10",
    ]

    for t in tests:
        print(f"{t:70} -> {eng.evaluate(t, env)}")