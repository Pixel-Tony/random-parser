import typing as T
import typing_extensions as Tx

DIGITS = frozenset('0123456789')
WHITESPACE_CHRS = frozenset('\n\t ')
BIN_OPS = frozenset((
    # arithmetic ops
    '+', '-', '*', '/', '%', '>>', '<<', '|', '&', '^',
    # comparison ops
    '>', '>=', '<', '<=', '==', '!=',
    # logical ops
    '||', '&&',
    # assignment
    '=', '+=', '-=', '*=', '/=', '%=',
    '//=', '>>=', '<<=', '|=', '&=', '^=',
))
LEN3_BIN_OPS = frozenset(a for a in BIN_OPS if len(a) == 3)
BIN_OP_START_CHRS = frozenset(a[0] for a in BIN_OPS)
OP_CHRS = frozenset('+-*/%><|&^=!')
UN_PREF_OPS = frozenset('+-!')
UN_POSTF_OPS = frozenset(('++', '--'))

OPERATORS = BIN_OPS.union(UN_PREF_OPS).union(UN_POSTF_OPS)

OPENING_PARENS = frozenset('[{(')
CLOSING_PARENS = frozenset(']})')
PARENS_MATCH = {
    '(': ')', ')': '(',
    '{': '}', '}': '{',
    '[': ']', ']': '['
}

REASON_UNEXPECTED_CHAR = 1
REASON_BRACKET_NOT_CLOSED = 2
REASON_UNKNOWN_OPERATOR = 3
REASON_BACKSLASH_BUT_SPACE = 4

StateProcessIndicator = T.Callable[['ALS', 'LexerData', str, int], bool]
StateProcessMethod = T.Callable[['ALS', 'LexerData', str, int], None]


class LexingError(Exception):
    pass


class LexingDetonate(Exception):
    __slots__ = ['code']

    def __init__(self, code: int = REASON_UNEXPECTED_CHAR) -> None:
        self.code = code


class LexerData:
    __slots__ = ['result', 'token', 'stack', 'state']

    def __init__(self, state: 'ALS') -> None:
        self.result = []
        self.token = ''
        self.stack = []
        self.state = state


# class Token:
#     __slots__ = ['start', 'end', 'value', 'type']

#     def __init__(self, start: int, end: int, ttype: str, value: str):
#         self.start = start
#         self.end = end
#         self.type = ttype
#         self.value = value

#     def __repr__(self) -> str:
#         return (f"<{self.type.capitalize()}', "
#                 f"({self.start, self.end}), {self.value}>")


class Token:
    __slots__ = ['value']

    def __init__(self, value) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}, value={{{self.value}}}>"


class Literal(Token):
    pass

# # # # # # # # # # # # # |  Lexer state machine  | # # # # # # # # # # # # #


class ALS:
    '''Abstract Lexing State'''
    _insts = {}

    def __new__(cls: type[Tx.Self]) -> Tx.Self:
        return cls._insts.setdefault(cls, object.__new__(cls))

    def process(self, d: LexerData, char: str, char_i: int) -> None:
        raise NotImplementedError

    def at_EOF(self, d: LexerData, char_i: int) -> None:
        if d.token:
            d.result.append(Token(d.token))
        if d.stack:
            raise LexingDetonate(REASON_BRACKET_NOT_CLOSED)

    @T.final
    @staticmethod
    def use(*checks: StateProcessIndicator):
        '''
        Decorator for using default character checks\n
        Pass methods of this class with names starting with 'on',\n
        in order which they should be checked
        '''
        def _(func: StateProcessMethod):
            def _2(self, d: LexerData, char: str, char_i: int):
                if any(ch(self, d, char, char_i) for ch in checks):
                    return
                return func(self, d, char, char_i)
            return _2
        return _

    @T.final
    def record(self, d: LexerData, next_tok: str, state: type[Tx.Self]):
        '''
        Add current token to stream (if it is not empty)
        and set current token to `token` and state to `state`
        '''
        if d.token:
            d.result.append(Token(d.token))
        d.token = next_tok
        d.state = state()

    @T.final
    def on_whitespace(self, d: LexerData, char: str, char_i: int):
        if char in WHITESPACE_CHRS:
            self.record(d, '', DefaultState)
            return True

    @T.final
    def on_op_start(self, d: LexerData, char: str, char_i: int):
        if char in OP_CHRS:
            self.record(d, char, OperatorState)
            return True

    @T.final
    def on_parens(self, d: LexerData, char: str, char_i: int):
        return self.on_opening_parens(d, char, char_i) \
            or self.on_closing_parens(d, char, char_i)

    @T.final
    def on_opening_parens(self, d: LexerData, char: str, char_i: int):
        if char in OPENING_PARENS:
            d.stack.append(char)
            self.record(d, char, DefaultState)
            return True

    @T.final
    def on_closing_parens(self, d: LexerData, char: str, char_i: int):
        if char in CLOSING_PARENS:
            if not len(d.stack) or PARENS_MATCH[char] != d.stack[-1]:
                raise LexingDetonate(REASON_BRACKET_NOT_CLOSED)
            d.stack.pop()
            self.record(d, char, DefaultState)
            return True

    @T.final
    def on_digit(self, d: LexerData, char: str, char_i: int):
        if char in DIGITS:
            self.record(d, char, NumberIntState)
            return True

    @T.final
    def on_backslash(self, d: LexerData, char: str, char_i: int):
        if char == '\\':
            d.token += char
            d.state = BackslashState().store(d.state)
            return True

    @T.final
    def default(self, d: LexerData, char: str, char_i: int):
        if self.on_whitespace(d, char, char_i)          \
                or self.on_digit(d, char, char_i)       \
                or self.on_parens(d, char, char_i)      \
                or self.on_op_start(d, char, char_i)    \
                or self.on_backslash(d, char, char_i):
            return

        raise LexingDetonate


class DefaultState(ALS):
    @T.final
    def process(self, d: 'LexerData', char: str, char_i: int) -> None:
        return self.default(d, char, char_i)


class NumberIntState(ALS):
    @T.final
    @ALS.use(ALS.on_whitespace, ALS.on_closing_parens, ALS.on_op_start,
             ALS.on_backslash)
    def process(self, d: 'LexerData', char: str, char_i: int) -> None:
        if char == '.':
            d.state = NumberFloatState()
        elif char == 'e':
            d.state = NumberPreSciState()
        elif char not in DIGITS:
            raise LexingDetonate

        d.token += char


class NumberFloatState(ALS):
    @T.final
    @ALS.use(ALS.on_whitespace, ALS.on_closing_parens)
    def process(self, d: LexerData, char: str, char_i: int) -> None:
        if char == 'e':
            d.state = NumberPreSciState()
        elif char not in DIGITS:
            raise LexingDetonate
        d.token += char


class NumberPreSciState(ALS):
    @T.final
    def process(self, d: LexerData, char: str, char_i: int) -> None:
        if char not in DIGITS:
            raise LexingDetonate

        d.token += char
        d.state = NumberSciState()

    @T.final
    def at_EOF(self, d: LexerData, char_i: int) -> None:
        raise LexingDetonate


class NumberSciState(ALS):
    @T.final
    @ALS.use(ALS.on_whitespace, ALS.on_opening_parens, ALS.on_op_start)
    def process(self, d: LexerData, char: str, char_i: int) -> None:
        if char in DIGITS:
            d.token += char
            return

        raise LexingDetonate


class OperatorState(ALS):
    # TODO:
    @T.final
    def process(self, d: LexerData, char: str, char_i: int) -> None:
        if char in OP_CHRS:
            d.token += char
            return

        # also captures any operator with length of 1
        if d.token in OPERATORS:
            return self.default(d, char, char_i)

        match len(d.token):
            case 2:
                if d.token in OPERATORS:
                    return self.default(d, char, char_i)
                elif d.token[1] in UN_PREF_OPS and d.token[0] in OPERATORS:
                    d.result.extend(map(Token, d.token))
                    d.token = ''
                    return self.default(d, char, char_i)
                raise LexingDetonate
            case 3:
                if d.token in OPERATORS:
                    self.default(d, char, char_i)
            case 4:
                if d.token[-1] not in UN_PREF_OPS \
                        or d.token[:3] not in LEN3_BIN_OPS:
                    raise LexingDetonate(REASON_UNKNOWN_OPERATOR)
                self.record(d, char, UnaryOperatorState)
            case _:
                raise LexingDetonate(REASON_UNKNOWN_OPERATOR)


class UnaryOperatorState(ALS):
    @T.final
    @ALS.use(ALS.on_whitespace, ALS.on_digit)
    def process(self, d: LexerData, char: str, char_i: int) -> None:
        raise LexingDetonate


class BackslashState(ALS):
    '''Backslash before newline to escape it'''

    _partials = frozenset(('\\', '\\\r'))

    def store(self, prev_state: ALS):
        self._previous_state = prev_state
        return self

    @T.final
    def process(self, d: LexerData, char: str, char_i: int) -> None:
        match char:
            case '\r':
                if d.token[-1] != '\\':
                    raise LexingDetonate
                d.token += char
            case '\n':
                d.token = d.token[:-1] if d.token[-1] == '\\' else d.token[:-2]
                d.state = self._previous_state
            case ' ':
                raise LexingDetonate(REASON_BACKSLASH_BUT_SPACE)
            case _:
                raise LexingDetonate

# # # # # # # # # # # # # # # | Parser itself | # # # # # # # # # # # # # # #


class Lexer:
    def ast_build(self, filename: str):
        with open(filename, 'r', encoding='UTF-8') as file:
            lines = file.readlines()
            content = ''.join(lines)

        self.data = LexerData(DefaultState())
        try:
            for char_i, char in enumerate(content):
                self.data.state.process(self.data, char, char_i)
            self.data.state.at_EOF(self.data, char_i)
        except LexingDetonate as s:
            raise LexingError(self.traceback(
                s.code, char, char_i, lines, filename
            ))

        return self.data.result

    def traceback(self,
                  code: int,
                  char: str,
                  char_i: int,
                  lines: list[str],
                  filename: str
                  ):
        line_i = 0
        while (l := len(lines[line_i])) < char_i:
            char_i -= l
            line_i += 1
        return (
            f"\nAt {filename}:{line_i + 1}:{char_i + 1}:\n"
            + {
                REASON_UNEXPECTED_CHAR:
                    f"Unexpected character '{char}'",
                REASON_BRACKET_NOT_CLOSED: (
                    f"Unexpected '{char}' closing"
                    if len(self.data.stack) == 0
                    else f"Bracket '{self.data.stack[-1]}' not closed"
                ),
                REASON_UNKNOWN_OPERATOR:
                    f"Unknown operator '{self.data.token}'"
            }[code]
        )


if __name__ == '__main__':
    print(Lexer().ast_build('test.txt'))
