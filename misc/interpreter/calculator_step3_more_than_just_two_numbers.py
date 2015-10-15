# encoding: utf-8

INTEGER, PLUS, MINUS, EOF = 'INTEGER', 'PLUS', 'MINUS', 'EOF'

class Token(object):
    def __init__(self, type, value):
        """ type: INTEGER, PLUS, EOF ... """
        self.type = type
        self.value = value

    def __str__(self):
        """ like: Token(INTEGER, 3), Token(PLUS, '+') """
        return 'Token({type},{value})'.format(
            # print out '+', instead of +, by using repr instead of str
            type=self.type, value=repr(self.value))

    def __repr__(self):
        return self.__str__()


class Interpreter(object):
    def __init__(self, text):
        self.text = text   # like 3 + 5
        self.pos = 0
        self.current_token = None
        self.current_char  = self.text[self.pos]

    def error(self):
        raise Exception('Error parsing input')
 
    def advance(self):
        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None
        else:
            self.current_char = self.text[self.pos]

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def integer(self):
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        return int(result)

    ''' Lexical Analyzer (Scanner / Tokenizer) '''
    def get_next_token(self):
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char.isdigit():
                return Token(INTEGER, self.integer())

            if self.current_char == '+':
                self.advance()
                return Token(PLUS, '+')

            if self.current_char == '-':
                self.advance()
                return Token(MINUS, '-')
  
            self.error()
        return Token(EOF, None)

    # expecting the given token_type, if not, raise error
    def eat(self, token_type):
        if self.current_token.type == token_type:
            self.current_token = self.get_next_token()
        else:
            self.error()        

    def term(self):
        token = self.current_token
        self.eat(INTEGER)
        return token.value

    """ Parser / Interpreter """
    def expr(self):
        self.current_token = self.get_next_token()
        result = self.term()
        while self.current_token.type in (PLUS, MINUS):
            token = self.current_token
            if token.type == PLUS:
                self.eat(PLUS)
                result += self.term()
            elif token.type == MINUS:
                self.eat(MINUS)
                result -= self.term()
        return result

def main():
    while True:
        try:
            text = raw_input('calc> ')
        except EOFError:
            break
        
        if not text:
            continue
        interpreter = Interpreter(text)
        result = interpreter.expr()
        print(result)

if __name__ == '__main__':
    main()


