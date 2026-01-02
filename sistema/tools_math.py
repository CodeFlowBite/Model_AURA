"""
Descripción: Sistema de ecuaciones para el sistema simbolico, será utilizado por el modelo para el manejo de ecuaciones

Carcteristicas: 
    Manejo de operaciones basicas: +, -, /, *
    Simplificación de ecuación (No completado)
    Creación de nuevas variables

estado: No terminado, estructura inicial
"""
import re

class EquationMath:

    TOKEN_REGEX = re.compile(
        r"""
        <VAR_\d{3}>      |
        [a-zA-Z]         |
        [+\-*/=()]       |
        """,
        re.VERBOSE
        )

    def __init__(self, equation):
        self.map_var = {}
        self._count_var = 0

        # Dividir ecuación en dos partes con respecto al signo igual
        self.equation = re.split('=', equation)

        self.left_tok = self._build_tok(self.equation[0])
        self.right_tok = self._build_tok(self.equation[1])
    
    def _build_tok(self, text):

        # Obtener tokens
        tokens_init = self.TOKEN_REGEX.findall(text)

        # Limpiar lista de tokens
        tokens_clear = []
        for t in tokens_init:
            if t != '': tokens_clear.append(t)
        
        # Agregar tag a cada token
        tokens = []
        operadores = ['+', '-', '/', '*']
        for t in tokens_clear:
            if t in operadores: tokens.append(['OP', t])
            elif t == '(' or t == ')': tokens.append(['PAR', t])
            elif t[0] == '<': tokens.append(['TOK', t])
            elif re.search(r'[a-z]', t): tokens.append(['VAR', t])
        
        return tokens
    
    def set_value_map(self, value_map: dict):
        self.map_var = value_map
        self._count_var = len(value_map) + 1
    
    def _new_token(self, value):
        var = f'<VAR_{self._count_var:03d}>'
        self.map_var[var] = value
        self._count_var += 1
        return var
    
    # ----- Operaciones aritmeticas -----

    def add_tokens(self, t1, t2):
        v1 = self.map_var[t1]
        v2 = self.map_var[t2]
        return self._new_token(v1 + v2)
    
    def sub_tokens(self, t1, t2):
        v1 = self.map_var[t1]
        v2 = self.map_var[t2]
        return self._new_token(v1 - v2)
    
    def mul_tokens(self, t1, t2):
        v1 = self.map_var[t1]
        v2 = self.map_var[t2]
        return self._new_token(v1 * v2)
    
    def div_tokens(self, t1, t2):
        v1 = self.map_var[t1]
        v2 = self.map_var[t2]
        return self._new_token(v1 / v2)
    
    def eval(self, t1, t2, op):
        if op == '*': return self.mul_tokens(t1, t2)
        if op == '/': return self.div_tokens(t1, t2)
        if op == '-': return self.sub_tokens(t1, t2)
        if op == '+': return self.add_tokens(t1, t2)
    
    # ----- Simplificar -----

    def _simplify_equation(self):
        pass

    def _simplify_par(self, par):
        par_t = par.copy()
        prioridad = [
            {'*', '/'},
            {'+', '-'}
        ]
        for level in prioridad:
            i = 0
            while i < len(par_t):
                if par_t[i][1] in level:
                    # Detectar si se puede aplicar una operación aritmetica
                    print(par_t)
                    print(par_t[i-1][1], par_t[i][1], par_t[i+1][1])
                    if par_t[i-1][0] == 'VAR' or par_t[i+1][0] == 'VAR':
                        # El token estará 'bloqueado' por que esta multiplicando o dividiendo a una variable
                        if par_t[i][1] == '*' or par_t[i][1] == '/':
                            if par_t[i-1][0] == 'TOK': par_t[i-1][0] = 'TOK_A' 
                            if par_t[i+1][0] == 'TOK': par_t[i+1][0] = 'TOK_A' 
                        i += 1
                    elif par_t[i-1][0] == 'PAR' or par_t[i+1][0] == 'PAR':
                        i += 1
                    elif par_t[i-1][0] == 'TOK' and par_t[i+1][0] == 'TOK':
                        result = self.eval(par_t[i-1][1], par_t[i+1][1], par_t[i][1])
                        par_t[i-1: i+2] = [['TOK', result]]
                        i -= 1
                    elif par_t[i-1][0] == 'TOK_A' or par_t[i+1][0] == 'TOK_A':
                        i += 1
                else:
                    print(par_t, "Par")
                    if par_t[i][0] == 'PAR':
                        end = 0
                        for t in range(i, len(par_t)):
                            if par_t[t][0]=='PAR': end = t

                        result = self._simplify_par(par_t[i+1:end])
                        print(result, "Resuñt")
                        par_t[i: end+1] = result
                    i += 1
        return par_t
    
    def __str__(self):
        return f"{self.list_to_str(self.left_tok)} = {self.list_to_str(self.right_tok)}"
    
    def list_to_str(self, lis):
        lis_txt = ""
        for ty, op in lis:
            lis_txt = lis_txt + " " + op
        return lis_txt


ecuacion = "x + <VAR_001> = (<VAR_002> - <VAR_001> * x) / (<VAR_003> + <VAR_004>)"
eq1 = EquationMath(equation=ecuacion)
vars_num = {
    '<VAR_001>': 1,
    '<VAR_002>': 2,
    '<VAR_003>': 4,
    '<VAR_004>': 7
}
eq1.set_value_map(vars_num)
print(eq1.list_to_str(eq1._simplify_par(eq1.right_tok)))