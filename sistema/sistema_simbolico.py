"""
Descripción: Sistema de manejo del modelo
Estado: No terminado, estructura inicial
"""
from sistema.Auxiliar_def import *

class EngineSimbolic:
    def __init__(self):
        self.var_abs = {}
        self.max = 0
        self.tokens_special = []

    def code(self, texto):
        text_code, mapa = reemplazar_numeros_por_tokens(texto=texto)
        if len(self.var_abs) == 0:
            self.var_abs = mapa
        else:
            self.var_abs = mapa
            print("Se reemplazaron las variables existenes en var_abs por nuevas")

        self.max = len(self.var_abs)
        return text_code

    def encode(self, texto):
        reemplazar_tokens_por_numeros(texto, self.var_abs)
    
    def add_var(self, var):
        self.var_abs[f'<VAR_{self.max:03d}>'] = var
        self.max += 1
    
    def review(self, texto):
        bloques = extraer_bloques(texto)
