"""
Descripción: Funciones auxiliares para el sistema
"""

import re

def reemplazar_numeros_por_tokens(texto):
    token_por_valor = {}
    valor_por_token = {}
    contador = 1

    def reemplazo(match):
        nonlocal contador
        valor_str = match.group()

        # Convertimos a número real (int o float)
        if '.' in valor_str:
            valor = float(valor_str)
        else:
            valor = int(valor_str)

        if valor not in token_por_valor:
            token = f"<VAR_{contador:03d}>"
            token_por_valor[valor] = token
            valor_por_token[token] = valor
            contador += 1

        return token_por_valor[valor]

    # Soporta: enteros, decimales y negativos
    patron = r'(?<!\w)-?\d+(?:\.\d+)?\b'

    texto_codificado = re.sub(patron, reemplazo, texto)
    return texto_codificado, valor_por_token

def reemplazar_tokens_por_numeros(texto, mapa):
    """
    texto: str con tokens tipo <VAR_001>
    mapa: dict -> { "<VAR_001>": 4, "<VAR_002>": -3.5 }
    """

    def reemplazo(match):
        token = match.group()
        if token not in mapa:
            raise ValueError(f"Token no encontrado en el mapeo: {token}")
        return str(mapa[token])

    patron = r'<VAR_\d{3}>'
    return re.sub(patron, reemplazo, texto)

def parse_state_block(state_text):
    """
    Parsea el contenido interno de <STATE>...</STATE>
    """
    resultado = {}

    # STEP
    step_match = re.search(r'STEP\s+(\d+)', state_text)
    if step_match:
        resultado["STEP"] = int(step_match.group(1))

    # Equation
    eq_match = re.search(r'Equation:\s*(.+)', state_text)
    if eq_match:
        resultado["Equation"] = eq_match.group(1).strip()

    # Goal
    goal_match = re.search(r'Goal:\s*(.+)', state_text)
    if goal_match:
        resultado["Goal"] = goal_match.group(1).strip()

    return resultado

def extraer_bloques(texto):
    """
    Extrae y parsea STATE, ACTION y TAG
    """
    resultado = {
        "INTENT": [],
        "STATE": [],
        "ACTION": [],
        "TAG": []
    }

    # --- INTENT ---
    intent = re.findall(r'<INTENT>\s*(.*?)\s*</INTENT>', texto, re.DOTALL)
    resultado["INTENT"] = [i.strip() for i in intent]
    if len(resultado["INTENT"]) == 0: resultado["INTENT"] = None

    # --- STATE ---
    estados = re.findall(r'<STATE>\s*(.*?)\s*</STATE>', texto, re.DOTALL)
    for estado in estados:
        subblock = parse_state_block(estado)
        if len(subblock)==0: subblock = None
        resultado["STATE"].append(subblock)
    if len(resultado["STATE"]) == 0: resultado["STATE"] = None

    # --- ACTION ---
    acciones = re.findall(r'<ACTION>\s*(.*?)\s*</ACTION>', texto, re.DOTALL)
    resultado["ACTION"] = [a.strip() for a in acciones]
    if len(resultado["ACTION"]) == 0: resultado["ACTION"] = None

    # --- TAG ---
    tags = re.findall(r'<TAG>\s*(.*?)\s*</TAG>', texto, re.DOTALL)
    resultado["TAG"] = [t.strip() for t in tags]
    if len(resultado["TAG"]) == 0: resultado["TAG"] = None

    if resultado["ACTION"] == None and resultado["INTENT"] == None and resultado["STATE"] == None and resultado["TAG"] == None:
        resultado = None

    return resultado


texto = """

<INTENT> solve_operation </INTENT>

<STATE>
STEP 2
Equation: (<VAR_001> * x) + <VAR_002> = <VAR_003>
Goal: isolate x
</STATE>

<TAG> REMOVE_ADDITION </TAG>

<HALT_GENERATION/>
"""

resultado = extraer_bloques(texto)
print(resultado)
