"""Punto 2. Inferencia exacta por enumeración."""
from algorithms.probability import normalize


def enumeration_ask(network, query: str, evidence: dict) -> dict:
    """Calcula P(query | evidence) sobre la conjunta factorizada.

    Entradas: network es la red; query es el nombre de la
    variable consultada; evidence es {nombre de variable: 0 o 1}.
    Salida: {0: probabilidad, 1: probabilidad}.

    Tips:
    - Recorra network.variables en orden topológico.
    - Para una variable fijada multiplique su probabilidad; para una oculta
      sume las dos alternativas 0 y 1. Normalice al terminar.
    - Use copias del evento y no modifique evidence.
    - network.probability(variable, valor, evento) consulta la CPT;
      el evento debe contener los valores de todos los padres.
    - Puede usar una función auxiliar recursiva. Al terminar el recorrido,
      el producto sin factores restantes vale 1.
    - Evidencia imposible: ValueError. Consulta observada: distribución puntual
      solo si la evidencia es posible.
    """
    # Valida nombres de variables y dominio de la evidencia.
    network.validate_query(query, evidence)


    def enumerate_all(variables, event):
        # caso base, se procesaron todas las variables
        if not variables:
            return 1.0

        # coger la siguiente variable en orden topologico
        variable = variables[0]
        rest = variables[1:]

        #caso 1: la variable tiene un valor conocido
        if variable in event:
            probability = network.probability(variable, event[variable], event)
            return probability * enumerate_all(rest, event)

        # caso 2: variable oculta, considerar 0 y 1
        total = 0.0
        for value in (0, 1):
            new_event = {**event, variable: value}
            probability = network.probability(variable, value, new_event)
            total += probability * enumerate_all(rest, new_event)

        return total

    # calculo de los pesos para cada valor de query
    weights = {}

    for value in (0, 1):
        if query in evidence and evidence[query] != value:
            weights[value] = 0.0
        else:
            event = {**evidence, query: value}
            weights[value] = enumerate_all(network.variables, event)

    # reutilizar la normalizacion del punto 1
    return normalize(weights)

    # FIN DEL BLOQUE DEL ESTUDIANTE
