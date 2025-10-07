import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re

# ------------------ FUNCIONES BÁSICAS ------------------

def limpiar_nombre(nombre):
    """Elimina caracteres no válidos para nombres de archivo."""
    return re.sub(r'[^a-zA-Z0-9_\-]+', '_', nombre)

def newton_raphson(f, x0, tol=1e-6, max_iter=1000):
    """Método de Newton-Raphson básico con control de división por cero."""
    x = sp.symbols('x')
    f_prime = sp.diff(f, x)
    f_lamb = sp.lambdify(x, f, "numpy")
    f_prime_lamb = sp.lambdify(x, f_prime, "numpy")

    iteraciones = []
    for i in range(max_iter):
        fx, fpx = f_lamb(x0), f_prime_lamb(x0)
        if abs(fpx) < 1e-12:
            break  # evita división por cero

        x1 = x0 - fx / fpx
        iteraciones.append((i + 1, x0, fx, fpx))

        if abs(x1 - x0) < tol:
            return x1, iteraciones
        x0 = x1

    return x0, iteraciones

def serie_taylor(f, grado):
    """Serie de Taylor centrada en 0."""
    x = sp.symbols('x')
    return sp.series(f, x, 0, grado + 1).removeO()

def graficar_funcion(f, resultado=None, criterio=None):
    """Grafica f(x) con el punto óptimo si existe."""
    x = sp.symbols('x')
    f_lamb = sp.lambdify(x, f, "numpy")
    xs = np.linspace(-10, 10, 400)
    ys = f_lamb(xs)

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, label='f(x)')
    if resultado is not None:
        plt.scatter(resultado, f_lamb(resultado), color='red', label=f"Óptimo ({criterio})")
    plt.title(f"Minimización - {criterio}")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    return plt

def exportar_excel(iteraciones, criterio, f_str):
    """Exporta proceso a Excel."""
    df = pd.DataFrame(iteraciones, columns=["Iteración", "x", "f(x)", "f'(x)"])
    os.makedirs("resultados", exist_ok=True)
    nombre = f"resultados/{limpiar_nombre(f_str)}_{criterio}.xlsx"
    df.to_excel(nombre, index=False)
    print(f"Archivo Excel guardado en: {nombre}")

# ------------------ PROGRAMA PRINCIPAL ------------------

criterios = ["costos", "distancia", "combustible", "tiempo", "cantidad_de_entregas"]
print("Seleccione qué desea minimizar:")
for i, c in enumerate(criterios, start=1):
    print(f"{i}. {c.capitalize()}")

opcion = int(input("Seleccione una opción (1-5): "))
criterio = criterios[opcion - 1]

x = sp.symbols('x')
f_str = input("Ingrese la función a analizar (use 'x' como variable): ")
f = sp.sympify(f_str)

print("\nSeleccione el método a utilizar:")
print("1. Serie de Taylor")
print("2. Newton-Raphson")
print("3. Ambos métodos")
opcion_metodo = input("Opción: ")

resultado = None
iteraciones = []

if opcion_metodo == "1":
    grado = int(input("Ingrese el grado de la serie de Taylor: "))
    f_taylor = serie_taylor(f, grado)
    print(f"\nSerie de Taylor de grado {grado}: {f_taylor}")

elif opcion_metodo == "2":
    x0 = float(input("Ingrese el valor inicial para Newton-Raphson: "))
    resultado, iteraciones = newton_raphson(f, x0)
    print(f"\nResultado Newton-Raphson: x ≈ {resultado}")

elif opcion_metodo == "3":
    grado = int(input("Ingrese el grado de la serie de Taylor: "))
    f_taylor = serie_taylor(f, grado)
    x0 = float(input("Ingrese el valor inicial para Newton-Raphson: "))
    resultado, iteraciones = newton_raphson(f_taylor, x0)
    print(f"\nResultado Newton-Raphson sobre la serie de Taylor: x ≈ {resultado}")

print(f"\nProceso completado. Se minimizó: {criterio}")
if resultado is not None:
    print(f"Valor óptimo estimado: x = {resultado}")

# Guardar gráfica y resultados
plt_obj = graficar_funcion(f, resultado, criterio)
os.makedirs("resultados", exist_ok=True)
img_path = f"resultados/{limpiar_nombre(f_str)}_{criterio}.png"
plt_obj.savefig(img_path)
plt_obj.close()
print(f"Gráfica guardada en: {img_path}")

if iteraciones:
    exportar_excel(iteraciones, criterio, f_str)
