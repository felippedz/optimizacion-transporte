import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re

# Validaciones

def limpiar_nombre(nombre):
    """Elimina caracteres no válidos para nombres de archivo."""
    return re.sub(r'[^a-zA-Z0-9_\-]+', '_', nombre)

def validar_funcion(expr):
    """Valida que la función solo tenga valores reales y numéricos."""
    x = sp.symbols('x')
    try:
        f = sp.sympify(expr)
        f_lamb = sp.lambdify(x, f, "numpy")
        test = f_lamb(1.0)
        if np.iscomplexobj(test):
            raise ValueError("La función produce valores complejos.")
        return f
    except Exception as e:
        raise ValueError(f"Error al procesar la función: {e}")

# Aplicación de métodos numéricos

def newton_raphson(f, x0, tol=1e-6, max_iter=1000):
    """Método de Newton-Raphson."""
    x = sp.symbols('x')
    f_prime = sp.diff(f, x)
    f_lamb = sp.lambdify(x, f, "numpy")
    f_prime_lamb = sp.lambdify(x, f_prime, "numpy")

    iteraciones = []
    for i in range(max_iter):
        fx, fpx = f_lamb(x0), f_prime_lamb(x0)
        if np.iscomplex(fx) or np.iscomplex(fpx):
            print("Alerta: valor complejo detectado.")
            break
        if abs(fpx) < 1e-12:
            print(f"Iteración {i}: derivada cercana a 0. Método detenido.")
            break
        x1 = x0 - fx / fpx
        iteraciones.append((i + 1, x0, fx, fpx, abs(x1 - x0)))
        if abs(x1 - x0) < tol:
            return x1, iteraciones
        x0 = x1
    return x0, iteraciones

def serie_taylor(f, grado):
    """Serie de Taylor centrada en 0."""
    x = sp.symbols('x')
    return sp.series(f, x, 0, grado + 1).removeO()

# Calculo de errores

def calcular_errores(f_real, f_aprox, rango=(-5, 5), puntos=50):
    """Calcula errores absolutos y relativos entre f_real y f_aprox."""
    x = sp.symbols('x')
    f_real_l = sp.lambdify(x, f_real, "numpy")
    f_aprox_l = sp.lambdify(x, f_aprox, "numpy")

    xs = np.linspace(rango[0], rango[1], puntos)
    ys_real = f_real_l(xs)
    ys_aprox = f_aprox_l(xs)
    err_abs = np.abs(ys_real - ys_aprox)
    err_rel = np.where(ys_real != 0, err_abs / np.abs(ys_real), 0)

    df = pd.DataFrame({
        "x": xs,
        "f_real(x)": ys_real,
        "f_aprox(x)": ys_aprox,
        "Error absoluto": err_abs,
        "Error relativo": err_rel
    })
    return df

# Graficación

def graficar_funcion(f, resultado=None, criterio=None, f_aprox=None):
    """Grafica f(x) y su aproximación si existe."""
    x = sp.symbols('x')
    f_lamb = sp.lambdify(x, f, "numpy")
    xs = np.linspace(-10, 10, 400)
    ys = f_lamb(xs)

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, label='f(x)', linewidth=2)
    if f_aprox is not None:
        f_aprox_l = sp.lambdify(x, f_aprox, "numpy")
        plt.plot(xs, f_aprox_l(xs), '--', label='Serie de Taylor', color='orange')
    if resultado is not None:
        plt.scatter(resultado, f_lamb(resultado), color='red', label=f"Óptimo ({criterio})")
    plt.title(f"Minimización - {criterio.capitalize()}")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    return plt

# Exportar resultados a Excel

def exportar_proceso(f_str, criterio, iteraciones, df_errores):
    """Guarda en un único Excel el proceso numérico y los errores."""
    os.makedirs("resultados", exist_ok=True)
    nombre = f"resultados/{limpiar_nombre(f_str)}_{criterio}.xlsx"

    with pd.ExcelWriter(nombre, engine='openpyxl') as writer:
        if iteraciones:
            df_iter = pd.DataFrame(iteraciones, columns=["Iteración", "x", "f(x)", "f'(x)", "Δx"])
            df_iter.to_excel(writer, sheet_name="Newton-Raphson", index=False)
        df_errores.to_excel(writer, sheet_name="Errores", index=False)
    print(f"Archivo Excel generado en: {nombre}")

# ------------------ PROGRAMA PRINCIPAL ------------------

criterios = ["costos", "distancia", "combustible", "tiempo", "cantidad_de_entregas"]
print("Seleccione qué desea minimizar:")
for i, c in enumerate(criterios, start=1):
    print(f"{i}. {c.capitalize()}")

try:
    opcion = int(input("Seleccione una opción (1-5): "))
    criterio = criterios[opcion - 1]
except (ValueError, IndexError):
    print("Opción inválida. Finalizando programa.")
    exit()

f_str = input("Ingrese la función a analizar (use 'x' como variable): ")
try:
    f = validar_funcion(f_str)
except ValueError as e:
    print(e)
    exit()

print("\nSeleccione el método a utilizar:")
print("1. Serie de Taylor")
print("2. Newton-Raphson")
print("3. Ambos métodos")
opcion_metodo = input("Opción: ")

resultado, iteraciones, f_taylor = None, [], None

if opcion_metodo == "1":
    grado = int(input("Ingrese el grado de la serie de Taylor: "))
    f_taylor = serie_taylor(f, grado)
    df_errores = calcular_errores(f, f_taylor)
    print(f"\nSerie de Taylor de grado {grado}: {f_taylor}")

elif opcion_metodo == "2":
    x0 = float(input("Ingrese el valor inicial para Newton-Raphson: "))
    resultado, iteraciones = newton_raphson(f, x0)
    df_errores = pd.DataFrame()

elif opcion_metodo == "3":
    grado = int(input("Ingrese el grado de la serie de Taylor: "))
    f_taylor = serie_taylor(f, grado)
    x0 = float(input("Ingrese el valor inicial para Newton-Raphson: "))
    resultado, iteraciones = newton_raphson(f_taylor, x0)
    df_errores = calcular_errores(f, f_taylor)
    print(f"\nSerie de Taylor de grado {grado}: {f_taylor}")

print(f"\nProceso completado. Se minimizó: {criterio} aplicando: {opcion_metodo}")
if resultado is not None:
    print(f"Valor óptimo estimado: x = {resultado}")

# Gráfica
plt_obj = graficar_funcion(f, resultado, criterio, f_taylor)
img_path = f"resultados/{limpiar_nombre(f_str)}_{criterio}.png"
os.makedirs("resultados", exist_ok=True)
plt_obj.savefig(img_path)
plt_obj.close()
print(f"Gráfica guardada en: {img_path}")

# Exportar resultados a Excel
exportar_proceso(f_str, criterio, iteraciones, df_errores)
