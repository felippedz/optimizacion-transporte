import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# ===============================
# FUNCIONES DE APOYO
# ===============================

def validar_numero(mensaje):
    """Valida que el valor ingresado sea numérico (entero o decimal)."""
    while True:
        try:
            valor = float(input(mensaje))
            return valor
        except ValueError:
            print("Error: Ingrese un valor numérico válido (use punto para decimales).")

def newton_raphson(f, x0, tol=1e-6, max_iter=100):
    """Método de Newton-Raphson con control de división por cero."""
    x = sp.Symbol('x')
    df = sp.diff(f, x)
    f_l = sp.lambdify(x, f, 'numpy')
    df_l = sp.lambdify(x, df, 'numpy')
    iteraciones = []

    x1 = None  # Inicializamos la variable

    for i in range(max_iter):
        fx = f_l(x0)
        dfx = df_l(x0)

        if dfx == 0:
            print(f"Derivada nula detectada en la iteración {i}. No se puede dividir por 0.")
            break

        x1 = x0 - fx / dfx
        iteraciones.append([i + 1, x0, fx, dfx, x1])

        if abs(x1 - x0) < tol:
            print(f"Convergencia alcanzada en {i + 1} iteraciones.")
            break

        x0 = x1

    if x1 is None:
        print("El método no pudo continuar debido a una derivada nula o falta de convergencia.")
        return None, iteraciones

    return x1, iteraciones

def serie_taylor(f, x0, grado):
    """Expande la serie de Taylor de la función f alrededor de x0."""
    x = sp.Symbol('x')
    serie = f.series(x, x0, grado + 1).removeO()
    return serie

def analizar_derivadas(f):
    """Calcula derivadas de primer y segundo orden."""
    x = sp.Symbol('x')
    f1 = sp.diff(f, x)
    f2 = sp.diff(f1, x)
    print("\nDerivadas:")
    print(f"Primera derivada: {f1}")
    print(f"Segunda derivada: {f2}")
    return f1, f2

def graficar_funcion(f, minimo, maximo, x_opt=None, titulo="Gráfico de la función"):
    """Grafica la función y anota los puntos mínimos, máximos y óptimos."""
    x = sp.Symbol('x')
    f_l = sp.lambdify(x, f, 'numpy')
    X = np.linspace(minimo, maximo, 400)
    Y = f_l(X)

    plt.figure(figsize=(9,6))
    plt.plot(X, Y, label='Función original', color='blue')
    plt.title(titulo)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, linestyle='--', alpha=0.6)

    if x_opt is not None:
        y_opt = f_l(x_opt)
        plt.scatter(x_opt, y_opt, color='red', label='Punto óptimo')
        plt.annotate(f'Óptimo ({x_opt:.4f}, {y_opt:.4f})', 
                     xy=(x_opt, y_opt),
                     xytext=(x_opt + 0.5, y_opt + 0.5),
                     arrowprops=dict(arrowstyle="->", color='red'))

    plt.legend()
    plt.show()

def exportar_excel(iteraciones, f, x_opt, archivo="resultado_optimizacion.xlsx"):
    """Exporta el proceso numérico y la gráfica a un archivo Excel."""
    if not iteraciones:
        print("No hay iteraciones disponibles para exportar.")
        return

    df = pd.DataFrame(iteraciones, columns=["Iteración", "x", "f(x)", "f'(x)", "x siguiente"])
    df.loc[len(df.index)] = ["", "", "", "", ""]
    df.loc[len(df.index)] = ["Punto óptimo", x_opt, "", "", ""]

    df.to_excel(archivo, index=False)

    # Guardar la gráfica
    x = sp.Symbol('x')
    f_l = sp.lambdify(x, f, 'numpy')
    X = np.linspace(-10, 10, 400)
    Y = f_l(X)

    plt.figure(figsize=(9,6))
    plt.plot(X, Y, color='blue', label='Función')
    if x_opt is not None:
        plt.scatter(x_opt, f_l(x_opt), color='red', label='Óptimo')
    plt.legend()
    plt.title('Optimización numérica')
    plt.grid(True)
    img_path = os.path.splitext(archivo)[0] + "_grafica.png"
    plt.savefig(img_path, dpi=150)
    plt.close()

    print(f"Archivo Excel exportado con éxito: {archivo}")
    print(f"Gráfica guardada: {img_path}")

# ===============================
# PROGRAMA PRINCIPAL
# ===============================

print("OPTIMIZACIÓN DE COSTOS EN RUTAS DE TRANSPORTE")
print("------------------------------------------------")

x = sp.Symbol('x')
expr_input = input("Ingrese la función de costo en términos de x (use ^ para potencias, ej: 0.5*x^2 + 3*x + 10): ")
expr_input = expr_input.replace("^", "**")

try:
    f = sp.sympify(expr_input)
except Exception as e:
    print("Error en la función ingresada. Verifique la sintaxis.")
    exit()

print("\nSeleccione el método a utilizar:")
print("1) Serie de Taylor")
print("2) Newton-Raphson")
print("3) Derivadas")
print("4) Ambos métodos")

opcion = input("Opción: ")

if opcion == "1":
    x0 = validar_numero("Ingrese el punto de expansión x0: ")
    grado = int(validar_numero("Ingrese el grado de la serie de Taylor: "))
    taylor = serie_taylor(f, x0, grado)
    print(f"\nSerie de Taylor de grado {grado} alrededor de x0={x0}:")
    print(taylor)
    graficar_funcion(f, x0-10, x0+10, x_opt=x0)

elif opcion == "2":
    x0 = validar_numero("Ingrese el valor inicial para Newton-Raphson: ")
    raiz, iteraciones = newton_raphson(f, x0)
    if raiz is not None:
        print(f"\nAproximación encontrada: x ≈ {raiz}")
        graficar_funcion(f, x0-10, x0+10, x_opt=raiz)

elif opcion == "3":
    analizar_derivadas(f)
    graficar_funcion(f, -10, 10)

elif opcion == "4":
    x0 = validar_numero("Ingrese el valor inicial para Newton-Raphson: ")
    grado = int(validar_numero("Ingrese el grado de la serie de Taylor: "))
    taylor = serie_taylor(f, x0, grado)
    raiz, iteraciones = newton_raphson(f, x0)

    print(f"\nSerie de Taylor: {taylor}")
    if raiz is not None:
        print(f"Aproximación encontrada: x ≈ {raiz}")
        graficar_funcion(f, x0-10, x0+10, x_opt=raiz)

else:
    print("Opción no válida.")
    exit()

# ===============================
# OPCIÓN FINAL: EXPORTAR A EXCEL
# ===============================

exportar = input("\n¿Desea exportar el proceso numérico y la gráfica a Excel? (s/n): ").lower()
if exportar == 's':
    if opcion in ['2', '4'] and raiz is not None:
        exportar_excel(iteraciones, f, raiz)
    else:
        print("Solo se exportan resultados del método Newton-Raphson.")
else:
    print("Proceso completado sin exportar.")
