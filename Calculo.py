import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def ingresar_polinomio(grado):
    print("\nIngrese los coeficientes del polinomio desde el término de mayor grado hasta el independiente:")
    coeficientes = []
    for i in range(grado, -1, -1):
        c = float(input(f"Coeficiente para x^{i}: "))
        coeficientes.append(c)
    return coeficientes

def construir_funcion(coef):
    x = sp.Symbol('x')
    f = sum(c * x**i for i, c in enumerate(reversed(coef)))
    return f, x

def metodo_taylor(f, x, punto, orden, valor):
    serie = f.series(x, punto, orden).removeO()
    aproximacion = serie.subs(x, valor)
    print(f"\nSerie de Taylor alrededor de x={punto} hasta orden {orden}:")
    print(serie)
    print(f"\nValor aproximado de f({valor}) ≈ {float(aproximacion)}")
    return float(aproximacion)

def metodo_newton_raphson(f, x, x0, tolerancia=1e-6, max_iter=100):
    df = sp.diff(f, x)
    x_val = x0
    for i in range(max_iter):
        f_val = f.subs(x, x_val)
        df_val = df.subs(x, x_val)
        if df_val == 0:
            print("Derivada cero. No se puede continuar.")
            return None
        x_new = x_val - f_val/df_val
        if abs(x_new - x_val) < tolerancia:
            print(f"\nConvergencia alcanzada en {i+1} iteraciones.")
            return float(x_new)
        x_val = x_new
    print("No se alcanzó la convergencia.")
    return None

def graficar_funcion(f, x, minimo=None):
    f_np = sp.lambdify(x, f, 'numpy')
    x_vals = np.linspace(-10, 10, 400)
    y_vals = f_np(x_vals)
    plt.figure(figsize=(8,5))
    plt.plot(x_vals, y_vals, label='f(x)')
    if minimo is not None:
        plt.scatter(minimo, f_np(minimo), color='red', label=f'Mínimo aproximado (x={minimo:.4f})')
    plt.title("Función de Costo y Punto Crítico")
    plt.xlabel("x (unidades, distancia, etc.)")
    plt.ylabel("Costo")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    print("=== Optimización Numérica: Minimización de Costos ===\n")
    
    grado = int(input("Ingrese el grado del polinomio (ejemplo: 2 para cuadrática): "))
    coef = ingresar_polinomio(grado)
    f, x = construir_funcion(coef)
    
    print(f"\nFunción ingresada: f(x) = {f}")
    print("\nSeleccione el método a utilizar:")
    print("1. Serie de Taylor")
    print("2. Método de Newton-Raphson")
    
    opcion = int(input("\nOpción: "))
    
    if opcion == 1:
        punto = float(input("Punto de expansión (a): "))
        orden = int(input("Orden de la serie: "))
        valor = float(input("Valor de x para evaluar: "))
        metodo_taylor(f, x, punto, orden, valor)
        graficar_funcion(f, x)
        
    elif opcion == 2:
        x0 = float(input("Valor inicial x0: "))
        minimo = metodo_newton_raphson(sp.diff(f, x), x, x0)
        if minimo is not None:
            print(f"\nMínimo aproximado en x = {minimo:.6f}")
            graficar_funcion(f, x, minimo)
    else:
        print("Opción no válida.")

if __name__ == "__main__":
    main()
