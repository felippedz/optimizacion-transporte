import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =========================================================
# 1️⃣ Entrada de datos
# =========================================================
x = sp.Symbol('x')
funcion_str = input("Ingrese la función (en términos de x): ")  # Ejemplo: sin(x), exp(x), log(1+x), 2*x**2+x+8
a = float(input("Ingrese el punto de expansión a: "))            # Ejemplo: 0
grado = int(input("Ingrese el grado de la serie de Taylor: "))   # Ejemplo: 4
rango_min = float(input("Ingrese el límite inferior del rango: "))  # Ejemplo: -2
rango_max = float(input("Ingrese el límite superior del rango: "))  # Ejemplo: 2
num_puntos = 25  # cantidad de puntos de comparación

# =========================================================
# 2️⃣ Serie de Taylor simbólica y derivadas
# =========================================================
try:
    f = sp.sympify(funcion_str)
except Exception as e:
    print(f"❌ Error al interpretar la función: {e}")
    exit()

# Cálculo de derivadas y coeficientes
derivadas = []
coeficientes = []
for n in range(grado + 1):
    deriv = sp.diff(f, x, n)
    deriv_a = deriv.subs(x, a)
    coef = deriv_a / sp.factorial(n)
    derivadas.append(deriv)
    coeficientes.append(coef)

# Serie de Taylor completa
taylor = sum([coeficientes[n] * (x - a)**n for n in range(grado + 1)])
print(f"\nSerie de Taylor de grado {grado}:")
print(taylor.simplify())

# =========================================================
# 3️⃣ Evaluación numérica y comparación analítica
# =========================================================
f_num = sp.lambdify(x, f, 'numpy')
taylor_num = sp.lambdify(x, taylor, 'numpy')

x_vals = np.linspace(rango_min, rango_max, num_puntos)
f_vals, taylor_vals, errores_abs, errores_rel = [], [], [], []

for xv in x_vals:
    try:
        real = f_num(xv)
        aprox = taylor_num(xv)
        err_abs = abs(real - aprox)
        err_rel = abs(err_abs / real) if real != 0 else 0
    except Exception:
        real, aprox, err_abs, err_rel = np.nan, np.nan, np.nan, np.nan
    f_vals.append(real)
    taylor_vals.append(aprox)
    errores_abs.append(err_abs)
    errores_rel.append(err_rel)

# =========================================================
# 4️⃣ Interpolación polinómica de Lagrange
# =========================================================
try:
    puntos_x = np.linspace(rango_min, rango_max, 5)
    puntos_y = f_num(puntos_x)
    L_poly = sp.interpolate(list(zip(puntos_x, puntos_y)), x)
    L_num = sp.lambdify(x, L_poly, 'numpy')
    interp_vals = L_num(x_vals)
except Exception as e:
    print(f"⚠ Error en interpolación: {e}")
    interp_vals = [np.nan]*len(x_vals)
    L_poly = sp.nan

# =========================================================
# 5️⃣ Exportar a Excel (todo el proceso)
# =========================================================
# --- Hoja 1: Derivadas y coeficientes ---
datos_derivadas = {
    'Orden n': list(range(grado + 1)),
    'Derivada f⁽ⁿ⁾(x)': [str(d) for d in derivadas],
    f'f⁽ⁿ⁾({a})': [str(d.subs(x, a)) for d in derivadas],
    'Coeficiente (f⁽ⁿ⁾(a)/n!)': [str(c) for c in coeficientes]
}
df_derivadas = pd.DataFrame(datos_derivadas)

# --- Hoja 2: Evaluación numérica ---
df_eval = pd.DataFrame({
    'x': x_vals,
    'f(x) Real': f_vals,
    'Tn(x) Aproximación': taylor_vals,
    'Interpolación Lagrange': interp_vals,
    'Error Absoluto': errores_abs,
    'Error Relativo (%)': [e*100 for e in errores_rel]
})

# --- Hoja 3: Resumen ---
resumen = {
    'Función original': [funcion_str],
    'Punto de expansión a': [a],
    'Grado de Taylor': [grado],
    'Serie de Taylor': [str(taylor.simplify())],
    'Polinomio de Lagrange': [str(L_poly)]
}
df_resumen = pd.DataFrame(resumen)

# Exportar todas las hojas al mismo archivo
with pd.ExcelWriter('procedimiento_taylor_completo.xlsx', engine='openpyxl') as writer:
    df_resumen.to_excel(writer, sheet_name='Resumen', index=False)
    df_derivadas.to_excel(writer, sheet_name='Derivadas y Coeficientes', index=False)
    df_eval.to_excel(writer, sheet_name='Comparación Numérica', index=False)

print("\n✅ Todo el procedimiento fue exportado a 'procedimiento_taylor_completo.xlsx'")

# =========================================================
# 6️⃣ Representación gráfica
# =========================================================
plt.figure(figsize=(10,6))
plt.plot(x_vals, f_vals, label='Función Real', linewidth=2)
plt.plot(x_vals, taylor_vals, '--', label=f'Serie de Taylor (grado {grado})', linewidth=2)
plt.plot(x_vals, interp_vals, ':', label='Interpolación Lagrange', linewidth=2)
plt.title(f'Comparación Analítica - {funcion_str}', fontsize=13)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

# =========================================================
# 7️⃣ Resumen final en consola
# =========================================================
print("\n=== RESUMEN DE ERRORES ===")
print(df_eval[['x', 'Error Absoluto', 'Error Relativo (%)']].head(10))
print("\nProceso completado con éxito ✅")
