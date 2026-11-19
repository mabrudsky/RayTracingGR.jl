# `propagate`: tolerancias, algoritmos y corte en el borde (2026-11-19)

Resumen de las mediciones y decisiones para la función `propagate` (llamada a `solve`).
Todas las mediciones se hicieron fuera del repo, en la notebook (Intel Core Ultra 7 155H, 22 hilos),
con el escenario del púlsar en Kerr-Schild: $M = 1$, $a = 0.1$, región $5 < r < 50$.
El kernel de GPU se corrió sobre la CPU (`EnsembleGPUKernel(CPU())`); **nada de esto se midió todavía en la A30**.

---

# 1. Qué miden las tablas

Cada rayo es una columna del estado $u = (x^\mu, k^\mu) = (t, x, y, z, k^t, k^x, k^y, k^z)$.
La referencia $u^{\text{ref}}$ es `Vern9` en Float64 con tolerancia $10^{-15}$ (para Float32 se integra en
Float64 a partir de los mismos datos iniciales en Float32).

**Error relativo del estado final** de un rayo:

$$
\delta = \max_{i} \frac{\lvert u_i - u_i^{\text{ref}} \rvert}{\max\left(\lvert u_i^{\text{ref}} \rvert,\, 1\right)}
$$

**Condición nula** al final (debería ser $0$ para un fotón):

$$
\mathcal{N} = \frac{\lvert g_{\mu\nu}\, k^\mu k^\nu \rvert}{(k^t)^2}
$$

**Distancia al borde**, donde $r$ es el radio de Kerr (`coordinate_radius`):

$$
d = \min\left(\lvert r - R_{\text{in}} \rvert,\; \lvert r - R_{\text{out}} \rvert\right)
$$

En las tablas, "mediana" y "p99" son la mediana y el percentil 99 sobre los rayos.

---

# 2. Tolerancias

## 2.1 Por qué cambian respecto de Skylight-Pulsar

En Skylight-Pulsar se usaba $\text{abstol} = 10^{-6}$ y $\text{reltol} = 10^{-9}$, porque el borde no se
interpolaba: el lado derecho se multiplicaba por una condición y el rayo se "congelaba". Ahora el callback
ubica el borde, así que conviene bajar las tolerancias al mínimo que permite la precisión.

Además, en Float32 $\text{reltol} = 10^{-9}$ está por debajo del épsilon de máquina
$\varepsilon_{\text{Float32}} = 2^{-23} \approx 1.2 \times 10^{-7}$, así que no tenía efecto.

## 2.2 Regla elegida

$$
\text{abstol} = \text{reltol} = 5\,\varepsilon_T
\qquad\Longrightarrow\qquad
\begin{cases}
\text{Float64:} & 5 \cdot 2^{-52} \approx 1.1 \times 10^{-15} \\[4pt]
\text{Float32:} & 5 \cdot 2^{-23} \approx 6.0 \times 10^{-7}
\end{cases}
$$

Es el valor medido más chico en el que el error todavía baja y ningún rayo falla.

## 2.3 Medición en Float64 (2000 rayos)

| tol | `Tsit5`: mediana de $\delta$ | `Tsit5`: máx. $\mathcal{N}$ | `Vern9`: mediana de $\delta$ | `Vern9`: máx. $\mathcal{N}$ | tiempo `Tsit5` | tiempo `Vern9` |
|---|---|---|---|---|---|---|
| $10^{-12}$ | $7.0 \times 10^{-13}$ | $3.0 \times 10^{-12}$ | $8.2 \times 10^{-15}$ | $1.3 \times 10^{-13}$ | 0.08 s | 0.05 s |
| $10^{-13}$ | $7.8 \times 10^{-14}$ | $3.5 \times 10^{-13}$ | $4.6 \times 10^{-15}$ | $8.3 \times 10^{-15}$ | 0.10 s | 0.07 s |
| $10^{-14}$ | $1.0 \times 10^{-14}$ | $4.1 \times 10^{-14}$ | $4.3 \times 10^{-15}$ | $4.7 \times 10^{-15}$ | 0.15 s | 0.08 s |
| $10^{-15}$ | $3.4 \times 10^{-15}$ | $1.8 \times 10^{-14}$ | (referencia) | $5.5 \times 10^{-15}$ | 0.23 s | 0.11 s |

- Los 2000 rayos terminan en el borde con `retcode` exitoso en todas las filas.
- `Vern9` llega al piso de redondeo ($\sim 4 \times 10^{-15}$) con tolerancias más grandes y es más rápido
  que `Tsit5` a igual tolerancia. **En CPU (Float64) queda `Vern9()`.**

## 2.4 Medición en Float32 (`GPUTsit5`, 2000 rayos)

| tol | mediana de $\delta$ | máx. $\mathcal{N}$ |
|---|---|---|
| $10^{-5}$ | $5.3 \times 10^{-7}$ | $3.6 \times 10^{-5}$ |
| $10^{-6}$ | $3.7 \times 10^{-7}$ | $3.3 \times 10^{-6}$ |
| $5 \times 10^{-7}$ | $3.7 \times 10^{-7}$ | $2.6 \times 10^{-6}$ |
| $2 \times 10^{-7}$ | $3.6 \times 10^{-7}$ | $1.7 \times 10^{-6}$ |
| $10^{-7}$ | $3.5 \times 10^{-7}$ | $1.7 \times 10^{-6}$ |

La mediana se estanca en $\delta \approx 3.6 \times 10^{-7} \approx 3\,\varepsilon_{\text{Float32}}$:
por debajo de $\sim 5 \times 10^{-7}$ bajar la tolerancia no mejora nada.

---

# 3. Algoritmo en GPU (Float32)

## 3.1 Comparación (5000 rayos, tol $= 6 \times 10^{-7}$)

| | mediana de $\delta$ | p99 de $\delta$ | rayos con $\delta > 10^{-3}$ | máx. $\mathcal{N}$ | tiempo |
|---|---|---|---|---|---|
| `GPUTsit5` | $3.7 \times 10^{-7}$ | $7.0 \times 10^{-6}$ | 2 | $3.4 \times 10^{-6}$ | 0.027 s |
| `GPUVern7` | $3.3 \times 10^{-7}$ | $1.1 \times 10^{-5}$ | 3 | $3.1 \times 10^{-6}$ | 0.090 s |
| `GPUVern9` | $8.6 \times 10^{-7}$ | $7.0 \times 10^{-5}$ | 0 | $8.7 \times 10^{-6}$ | 0.288 s |

Con $a = 0.9$ los resultados son prácticamente iguales.

## 3.2 Por qué el orden alto no ayuda en Float32

Un método de orden $p$ con paso $h$ tiene error de truncamiento local $\sim C\,h^{p+1}$, pero cada paso
también acumula redondeo $\sim \varepsilon_T$ multiplicado por el número de etapas y por el tamaño de los
coeficientes del método. En Float32 el redondeo domina: subir $p$ no baja el error, y los métodos de orden
alto (muchas etapas, coeficientes grandes) acumulan **más** redondeo. Por eso `GPUVern9` es peor.

Los rayos con $\delta > 10^{-3}$ son los mismos pocos con cualquier método: rayos cerca de la órbita
crítica, donde Float32 no alcanza para decidir si el rayo cae o escapa. Solo los arregla Float64.

**Decisión: en GPU queda `GPUTsit5()`** (misma calidad que `GPUVern7` y es el probado en la A30).

## 3.3 `Vern9` en Float32 ubica mal el borde

Con `Vern9` en CPU y Float32 (2000 rayos), el corte queda lejos del borde:

| | mediana de $d$ | máx. $d$ | rayos con $d > 10^{-3}$ |
|---|---|---|---|
| `Vern9` | $3.8 \times 10^{-6}$ | $3.5 \times 10^{-3}$ | 24 |
| `Vern7` | $3.8 \times 10^{-6}$ | $7.5 \times 10^{-4}$ | 0 |
| `Tsit5` | $0$ | $3.4 \times 10^{-5}$ | 0 |

El interpolante de orden 9 pierde precisión por redondeo en Float32. **No afecta**, porque la CPU usa
siempre Float64 (Float32 es solo para la GPU), pero confirma la elección de `GPUTsit5` para Float32.

---

# 4. El corte en el borde (GPU, Float32)

## 4.1 Cómo se corta hoy

La región se describe con la condición

$$
c(x) = (r - R_{\text{in}})(R_{\text{out}} - r),
\qquad
c > 0 \text{ adentro}, \quad c = 0 \text{ en el borde}, \quad c < 0 \text{ afuera}.
$$

Al final de cada paso $[\lambda_{n-1}, \lambda_n]$, si $c < 0$, se busca por bisección el $\lambda^*$ con
$c\big(\tilde u(\lambda^*)\big) = 0$, donde $\tilde u(\lambda)$ es la interpolación del integrador dentro del
paso. El estado final es $\tilde u(\lambda^*)$ y se detiene la integración.

## 4.2 El límite de Float32

En Float32 no todos los números existen: cerca de $r = 50$ el paso mínimo representable es

$$
\operatorname{ulp}(50) = 2^{-18} \approx 3.8 \times 10^{-6}.
$$

Ninguna distancia al borde puede ser menor que eso. Lo mismo vale para $\lambda$: cerca de
$\lambda \approx 50$ la bisección solo puede moverse en saltos de $\sim 3.8 \times 10^{-6}$, y el corte queda
del lado de afuera.

Medido: con la bisección actual, el p99 de $d$ es $1.5 \times 10^{-5}$ y el máximo
$3.4 \times 10^{-5} \approx 9\,\operatorname{ulp}(50)$. La resolución de $\lambda$ explica solo
$\sim 1\,\operatorname{ulp}$; el resto no lo aislé (candidatos: el redondeo al calcular $c$ cerca de la raíz y
el error del interpolante). La corrección de abajo lo resuelve igual, porque evalúa $c$ directamente sobre el
estado.

## 4.3 Corrección propuesta: un paso de Newton sobre el rayo

Después de la bisección, se mueve el estado a lo largo del rayo hasta $c = 0$, sin tocar $\lambda$.
Desarrollando $c$ a primer orden en $\Delta\lambda$:

$$
c\big(x + \Delta\lambda\, k\big) \approx c(x) + \Delta\lambda\, \frac{dc}{d\lambda} = 0
\qquad\Longrightarrow\qquad
\Delta\lambda = -\frac{c(x)}{dc/d\lambda},
\qquad
\frac{dc}{d\lambda} = \partial_\mu c \; k^\mu .
$$

La derivada $\partial_\mu c\, k^\mu$ se obtiene con un número dual de ForwardDiff (igual que `christoffel`):

$$
c(x + \epsilon\, k) = c(x) + \epsilon\, \partial_\mu c\, k^\mu .
$$

Y se actualiza todo el estado (posición y momento) con un paso de Euler:

$$
u \;\leftarrow\; u + \Delta\lambda\, f(u),
$$

donde $f$ es el lado derecho de la ecuación geodésica. Como $\Delta\lambda \sim 10^{-6}$, el error de este
paso es $\mathcal{O}(\Delta\lambda^2) \sim 10^{-12}$, despreciable en Float32.

## 4.4 Resultado (2000 rayos, `GPUTsit5`, tol $= 6 \times 10^{-7}$)

| Variante | p99 de $d$ | máx. $d$ | p99 de $d$ ($r$ en Float64) | máx. $d$ ($r$ en Float64) | p99 de $\delta$ |
|---|---|---|---|---|---|
| Actual (bisección) | $1.5 \times 10^{-5}$ | $3.4 \times 10^{-5}$ | $1.8 \times 10^{-5}$ | $3.5 \times 10^{-5}$ | $7.6 \times 10^{-6}$ |
| Bisección + 1 Newton | $3.8 \times 10^{-6}$ | $7.6 \times 10^{-6}$ | $4.0 \times 10^{-6}$ | $6.6 \times 10^{-6}$ | $7.1 \times 10^{-6}$ |
| Bisección + 2 Newton | $3.8 \times 10^{-6}$ | $7.6 \times 10^{-6}$ | $4.0 \times 10^{-6}$ | $6.7 \times 10^{-6}$ | $7.1 \times 10^{-6}$ |
| **Librería** (bisección + 1 Newton, implementado) | $3.8 \times 10^{-6}$ | $1.1 \times 10^{-5}$ | $4.4 \times 10^{-6}$ | $7.1 \times 10^{-6}$ | $7.1 \times 10^{-6}$ |

"$r$ en Float64" significa: el mismo estado final en Float32, pero calculando $r$ en Float64, para separar el
redondeo del propio cálculo de $r$.

- El corte pasa de $\sim 9\,\operatorname{ulp}$ a $1$–$2\,\operatorname{ulp}$: el límite de Float32.
- Un segundo paso de Newton no agrega nada.
- **El error del estado $\delta$ no cambia** ($\sim 7 \times 10^{-6}$ en el p99): viene de integrar en
  Float32, no del corte. La corrección hace que el rayo se detenga exactamente en el borde, pero la trayectoria
  tiene la misma precisión que antes.
- Costo: una evaluación de $f$ y una de $c$ con un dual, una sola vez por rayo.
- **Falta confirmar en la A30** que no haga escapar al integrador a la memoria de la GPU (las funciones son
  `@inline`, como el resto del callback, y ForwardDiff ya corre dentro del kernel en `christoffel`).

En CPU (Float64) no hace falta: el `ContinuousCallback` ya deja $d < 10^{-10}$.

---

# 5. Estado y decisiones pendientes

## 5.1 Implementado (sin commit; `Pkg.test()` 166/166)

| Archivo | Contenido |
|---|---|
| `src/propagate/Solve/solve.jl` | `propagate(configuration, initial_data, region; alg, kwargs...)`, tolerancia $5\,\varepsilon_T$, opciones por defecto |
| `src/propagate/Solve/solve_cpu.jl` | `Vern9()` por defecto, `EnsembleThreads()` |
| `src/propagate/Solve/solve_gpu.jl` | `GPUTsit5()` por defecto, `EnsembleGPUKernel(backend)`, `batch_size = 50_000`, `merge_callbacks = true` |
| `src/propagate/Callback/boundary_callback_gpu.jl` | Paso de Newton después de la bisección (sección 4.3) |
| `test/propagate/solve_tests.jl` | Casos separados: CPU Float64 con `Vern9`; Float32 con `Tsit5` (versión CPU de `GPUTsit5`) |
| `test/propagate/callback_tests.jl` | El kernel de GPU deja el rayo a $\leq 3\,\operatorname{ulp}$ del borde |

Opciones de `solve` por defecto (se pueden cambiar con argumentos con nombre):
$\text{abstol} = \text{reltol} = 5\,\varepsilon_T$, `adaptive = true`, `save_everystep = false`,
`save_start = false`. Fijas por dentro: `trajectories`, `callback` y, en GPU, `batch_size` y `merge_callbacks`.

## 5.2 Decisiones

1. **Paso de Newton en el callback de GPU**: implementado. Falta confirmarlo en la A30.
2. **Float32 en CPU**: no se prohíbe (la CPU se usa con Float64 y Float32 es para la GPU, pero usar Float32
   en CPU no rompe nada). Los tests se separan por caso.
