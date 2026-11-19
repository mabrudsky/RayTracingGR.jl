# Tareas pendientes — RayTracingGR.jl

Registro de trabajo para continuar entre sesiones.
Detalle y código propuesto de cada punto: [REVISION_2026-11-12.md](REVISION_2026-11-12.md) (el número `#N` remite a esa revisión).

Reglas de trabajo:
- No ejecutar nada sin autorización explícita; por defecto solo revisión y sugerencias.
- Respetar la estructura actual; solo corregir inconsistencias y bugs.
- GPU: todo isbits, sin alocaciones, SArray, mínima RAM, máximo uso de caché.

Estado: `[ ]` pendiente · `[~]` en curso · `[x]` hecho

---

## Revisión 2026-11-12 (orden sugerido)

<details>
<summary><b>Bloque 1: Bugs ✅ completo (2026-11-12 / 2026-11-13)</b></summary>

**#1 — `KerrSchildCoordinates(1.0, 0.5)` siempre daba error** (commit `fcd4edc`)

Causa: antes había solo dos formas de crear el objeto.

| Llamada | Qué hacía |
|---|---|
| `KerrSchildCoordinates(M=1.0, a=0.5)` (con nombres) | ✅ creaba el objeto |
| `KerrSchildCoordinates(M, a)` (sin nombres, con cualquier número) | ❌ mostraba un error |

La segunda estaba pensada para rechazar valores de tipo inválido (por ejemplo `Int`), pero como era la
única forma sin nombres, también rechazaba `KerrSchildCoordinates(1.0, 0.5)`, que es válida.

| Llamada | Antes | Ahora |
|---|---|---|
| `KerrSchildCoordinates(M=1.0, a=0.5)` | ✅ | ✅ |
| `KerrSchildCoordinates(1.0, 0.5)` | ❌ error (bug) | ✅ |
| `KerrSchildCoordinates(1, 0.5)` | ❌ error | ❌ error (correcto) |

| Archivo | Cambio |
|---|---|
| `src/spacetime/Kerr/KerrSchildCoordinates.jl` | Sin `@with_kw`: constructor interno `{T}(M, a)` con validación por `throw`; métodos posicional `(M::T, a::T)`, keywords `(; M, a)` y fallback `(::Real, ::Real)` con error claro. Se conserva `topology`. |

Uso: `KerrSchildCoordinates(M=1.0, a=0.5)` y `KerrSchildCoordinates(1.0, 0.5)`.

**#2 — `Configuration` no se podía construir** (commit `cf75af8`)

| Archivo | Cambio |
|---|---|
| `src/RayTracing_types.jl` | Se eliminó `AbstractPhysicalModel`. |
| `src/configuration/config_types.jl` | Se eliminó el campo `model` de `Configuration` e `IntegrationParameters`. |
| `src/propagate/ensemble.jl` | `p = IntegrationParameters(spacetime)`; firmas sin el parámetro `M`. |

Decisión: la física la define el tipo de ecuación (`GeodesicEquationsVacuum`). Si una ecuación futura tiene parámetros (p. ej. plasma `ωp`), van dentro de su struct.

**#3 — Un sistema de coordenadas era un subtipo de espacio-tiempo** (commit `cf75af8`)

| Archivo | Cambio |
|---|---|
| `src/spacetime/spacetime_types.jl` | `AbstractCoordinate <: AbstractRayTracing` (antes `<: AbstractSpacetime`). |

Decisión: se conserva el campo `topology` (futuros sistemas de coordenadas de Kerr y callbacks).

**#4 — El largo de `u0` no estaba ligado a la ecuación** (commit `edd944e`)

| Archivo | Cambio |
|---|---|
| `src/propagate/equations.jl` | Firmas `SVector{8,T}` (geodésica) y `SVector{16,T}` (transporte paralelo). |
| `src/propagate/ensemble.jl` | `state_length(eq)` (8 / 16) + chequeo en `Ensemble_Problem`. |

**#5 — Mezcla silenciosa de Float32 / Float64** (commit `9abbe00`)

| Archivo | Cambio |
|---|---|
| `src/spacetime/spacetime.jl` | Declaración `function numtype end`. |
| `src/spacetime/Kerr/Kerr.jl` | `numtype` agregado al `import`. |
| `src/spacetime/Kerr/KerrSchildCoordinates.jl` | `numtype(::KerrSchildCoordinates{T}) where {T} = T`. |
| `src/propagate/ensemble.jl` | Chequeo `numtype(spacetime) === T` en `Ensemble_Problem`. |

**Qué detecta ahora `Ensemble_Problem` antes de integrar**

| Error de configuración | Mensaje |
|---|---|
| Largo de `u0` distinto al de la ecuación | `DimensionMismatch: ParallelTransportEquationsVacuum needs u0 of length 16, got 8` |
| Precisión distinta entre el espacio-tiempo y `u0` | `ArgumentError: spacetime uses Float64 but u0 uses Float32; use the same precision ...` |
| Filas de `initial_data` ≠ largo de `u0` | `DimensionMismatch` (ya existía) |

**Verificación final del bloque** (test existente 3/3 + chequeo manual no agregado al repo)

| Caso | Resultado |
|---|---|
| `KerrSchildCoordinates(1.0, 0.5)` / `(1, 0.5)` | ✅ `KerrSchildCoordinates{Float64}` / `ArgumentError` claro |
| `Configuration` + `Ensemble_Problem` (Float64 y Float32) | ✅ `EnsembleProblem` |
| Coordenada como `spacetime` | ❌ `MethodError` al construir `Configuration` |
| Transporte paralelo con `u0` de 8 | ❌ `DimensionMismatch` claro |
| Espacio-tiempo Float64 + `u0` Float32 (y viceversa) | ❌ `ArgumentError` claro |
| Lado derecho Float32 | ✅ resultado Float32, 0 alocaciones (148 ns) |

</details>

<details>
<summary><b>Bloque 2: Dependencias ✅ completo (2026-11-13)</b></summary>

**#12. `DifferentialEquations` → `SciMLBase`** (commit `024b515`)

La librería solo usa `ODEProblem`, `EnsembleProblem` y `remake`, que están en `SciMLBase`.
`DifferentialEquations` instalaba además todos los solvers de CPU (`OrdinaryDiffEq` y otros), que no se
usan: los solvers de GPU vienen en `DiffEqGPU`.

| Archivo | Cambio |
|---|---|
| `Project.toml` | `SciMLBase` (compat `3.50`) en lugar de `DifferentialEquations`. |
| `src/RayTracingGR.jl` | `import DifferentialEquations as DE` → `import SciMLBase`. |
| `src/propagate/ensemble.jl` | `DE.ODEProblem`, `DE.remake`, `DE.EnsembleProblem` → `SciMLBase.…` |

**#13. `BenchmarkTools` solo en tests** (commits `024b515` y `d702886`)

| Archivo | Cambio |
|---|---|
| `Project.toml` | `BenchmarkTools` pasa de `[deps]` a `[extras]`; `[targets] test = ["Test", "BenchmarkTools"]`. |

Los tests ahora se corren con `Pkg.test()`:

| Comando | Antes | Ahora |
|---|---|---|
| `julia --project=. -e 'using Pkg; Pkg.test()'` | ✅ | ✅ |
| `julia --project=. test/runtests.jl` | ✅ | ❌ no encuentra `BenchmarkTools` |

**#14. Quitar `Parameters` y `import KernelAbstractions as KA`** (commit `024b515`)

| Archivo | Cambio |
|---|---|
| `Project.toml` | Se quitó `Parameters`. |
| `src/RayTracingGR.jl` | Sin `import Parameters: @with_kw`; `using KernelAbstractions` → `import KernelAbstractions as KA`. |
| `src/spacetime/Kerr/Kerr.jl` | Se borró `using Parameters` (sin uso desde el #1). |
| `src/configuration/config_types.jl` | `@with_kw` → `Base.@kwdef`; `B <: KA.Backend`. |
| `src/propagate/ensemble.jl` | `B <: KA.Backend` en las tres firmas. |

`using KernelAbstractions` traía todos los nombres del paquete (`CPU`, `@kernel`, `synchronize`, ...)
dentro de la librería, aunque solo se usa el tipo `Backend`. Con `import ... as KA` no se trae ningún
nombre suelto y se escribe `KA.Backend`, que es el mismo tipo. El backend se sigue eligiendo igual
desde el script del usuario:

```julia
using RayTracingGR
using KernelAbstractions   # CPU()   — o: using CUDA → CUDABackend()

cfg = RayTracingGR.Configuration(spacetime = st, equations = RayTracingGR.GeodesicEquationsVacuum(),
                                 backend = CPU(), u0 = u0, tspan = (0.0, 1.0))
```

Único cambio visible: si falta un campo al construir `Configuration`, el error ahora es
`UndefKeywordError: keyword argument 'u0' not assigned`.

</details>

<details>
<summary><b>Bloque 3: Estructura ✅ completo (2026-11-13)</b></summary>

- [x] #9 Unificar los dos `_Ensemble_Problem` con una función `rhs(...)` (2026-11-13)

  | Archivo | Cambio |
  |---|---|
  | `src/propagate/ensemble.jl` | `rhs(::GeodesicEquationsVacuum)` / `rhs(::ParallelTransportEquationsVacuum)`; un solo `Ensemble_Problem` con `ODEProblem{false}(rhs(equations), ...)`. Se borraron los dos `_Ensemble_Problem`. |

  Decisión del usuario: los tipos concretos de ecuaciones se quedan en `ensemble.jl`; `propagate_types.jl`
  es solo para subtipos abstractos de propagación (hoy vacío: `AbstractEquationSet` tiene que seguir en
  `RayTracing_types.jl` porque `Configuration` lo usa y se carga antes). Ecuación nueva = `struct` +
  `state_length` + `rhs` en `ensemble.jl`.
- [x] #10 Renombrar el submódulo `kerr` → `Kerr` y limpiar imports y exports redundantes (2026-11-13)

  | Archivo | Cambio |
  |---|---|
  | `src/spacetime/Kerr/Kerr.jl` | `module kerr` → `module Kerr`; se quitaron 6 imports sin uso (`AbstractSpacetime`, `AbstractCoordinate`, `SphericalCoordinate`, `AbstractChristoffel`, `ChristoffelTag`, `_Γcomp`) y `export metric, metric_inverse, christoffel`. |
  | `src/spacetime/spacetime.jl` | Se borró `using .kerr`. |
  | `src/RayTracingGR.jl` | `export kerr` → `export Kerr`. |
  | `test/…/christoffel_tests_KerrSchildCoordinates.jl`, `README.md`, `CONTRIBUTING.md` | `kerr.` → `Kerr.`; paso 3 de "Agregar un nuevo espacio-tiempo" corregido. |

  Decisión del usuario: `KerrSchildCoordinates` **no** se exporta (a propósito). Se usa siempre con el
  nombre de la familia adelante: `Kerr.KerrSchildCoordinates(M = 1.0, a = 0.9)`. Los submódulos de
  espacio-tiempo van en CamelCase y no exportan nada.
- [x] #11 Fallback genérico de `christoffel`; eliminar `AbstractChristoffel` (2026-11-13)

  | Archivo | Cambio |
  |---|---|
  | `src/spacetime/christoffel.jl` | `christoffel(spacetime::AbstractSpacetime, position::SVector{4,T})` = `_christoffel_ForwardDiff(...)` (sirve para cualquier espacio-tiempo); `struct ChristoffelTag end` sin `AbstractChristoffel`. |
  | `src/spacetime/Kerr/KerrSchildCoordinates.jl` | Se borró el método `christoffel` de Kerr (lo cubre el genérico). |
  | `src/spacetime/Kerr/Kerr.jl` | Ya no importa `christoffel` ni `_christoffel_ForwardDiff`. |
  | `README.md`, `CONTRIBUTING.md` | Jerarquía de tipos, árbol de carpetas y pasos para agregar un espacio-tiempo (ya no hace falta escribir `christoffel`). |

  Decisión del usuario: el fallback `metric_inverse = inv(metric)` **queda pendiente** hasta que haya
  métricas numéricas sin expresión analítica (ver tarea agregada abajo).

</details>

<details>
<summary><b>Bloque 4: GPU ✅ completo (2026-11-13)</b></summary>

- [-] #6 ~~Contracción directa de Γ sin armar el SArray completo~~ — **DESCARTADO por legibilidad** (2026-11-13)

  Se probó (prototipo fuera del repo) calcular Γ^λ_{μν}k^μk^ν dentro de la ecuación sin armar Γ:
  mismo resultado (dif. 3e-17 en Float64), 0 alocaciones y más rápido en CPU (geodésica 88.5 → 63.2 ns
  en Float64, 96.3 → 53.0 ns en Float32; transporte 143 → 130 ns). Decisión del usuario: la versión
  actual (`christoffel` + producto punto a punto) es mucho más legible para quien no conoce Julia, y se
  mantiene. `christoffel.jl` es solo para Christoffel. Retomar solo si la velocidad en GPU lo exige.
- [x] #7 `ntuple(f, 4)` → `ntuple(f, Val(4))` (2026-11-13)

  | Archivo | Cambio |
  |---|---|
  | `src/spacetime/christoffel.jl` | En `_christoffel_ForwardDiff` (versión activa), los dos `ntuple` usan `Val(4)`, con el comentario `# Val(4): fixed size known at compile time (unrolled loops, required for GPU kernels)`. El bloque comentado de abajo no se tocó. |

  Sin cambio medible en CPU (`christoffel` 65 ns, 0 alocaciones, tipo inferido `SArray{Tuple{4,4,4},T}` en
  Float64 y Float32): la ganancia es la garantía de tamaño fijo al compilar en kernels de GPU.
- [x] #8 Parte 2: `initial_data` tiene que estar en la CPU (2026-11-13)

  | Archivo | Cambio |
  |---|---|
  | `src/propagate/ensemble.jl` | Chequeo en `Ensemble_Problem`: `KA.get_backend(initial_data) isa KA.CPU`, si no `ArgumentError` claro. |

  Motivo: DiffEqGPU 3.20.1 corre `prob_func` en la CPU (`solve.jl`, línea 199) y lee una columna de
  `initial_data` por rayo; un arreglo de GPU daría lectura escalar (lentísima o error). Se usa
  `get_backend` en lugar de `isa Array` para aceptar arreglos de CPU como `view` o `transpose`.
  La parte 1 (`save_everystep = false`, `save_start = false`) pasó a la tarea del backend/`solve`
  (más abajo), porque la librería todavía no llama a `solve`.

</details>

<details>
<summary><b>Bloque 5: Tests y orden</b></summary>

- [x] #15 Benchmarks a `benchmark/`; tests de isbits, cero alocaciones y Float32 (2026-11-13)

  | Archivo | Cambio |
  |---|---|
  | `test/metric/christoffel_tests_KerrSchildCoordinates.jl` | Sin `@btime` (se mantienen los `println` de \|∇g\|); nuevo testset Float32 (atol `1e-5`, máx. medido ~1e-7). |
  | `test/metric/kerr_schild_tests.jl` (nuevo) | Constructores y errores (`Int`, precisión mixta, `M < 0`, `\|a\| > M`); isbits y 0 alocaciones de `metric`, `metric_inverse`, `christoffel` (Float64/Float32). |
  | `test/propagate/ensemble_tests.jl` (nuevo) | Casos válidos e inválidos de `Ensemble_Problem` (los 5 chequeos + coordenada como spacetime); isbits de `IntegrationParameters`; 0 alocaciones en las dos ecuaciones. GPU simulado con un arreglo cuyo `get_backend` no es CPU. |
  | `test/runtests.jl` | Incluye los tres archivos. |
  | `benchmark/benchmarks.jl`, `benchmark/Project.toml` (nuevos) | Los `@btime` de antes, con su entorno (`[sources]` apunta a `..`). |
  | `Project.toml` | `BenchmarkTools` fuera de `[extras]`, `[targets]` y `[compat]`. |
  | `.gitignore` | `/benchmark/Manifest*.toml`. |
  | `README.md`, `CONTRIBUTING.md` | Árbol de carpetas; tabla de tests y sección "Benchmarks". |

  Tests: 3 → 54.
- [x] #16 Varios (2026-11-16: hechas las partes 2, 3 y 4; 2026-11-17: parte 1 reordenada, commit `65730b6`)
  - [x] Parte 1: callback de terminación. Funciona en CPU y en GPU (A30, 10⁶ rayos); carpeta `Callback/`
    reordenada el 2026-11-17 (región y backends CPU/GPU en archivos separados; ver la tarea más abajo).
    Consideraciones del usuario (2026-11-16): (1) cómo detectar el borde depende del sistema de coordenadas;
    (2) depende del escenario (púlsar, disco, binaria): primero el púlsar de Skylight-Pulsar; (3) en CPU,
    `ContinuousCallback` + `terminate!`; en GPU, si no anda, buscar otra estrategia isbits que reproduzca el
    resultado, guardando solo el estado final; congelar el lado derecho (como Skylight-Pulsar) es la
    **última opción**; (4) estructura intuitiva y modular, y el código no depende del sentido de
    integración (lo deciden las etapas previa y posterior; en GPU $\lambda$ tiene que crecer, así que se
    invierte $k^\mu$).
    - [x] `coordinate_radius(spacetime, point)`: un método en cada archivo de métrica (Kerr-Schild: radio de
      Kerr $r$; Boyer-Lindquist: `point[2]`). Declarado en `spacetime.jl` y exportado.
    - [x] Experimento (fuera del repo): púlsar en Kerr-Schild ($a = 0.1$, $5 < r < 50$, 400 rayos).
      CPU `Tsit5` y kernel `GPUTsit5` en `EnsembleGPUKernel(CPU())`, cada uno con `ContinuousCallback` +
      `terminate!` y con `DiscreteCallback` + interpolación a mano (bisección sobre `integrator(t)` +
      `change_t_via_interpolation!`): los cuatro cortan los 400 rayos en el borde correcto (38 en $R_\text{in}$,
      362 en $R_\text{out}$), a 2e-14 del borde en Float64 y 3e-5 en Float32; CPU contra kernel 4e-10
      (Float64) / 3e-5 (Float32). Igual con DiffEqGPU 3.20.1 y 3.21.6. Sin alocación extra por callback.
    - [x] Primera implementación (2026-11-16), `ContinuousCallback` + `terminate!` para CPU y GPU
      (después, la versión de GPU se reemplazó por `DiscreteCallback` + bisección; ver abajo):

      | Archivo | Cambio |
      |---|---|
      | `src/propagate/propagate_types.jl` | `abstract type AbstractRegion <: AbstractPropagate end` |
      | `src/propagate/Callback/callback.jl` | `PulsarRegion{T}(R_in, R_out)` (isbits, constructores como las métricas); `region_condition` = $(r - R_\text{in})(R_\text{out} - r)$ con `coordinate_radius`; `boundary_callback(region)` → `ContinuousCallback(condition, terminate!)` (luego elegido según el backend, ver abajo), lee el espacio-tiempo de `integrator.p`. Reemplaza el código viejo comentado. |
      | `src/RayTracingGR.jl` | Exporta `PulsarRegion`, `boundary_callback`. |
      | `test/propagate/callback_tests.jl` | Constructores, isbits (también de la condición del callback), condición en KS y BL, y 40 rayos integrados con `Tsit5` y con `GPUTsit5` en `EnsembleGPUKernel(CPU())`, Float64 y Float32. |

      Uso: `cb = boundary_callback(region, configuration)`; GPU: `solve(ep, GPUTsit5(), EnsembleGPUKernel(backend); callback = cb, merge_callbacks = true, save_everystep = false, ...)`; CPU: `Tsit5()` y `EnsembleThreads()` (sin `merge_callbacks`).
    - [x] Primera prueba en la A30 (2026-11-16, DiffEqGPU 3.21.6, CUDA.jl 6): **fallan las dos**,
      `ContinuousCallback` y la alternativa `DiscreteCallback` + bisección del script, con
      "Out of dynamic GPU memory (trying to allocate 664 bytes)".
      Causa (verificada): el integrador de la GPU es un `struct` mutable; si se pasa a una función que no se
      inlinea, escapa al heap de la GPU (664 B por hilo). DiffEqGPU lo arregló con `@inline` en el PR #207
      (dic. 2022) "when not using Continuous callbacks"; para `ContinuousCallback` sigue abierto el #219
      (el PR #221 que lo intentaba se cerró sin integrar; los tests oficiales usan 2 trayectorias). En el
      script, `affect!` e `inside` no eran `@inline`.
      Medición con el kernel en CPU (bytes alocados por rayo; 636 B es la base de todas las variantes):
      Continuous 1452, Continuous con condición `@inline` 1372, Discrete + bisección sin `@inline` 1308,
      **Discrete + bisección todo `@inline` 636** (sin escape). `DiscreteCallback` + `terminate!` solo
      (como en 2023) no escapa en GPU pero corta un paso después del borde (9/2000 en el borde).
    - [x] `boundary_callback` elige el callback según el backend (decisión del usuario: automático):
      `boundary_callback(region, configuration)` o `boundary_callback(region, backend)`;
      `KA.CPU` → `ContinuousCallback` + `terminate!`; `KA.GPU` → `DiscreteCallback` + bisección (50 pasos)
      sobre `integrator(t)` + `change_t_via_interpolation!` + `terminate!`, todo `@inline`.
      Tests: tipo de callback según backend, isbits de condición y `affect!`, rayos en el borde con el
      kernel sobre la CPU (Float64/Float32).
    - [x] Confirmado en la A30 (2026-11-16, DiffEqGPU 3.21.6, CUDA.jl 6.4.2, Float32, tol 1e-6) con
      `test/gpu/callback_gpu.jl` (`boundary_callback(region, CUDABackend())`):

      | Prueba | Resultado |
      |---|---|
      | ¿El código de GPU pide memoria mientras corre? | No (`no allocations inside the kernel`) |
      | 10 000 rayos | 871 en la estrella, 9129 escapan, 0 sin cortar, error máx. 3.8e-5, 0.009 s |
      | 100 000 rayos | 8893 / 91 107 / 0, error máx. 4.6e-5, 0.065 s |
      | 1 000 000 rayos | 89 154 / 910 846 / 0, error máx. 4.6e-5, 0.809 s |
      | Comparación con la CPU (10 000 rayos) | mismos 871 / 9129; los 10 000 en el mismo borde; dif. relativa máx. 1.0e-4 |

      Nota: la sesión de Jupyter del cluster ve solo 0.73 GiB de la A30 (probablemente una porción MIG);
      alcanzó para 10⁶ rayos. Advertencias de CUDA por `LD_LIBRARY_PATH` con `/usr/local/cuda`: no afectan.
  - [x] Parte 2: se borró `src/Document/Chat_Copilot/analisis_raytracing.md` (notas de Copilot desactualizadas; quedan en el historial de git).
  - [x] Parte 3: se borró `src/configuration/utils_config_path.jl` (todo comentado, de otro proyecto) y su `include` en `configuration.jl`.
  - [x] Parte 4: en `_christoffel_ForwardDiff` el argumento `k` pasó a llamarse `spacetime` (`k` es el momento del fotón en el resto del código). El bloque comentado de abajo no se tocó.

</details>

<details>
<summary><b>Bloque 6: Tests de validación (diferenciación automática de Christoffel y métricas)</b></summary>

Objetivo: el test de $\nabla g = 0$ se hizo para verificar que la diferenciación automática de
Christoffel funcione y que las métricas estén bien escritas. Pero $\nabla g = 0$ es una identidad
algebraica: solo prueba $g\,g^{-1} = I$ y las convenciones de índices. Estos tests cubren lo que falta.
Detalle en `REVISION_2026-11-12.md` §17–21. Agregados el 2026-11-12, sin fecha límite.

- [ ] #17 **Diferenciación automática vs diferencias finitas** (objetivo: AD). Comparar `christoffel`
  con el $\Gamma$ armado a partir de derivadas por diferencias finitas centradas,
  $\partial_\sigma g_{\mu\nu} \approx \dfrac{g_{\mu\nu}(x + h\,e_\sigma) - g_{\mu\nu}(x - h\,e_\sigma)}{2h}$,
  con $h \sim 10^{-6}$ (por la escala) y tolerancia relativa $\sim 10^{-7}$ en Float64. Es la prueba
  independiente de ForwardDiff.
- [ ] #18 **Vector nulo de Kerr-Schild** (objetivo: métrica). Comprobar
  $\eta^{\mu\nu} l_\mu l_\nu = 0$, $g^{\mu\nu} l_\mu l_\nu = 0$ y $g_{\mu\nu} l^\mu l^\nu = 0$, con
  $l^\mu = \eta^{\mu\nu} l_\nu$. Valida `_H_l`.
- [ ] #19 **Límite $a = 0$** (objetivo: métrica). Comparar `metric` con Schwarzschild en Kerr-Schild
  escrito a mano: $2H = \dfrac{2M}{r}$, $l_\mu = \left(1, \dfrac{x}{r}, \dfrac{y}{r}, \dfrac{z}{r}\right)$.
- [ ] #20 **Ricci $= 0$ (vacío)** (objetivo: métrica + AD). Comprobar

  $$
  R_{\mu\nu} = \partial_\lambda \Gamma^\lambda{}_{\mu\nu} - \partial_\nu \Gamma^\lambda{}_{\mu\lambda}
  + \Gamma^\lambda{}_{\lambda\sigma} \Gamma^\sigma{}_{\mu\nu} - \Gamma^\lambda{}_{\nu\sigma} \Gamma^\sigma{}_{\mu\lambda} \approx 0
  $$

  derivando `christoffel` con Duales anidados (prueba también la AD de segundo orden) o con
  diferencias finitas. Es la prueba más fuerte de que la métrica es solución de las ecuaciones de Einstein.
- [ ] #21 **Constantes de movimiento** a lo largo de una geodésica integrada (cadena completa:
  métrica → $\Gamma$ → ecuaciones → solver). Comprobar $g_{\mu\nu} k^\mu k^\nu = 0$, $E = -k_t$,
  $L_z = x\,k_y - y\,k_x$ (índices bajos) y la constante de Carter. Depende de tener la propagación
  (`Ensemble_Problem` + `solve`) funcionando.

**Notas**
- Para todos: evaluar varios puntos, incluyendo dentro del horizonte ($r < r_+$), cerca del eje y en
  Float32 (con tolerancias más laxas).
- #21: la constante de Carter **no** es genérica (sale del tensor de Killing de Kerr), así que va como
  método específico de `AbstractKerrSpacetime`.

</details>

---

## Tareas agregadas por el usuario

<!-- Formato: - [ ] (YYYY-MM-DD, para: YYYY-MM-DD o "sin fecha") descripción -->

- [x] (2026-11-16, para: 2026-11-17; hecho 2026-11-17) **Reordenar toda la carpeta `src/propagate/Callback/`**. Hecho:
  `callback.jl` (índice: `function region_condition end`, `function boundary_callback end`, includes, método con
  `Configuration`), `Region/PulsarRegion.jl`, `boundary_callback_cpu.jl`, `boundary_callback_gpu.jl`.
  `AbstractRegion` queda en `propagate_types.jl`; regiones sin `module`; tests sin cambios. Texto original:
  Los callbacks
  funcionan (CPU con `ContinuousCallback`, GPU con `DiscreteCallback` + bisección; probado en la A30 con
  10⁶ rayos), pero según el usuario la implementación **no es modular y no respeta la estructura de la
  librería**. Hoy todo está en un solo archivo, `callback.jl`: la región `PulsarRegion` (struct y
  constructores), `region_condition` y las dos versiones de `boundary_callback` (CPU y GPU). Solo
  `AbstractRegion` está en `propagate_types.jl`. Antes de mover nada: proponer la estructura nueva
  siguiendo el patrón del resto de la librería (por ejemplo, cómo está organizado `spacetime/`) y
  esperar el OK del usuario. Mantener los tests (`test/propagate/callback_tests.jl`) y el script de GPU
  (`test/gpu/callback_gpu.jl`) funcionando.

### Funciones físicas genéricas — A PEDIDO (2026-11-13)
**Principio:** cada magnitud (frecuencias, energía, lapse/shift, tétradas, curvatura, callbacks...) se
implementa siguiendo el **patrón de `christoffel`**: función genérica de `(spacetime, x)` que solo usa
`metric`, `metric_inverse` y derivadas por Duales; sirve para cualquier métrica.
**No se planifica por adelantado:** el usuario pide cada función cuando la necesita.
Descartado: jerarquía de simetrías, Killing a mano, descubrimiento de Killing, `lapse2`.
Recordatorio: el horizonte de eventos no es local (no encaja en el patrón); para callbacks usar
criterios locales (p. ej. Kretschmann / frecuencia para captura, radio de coordenadas para escape).

---

- [x] (2026-11-13, para: sin fecha; hecho 2026-11-19, commit `d4d50c6`) **Usar `configuration.backend` para elegir el integrador** (CPU / GPU). Hecho con `propagate` en `src/propagate/Solve/` (mismo patrón que `Callback/`). Detalle y mediciones en [PROPAGATE_2026-11-19.md](PROPAGATE_2026-11-19.md).

  | Archivo | Cambio |
  |---|---|
  | `src/propagate/Solve/solve.jl` (nuevo) | `propagate(configuration, initial_data, region; alg, kwargs...)` → matriz `N × n` (column-major). Declara `default_algorithm` y `solve_ensemble`. Tolerancia `default_tolerance(T) = 5 eps(T)`. Opciones por defecto (se pueden cambiar): `abstol`, `reltol`, `adaptive = true`, `save_everystep = false`, `save_start = false`. Fijas: `trajectories`, `callback` (y en GPU `batch_size`, `merge_callbacks`): cambiarlas da `ArgumentError`. |
  | `src/propagate/Solve/solve_cpu.jl` (nuevo) | `Vern9()` por defecto, `EnsembleThreads()`. Algoritmo de GPU con `CPU()` → `ArgumentError`. |
  | `src/propagate/Solve/solve_gpu.jl` (nuevo) | `GPUTsit5()` por defecto, `EnsembleGPUKernel(backend)`, `batch_size = 50_000` (como Skylight-Pulsar), `merge_callbacks = true`. |
  | `src/propagate/Callback/boundary_callback_gpu.jl` | Paso de Newton después de la bisección: `Δλ = -c / (∂_μ c k^μ)` (dual de ForwardDiff), `u ← u + Δλ f(u)`. Corte en Float32: de ~9 ulp a 1–3 ulp. |
  | `src/RayTracingGR.jl` | Imports `Vern9`, `GPUTsit5`, `EnsembleGPUKernel`, `GPUODEAlgorithm`, `derivative`; `export propagate`. |
  | `test/propagate/solve_tests.jl` (nuevo) | Valores por defecto; CPU Float64 con `Vern9`; Float32 con `Tsit5` (versión CPU de `GPUTsit5`); errores. |
  | `test/propagate/callback_tests.jl` | El kernel de GPU deja el rayo a ≤ 3 ulp del borde. |

  Decisiones del usuario (2026-11-19): el algoritmo se pasa como el de SciML/DiffEqGPU (`alg = Tsit5()`), sin tipos propios; CPU usa Float64 y Float32 es solo para GPU, pero no se prohíbe Float32 en CPU (los tests se separan por caso); `batch_size` solo en GPU y fijo; el registro de la corrida (método, tolerancias) queda en la aplicación. Tests: 166/166.
- [x] (2026-11-19, para: sin fecha; hecho 2026-11-19) **Confirmar en la A30 el paso de Newton del callback de GPU** con `test/gpu/propagate_gpu.jl`. Resultado en la A30 (DiffEqGPU 3.21.6, CUDA.jl 6.4.2, Julia 1.12.6, Float32): sin memoria pedida dentro del kernel; 10⁴ / 10⁵ / 10⁶ rayos con 0 sin cortar, 1 ulp (p99) y 3 ulp (máx.) del borde, 0.91 s con 10⁶; contra CPU Float64 (10⁴ rayos): todos en el mismo borde, dif. relativa mediana 3.1e-7 y p99 8.7e-6. Texto original: que no haya alocaciones dentro del kernel (escape del integrador) y que el corte quede a 1–3 ulp. Probado solo con el kernel sobre la CPU. Script listo: `test/gpu/propagate_gpu.jl` (memoria en el kernel, corridas de 10⁴ a 10⁶ rayos con `propagate`, corte en ulp con y sin Newton, comparación con CPU Float64). En la notebook sin GPU: todo OK (2 y 3 ulp como máximo con 2000 y 20 000 rayos; sin Newton, 10 ulp). Correr en el cluster (el script activa su entorno `test/gpu/`): `julia -t auto test/gpu/propagate_gpu.jl` desde la raíz, o `julia -t auto propagate_gpu.jl` desde `test/gpu/`.

  Texto original de la tarea:

  | Backend | Integradores | Ensemble | Paquete |
  |---|---|---|---|
  | `CPU()` | `Tsit5`, `Vern7`, `Vern9` | `EnsembleThreads()` | `OrdinaryDiffEqTsit5`, `OrdinaryDiffEqVerner` (agregados 2026-11-13) |
  | GPU (`CUDABackend()`, ...) | `GPUTsit5`, `GPUVern7`, `GPUVern9` | `EnsembleGPUKernel(backend)` | `DiffEqGPU` |

  Idea GPU: `solve(ep, GPUTsit5(), EnsembleGPUKernel(configuration.backend); trajectories = n, save_everystep = false, save_start = false, dt = ...)`.
  Parte 1 del #8: sin `save_everystep = false` DiffEqGPU reserva `pasos × rayos` en la GPU (`lowerlevel_solve.jl`, líneas 83-85; p. ej. 10⁶ rayos × 1000 pasos × 8 Float32 = 32 GB, contra 32 MB solo con el estado final); `output_func` filtra recién después de integrar. Falta decidir cómo se elige el método y `dt`.

  Medición de referencia (2026-11-13, Intel Core Ultra 7 155H, 22 hilos, Kerr a = 0.9, `dt = 0.01` fijo, 2000 rayos × 1000 pasos), ns por paso y por rayo:

  | | `Tsit5` 1 hilo | `Tsit5` + `EnsembleThreads()` | `GPUTsit5` + `EnsembleGPUKernel(CPU())` |
  |---|---|---|---|
  | Geodésica Float64 | 666 | 122 | 510 |
  | Geodésica Float32 | 620 | 108 | 620 |
  | Transporte Float64 | 1007 | 214 | 858 |
  | Transporte Float32 | 824 | 153 | 801 |

- [ ] (2026-11-13, para: sin fecha) **Sección "Cómo citar" del README**: está oculta en un comentario HTML (`<!-- ... -->`). `CITATION.bib` ya corregido (versión `v0.1.0`, email `@gmail.com`); falta decidir cuándo mostrarla (descomentar).

- [ ] (2026-11-13, para: sin fecha) **Fallback genérico de `metric_inverse`** (`inv(metric(st, x))` para `AbstractSpacetime`), cuando se implementen métricas numéricas sin expresión analítica de la inversa. Postergado del #11. Ojo: si una métrica analítica se olvida su inversa, el fallback la cubre en silencio (más lento, menos preciso en Float32).

### Propuestas en evaluación (no confirmadas)
- `Configuration`: `tspan::Tuple{T,T}` no convierte tipos; `(0.0, 100)` da un `MethodError`. Hay que
  decidir entre documentarlo o agregar una conversión.

---

## Historial de sesiones

- 2026-11-12: Revisión completa (solo lectura). Creados `dev/REVISION_2026-11-12.md` y `dev/TAREAS.md`.
- 2026-11-12: Aplicado #1 en `src/spacetime/Kerr/KerrSchildCoordinates.jl` (constructores explícitos: `{T}(M,a)` con validación por `throw`, posicional `(M::T, a::T)`, keywords `(; M, a)`, fallback `(::Real, ::Real)` con error claro). Se mantiene `KerrSchildCoordinates(M=1.0, a=0.5)`. Test existente OK con `julia --project=. test/runtests.jl`: 3/3 pass, |∇g| máx ~1.7e-16, 0 alocaciones. Sin cambios en tests.
- 2026-11-12: README reescrito (estructura, diagramas de carpetas/includes/flujo/tipos, convenciones, sección Pendientes). Al cambiar tareas, mantener sincronizada la sección "Pendientes" del README con este archivo.
- 2026-11-12: Agregado Bloque 6 (#17–#21, tests de validación de AD y métricas) a pedido del usuario. Solo registrado, no implementado.
- 2026-11-12: Commit `fcd4edc` en la rama `revision-2026-11-12` (constructor #1, README, dev/). Mergeado a `main` (fast-forward).
- 2026-11-12: Push a GitHub desde WSL (`928dc13`).
- 2026-11-13: CI → job Documentation fallaba desde 2026-10-19 (tests OK en 1.10/1.12/pre). Causa: nombre viejo `RayTracing` tras renombrar a `RayTracingGR`. Corregido en docs/Project.toml, docs/make.jl (incl. `deploydocs` repo), docs/src/index.md, CI.yml (doctests), badges del README y CITATION.bib. Sin build local de docs (no autorizado). Pendiente tras el push: verificar CI y activar GitHub Pages (Settings → Pages → `gh-pages`). Manifests locales con `[[deps.RayTracing]]` (ignorados por git) → regenerar.
- 2026-11-13: Commit `a5816ba` + push (WSL). CI: job Documentation ✅ (primera vez desde 2026-10-19); docs publicadas en https://mabrudsky.github.io/RayTracingGR.jl/dev/ (GitHub Pages ya estaba activo). Manifests locales regenerados: raíz (entrada huérfana eliminada, versiones sin cambios) y docs/ (`Pkg.rm` por UUID + `Pkg.develop`). El PR #5 de Dependabot (Documenter 1.20) debería pasar ahora tras rebase.
- 2026-11-13: #2 resuelto con opción A (propuesta del usuario: `GeodesicEquationsVacuum` ya indica vacío, `model` era redundante). Cambios en `src/RayTracing_types.jl`, `src/configuration/config_types.jl`, `src/propagate/ensemble.jl`. Test 3/3 OK; chequeo manual (no agregado al repo): `Configuration` y `Ensemble_Problem` se construyen; `IntegrationParameters` isbits, 16 bytes. README y revisión actualizados.
- 2026-11-13: #3 redefinido: solo reubicar `AbstractCoordinate`; `topology` se queda (futuros sistemas de coordenadas + callbacks).
- 2026-11-13: Discutido diseño "todo genérico desde la métrica". Decisión final: patrón `christoffel` para cada magnitud, implementado a pedido; se descartó el Bloque 7 planificado (#22–#26).
- 2026-11-13: #3 aplicado en `src/spacetime/spacetime_types.jl`. Test 3/3 OK; chequeo manual (no agregado al repo): `CartesianCoordinate() isa AbstractSpacetime == false`, `KerrSchildCoordinates` sigue isbits con `topology`, `Configuration`/`Ensemble_Problem` OK, y `Configuration(spacetime = CartesianCoordinate(), ...)` ahora da `MethodError`. Commit de #2 + #3 + documentación.
- 2026-11-13: Commit `cf75af8` (#2, #3, docs) — sin push.
- 2026-11-13: #4 aplicado. `equations.jl`: firmas `SVector{8,T}` / `SVector{16,T}`. `ensemble.jl`: `state_length` por ecuación + chequeo en `Ensemble_Problem`. Test 3/3 OK. Chequeo manual (no agregado al repo): combinaciones correctas → `EnsembleProblem`; Parallel+8 y Geodesic+16 → `DimensionMismatch` claro; llamada directa con largo incorrecto → `MethodError`. RHS sin alocaciones (`@btime`: geodésica 139 ns, transporte paralelo 234 ns, 0 bytes). Nota: medir `@allocated` en scope global da falsos positivos (80/144 B); medir dentro de función o con `@btime $x`.
- 2026-11-13: Push de `cf75af8` y `edd944e` (WSL).
- 2026-11-13: #5 aplicado (`numtype` + chequeo en `Ensemble_Problem`). Verificación completa de #1–#5 OK (ver tabla en Bloque 1). Bloque 1 reemplazado por resumen en tablas (aquí y en README). Bloque 1 completo.
- 2026-11-13: README: convenciones en LaTeX (métrica de Kerr-Schild completa), causa del #1 explicada con tablas, guía "Agregar un nuevo espacio-tiempo" incluye `numtype`. CI de `edd944e` ✅.
- 2026-11-13: #12 y #13 aplicados. `Pkg.resolve()` (Manifest local sin `DifferentialEquations`/`OrdinaryDiffEq`; SciMLBase 3.50.2). `Pkg.test()` 3/3 OK, 0 alocaciones. Chequeo manual: `Ensemble_Problem` → `SciMLBase.EnsembleProblem`; `DifferentialEquations` no se carga. Falta #14.
- 2026-11-13: #14 aplicado. `Pkg.resolve()` + `Pkg.test()` 3/3 OK, 0 alocaciones. Chequeo manual: `KA.Backend === KernelAbstractions.Backend`; `Configuration` por keywords con `CPU()` OK (Float64 geodésica y Float32 transporte paralelo → `EnsembleProblem`); backend inválido → `MethodError`; campo faltante → `UndefKeywordError`; `IntegrationParameters` isbits. `Parameters` ya no se carga. Bloque 2 completo.
- 2026-11-13: README: Bloque 2 documentado con tablas (como el Bloque 1). Nuevo estilo fijado por el usuario: `#` secciones, `##` subsecciones, bloques de Pendientes como desplegables `<details>` con título en negrita. Commit del Bloque 2 + push (WSL). Pendiente menor: la sección Tests del README todavía muestra `test/runtests.jl` (ya no funciona tras #13).
- 2026-11-13: README sección Tests: se quitó `julia --project=. test/runtests.jl` (no funciona tras #13); queda solo `Pkg.test()`.
- 2026-11-13: Reorganización de la documentación: README para usuarios (instalación, uso, convenciones, principios, estructura del proyecto con sus diagramas, citar, licencia); `CONTRIBUTING.md` nuevo con agregar espacio-tiempo y tests; los Bloques (desplegables) y las propuestas pasaron de README a este archivo, que ahora es el único lugar de pendientes (ya no hay que sincronizar con el README). `dev/` se queda en el repositorio público.
- 2026-11-13: Sección "Cómo citar" oculta en el README (comentario HTML). Email corregido a `@gmail.com` en `CITATION.bib`, `LICENSE`, `Project.toml` y `docs/make.jl`; versión de `CITATION.bib` → `v0.1.0`. Commit + push.
- 2026-11-13: #9 aplicado (`rhs` + un solo `Ensemble_Problem`; tipos concretos quedan en `ensemble.jl`). `Pkg.test()` 3/3 OK. Chequeo manual: `@inferred Ensemble_Problem` OK para las dos ecuaciones, `prob.f.f` es la función correcta, `prob_func` arma `SVector{N}`; largo incorrecto → `DimensionMismatch`.
- 2026-11-13: #10 aplicado (`kerr` → `Kerr`, imports/exports limpios, `KerrSchildCoordinates` sin exportar a propósito). `Pkg.test()` 3/3 OK. Chequeo manual: `Kerr` exportado, `KerrSchildCoordinates` no queda suelto, `kerr` ya no existe, `metric`/`christoffel` funcionan con `Kerr.KerrSchildCoordinates`.
- 2026-11-13: #11 aplicado (fallback `christoffel` para `AbstractSpacetime`, se borra el método de Kerr, `ChristoffelTag` sin abstracto). `metric_inverse` genérico postergado (tarea nueva). `Pkg.test()` 3/3 OK. Chequeo manual: Kerr usa el método genérico, Γ idéntico a `_christoffel_ForwardDiff` en Float64 y Float32, 0 alocaciones en `christoffel` y en el RHS de la geodésica. Nota de medición: `@allocated` con un alias global no constante (`R = RayTracingGR`) da 192 B falsos; usar `const`. Bloque 3 completo.
- 2026-11-13: #6 descartado por legibilidad (medición registrada en el Bloque 4).
- 2026-11-13: #7 aplicado (`Val(4)` + comentario en `_christoffel_ForwardDiff`). `Pkg.test()` 3/3 OK; tipo inferido correcto y 0 alocaciones en Float64/Float32.
- 2026-11-13: #8 parte 2 aplicada (chequeo `KA.get_backend(initial_data) isa KA.CPU`). `Pkg.test()` 3/3 OK. Chequeo manual: `Matrix`, `view` y `transpose` aceptados; arreglo con backend GPU (simulado) → `ArgumentError`. Parte 1 movida a la tarea del backend/`solve`. Bloque 4 completo.
- 2026-11-13: Bloque 6 pasado a desplegable junto a los Bloques 1–5, con notación LaTeX; se quitó la tarea ya cumplida "Continuar con el Bloque 2".
- 2026-11-13: #15 aplicado. `Pkg.test()` 54/54 OK (antes 3). Benchmark: `julia --project=benchmark benchmark/benchmarks.jl` da los mismos tiempos (christoffel ~66 ns, 0 alocaciones).
- 2026-11-13: Dependencias `OrdinaryDiffEqTsit5` (compat `2.1`, instalada 2.1.5) y `OrdinaryDiffEqVerner` (compat `2.4`, instalada 2.4.2) agregadas (opción A: mismos métodos en CPU y GPU). Todavía sin código que las use (tarea del `solve`). Medido el costo por paso (tabla en esa tarea). `Pkg.test()` 54/54 OK.
- 2026-11-16: #16 partes 2, 3 y 4 aplicadas (borrados `src/Document/` y `utils_config_path.jl`; `k` → `spacetime` en `_christoffel_ForwardDiff`). README actualizado (árbol y diagrama). Parte 1 (callback) en espera de las consideraciones del usuario.
- 2026-11-16: Métrica de Kerr en Boyer-Lindquist (pedido del usuario): `src/spacetime/Kerr/BoyerLindquistCoordinates.jl` con struct `BoyerLindquistCoordinates{T}` (topology `SphericalCoordinate`), misma estructura que Kerr-Schild (constructores, `numtype`, auxiliar `_Σ_Δ`, `metric` y `metric_inverse` componente a componente). Tests nuevos: `christoffel_tests_BoyerLindquistCoordinates.jl` (∇g = 0, tolerancia relativa a max|g| porque g_φφ ~ r²) y `boyer_lindquist_tests.jl` (constructores, isbits, 0 alocaciones, g·g⁻¹ = I, det g = −Σ² sin²θ, igual a Kerr-Schild transformada con error ~1e-15, límite a = 0 = Schwarzschild). `LinearAlgebra` agregado a los paquetes de test. `Pkg.test()` 96/96 OK. README actualizado (estado, uso, convenciones, fórmula, árbol, diagrama, jerarquía).
- 2026-11-16: `coordinate_radius(spacetime, point)` agregado en cada métrica (Kerr-Schild: radio de Kerr r; Boyer-Lindquist: point[2]), declarado en spacetime.jl y exportado. Tests: eje, ecuador, punto general, mismo r en KS y BL, 0 alocaciones. `Pkg.test()` 110/110 OK. Experimento de callbacks registrado en #16 parte 1.
- 2026-11-16: Corte del púlsar implementado con `ContinuousCallback` + `terminate!` (`PulsarTermination`, `termination_condition`, `termination_callback`), igual para CPU y GPU. `Pkg.test()` 138/138 OK (incluye integración en CPU y en el kernel de GPU sobre la CPU). Falta probar en la A30.
- 2026-11-16: Nombres del corte cambiados a pedido del usuario: `AbstractTermination` → `AbstractRegion`, `PulsarTermination` → `PulsarRegion`, `termination_condition` → `region_condition`, `termination_callback` → `boundary_callback`. Uso: `boundary_callback(PulsarRegion(R_in = 5f0, R_out = 50f0))`.
- 2026-11-16: DiffEqGPU actualizado a 3.21.6 (compat `3.21`); `Pkg.test()` 138/138 OK. Creado `test/gpu/` (Project.toml con CUDA + `callback_gpu.jl`) para probar `boundary_callback` en la A30: alocaciones en el kernel, escala de rayos, comparación con CPU y alternativa Discrete + interp. Probado en CPU. README y CONTRIBUTING actualizados.
- 2026-11-16: A30: fallaron ContinuousCallback y la alternativa Discrete del script ("Out of dynamic GPU memory", 664 B). Causa: escape del integrador (falta de `@inline`; #207/#219 de DiffEqGPU). `boundary_callback(region, backend|configuration)` ahora elige: CPU → ContinuousCallback, GPU → DiscreteCallback + bisección `@inline` (sin escape: 636 B/rayo como la base). `Pkg.test()` 144/144. `test/gpu/callback_gpu.jl` actualizado; falta confirmarlo en la A30.
- 2026-11-16: Instalación con `Pkg.add(url=...)` fallaba en un entorno con `PlotlyJS` (reportado por el usuario). Causa: el compat `DiffEqGPU = "3.21"` puesto hoy; DiffEqGPU ≥ 3.17 exige `Parameters` 0.13.1, y con eso `PlotlyBase` (de `PlotlyJS`) no tiene versión posible. Solo DiffEqGPU ≤ 3.16 convive con `PlotlyJS`. Compat bajado a `DiffEqGPU = "3.16"`. Verificado: el entorno del usuario (PlotlyJS + DataFrames + Parameters) instala RayTracingGR con DiffEqGPU 3.16.1; los 144 tests pasan con DiffEqGPU 3.16.1 y con 3.21.6; el callback de GPU no escapa con 3.16.1 (igual que la referencia). (Se agregó y luego se quitó del README una recomendación de entorno propio: el usuario quiere que funcione en el entorno global.) Formato nuevo, más claro, de la salida de `test/gpu/callback_gpu.jl`.
- 2026-11-16: Prueba en la A30 OK con `boundary_callback(region, CUDABackend())` (10⁶ rayos, sin alocaciones en el kernel, igual que la CPU). Tarea para 2026-11-17: reordenar `src/propagate/Callback/` (no es modular). Documentación revisada: jerarquía de tipos del README con `AbstractRegion`/`PulsarRegion`, descripción de los tests de callback en CONTRIBUTING, estado del #16.
- 2026-11-17: `src/propagate/Callback/` reordenado según el patrón de `spacetime/` (decisiones del usuario: `AbstractRegion` sigue en `propagate_types.jl`, regiones sin `module`, código repetido CPU/GPU aceptado porque nunca se usan los dos a la vez, tests sin cambios). `callback.jl` = índice con funciones genéricas vacías; `Region/PulsarRegion.jl`; `boundary_callback_cpu.jl`; `boundary_callback_gpu.jl` (código idéntico al anterior). `Pkg.test()` 144/144 OK. README: árbol y diagrama actualizados. Commit `65730b6` + push (WSL).
- 2026-11-19: `propagate` (tarea del backend). Revisado el `solve` de Skylight-Pulsar (`Propagate/Utils_propagate.jl`): se respeta su estructura (CPU/GPU separados, `adaptive = true`, `save_everystep = false`, `batch_size = 50_000` en GPU, sin `dt`), con tolerancias nuevas. Mediciones fuera del repo (kernel de GPU sobre la CPU, púlsar KS a = 0.1 y 0.9): tolerancia `5 eps(T)`; CPU Float64 `Vern9`; GPU Float32 `GPUTsit5` (misma precisión que `GPUVern7`, `GPUVern9` peor); `Vern9` en Float32 deja rayos hasta 3.5e-3 del borde; paso de Newton en el callback de GPU (corte de ~9 a 1–3 ulp). Todo en `dev/PROPAGATE_2026-11-19.md`. `Pkg.test()` 166/166. README: uso de `propagate`, árbol y diagramas. Commit `d4d50c6` + push (WSL).
- 2026-11-19: Prueba de `propagate` en la A30 con `test/gpu/propagate_gpu.jl`: todo OK (sin memoria en el kernel; 10⁶ rayos con 0 sin cortar y ≤ 3 ulp del borde; coincide con CPU Float64). La sesión de Jupyter vio 6 GiB de la A30.
