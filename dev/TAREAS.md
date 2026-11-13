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
<summary><b>Bloque 3: Estructura</b></summary>

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
- [ ] #11 Fallbacks genéricos de `christoffel` y `metric_inverse`; eliminar `AbstractChristoffel`.

</details>

<details>
<summary><b>Bloque 4: GPU</b></summary>

- [ ] #6 Contraer Γ^λ_{μν}u^μv^ν directamente, sin construir el Γ completo (menos registros y
  operaciones).
- [ ] #7 `ntuple(f, 4)` → `ntuple(f, Val(4))`.
- [ ] #8 Resolver con `save_everystep=false, save_start=false`; `initial_data` tiene que ser una
  `Matrix` de CPU.

</details>

<details>
<summary><b>Bloque 5: Tests y orden</b></summary>

- [ ] #15 Mover los benchmarks a `benchmark/`; tests de isbits, `@allocated == 0` y Float32.
- [ ] #16 Callback de terminación (horizonte / radio máximo); mover `src/Document` a `docs/`; sacar
  `utils_config_path.jl`; renombrar `k` → `st` en `_christoffel_ForwardDiff`.

</details>

---

## Tareas agregadas por el usuario

<!-- Formato: - [ ] (YYYY-MM-DD, para: YYYY-MM-DD o "sin fecha") descripción -->

- [x] (2026-11-13, para: sin fecha) **Continuar con las correcciones del Bloque 2 — Dependencias** (#12 `DifferentialEquations` → `SciMLBase`, #13 `BenchmarkTools` solo en tests, #14 quitar `Parameters` / `import KernelAbstractions as KA`). Es lo próximo después del Bloque 1 (completo).

### Bloque 6 — Tests de validación (AD de Christoffel + métricas bien escritas)
Objetivo del usuario: el test de ∇g = 0 se hizo para verificar que la diferenciación automática de
Christoffel funcione y que las métricas estén bien escritas. Pero ∇g = 0 es una identidad algebraica:
solo prueba g·g⁻¹ = I y las convenciones de índices. Estos tests cubren lo que falta.
Detalle en REVISION_2026-11-12.md §17–21.

- [ ] (2026-11-12, para: sin fecha) #17 **AD vs diferencias finitas**: comparar `christoffel` con Γ armado a partir de ∂g por diferencias finitas centradas (paso h ~ 1e-6·escala, rtol ~ 1e-7, Float64). Es la prueba independiente de ForwardDiff. Objetivo: AD.
- [ ] (2026-11-12, para: sin fecha) #18 **Vector nulo de Kerr-Schild**: η^{μν} l_μ l_ν = 0, g^{μν} l_μ l_ν = 0 y g_{μν} lᵘ lᵛ = 0 (con lᵘ = η^{μν} l_ν). Valida `_H_l`. Objetivo: métrica.
- [ ] (2026-11-12, para: sin fecha) #19 **Límite a = 0**: comparar `metric` con Schwarzschild en Kerr-Schild escrito a mano (2H = 2M/r, l = (1, x/r, y/r, z/r)). Objetivo: métrica.
- [ ] (2026-11-12, para: sin fecha) #20 **Ricci = 0 (vacío)**: R_{μν} = ∂_λΓ^λ_{μν} − ∂_νΓ^λ_{μλ} + Γ^λ_{λσ}Γ^σ_{μν} − Γ^λ_{νσ}Γ^σ_{μλ} ≈ 0, derivando `christoffel` con Dual anidado (prueba también AD de segundo orden) o con diferencias finitas. Es la prueba más fuerte de que la métrica es solución de Einstein. Objetivo: métrica + AD.
- [ ] (2026-11-12, para: sin fecha) #21 **Constantes de movimiento** a lo largo de una geodésica integrada: g_{μν}kᵘkᵛ = 0, E = −k_t, L_z = x k_y − y k_x (índices bajos) y constante de Carter. Valida toda la cadena: métrica → Γ → ecuaciones → solver. Depende de tener la propagación (`Ensemble_Problem` + `solve`) funcionando.
- Nota para todos: evaluar varios puntos, incluyendo dentro del horizonte (r < r₊), cerca del eje y en Float32 (tolerancias más laxas).
- Nota #21: la constante de Carter NO es genérica (tensor de Killing de Kerr) → método específico de `AbstractKerrSpacetime`.

### Funciones físicas genéricas — A PEDIDO (2026-11-13)
**Principio:** cada magnitud (frecuencias, energía, lapse/shift, tétradas, curvatura, callbacks...) se
implementa siguiendo el **patrón de `christoffel`**: función genérica de `(spacetime, x)` que solo usa
`metric`, `metric_inverse` y derivadas por Duales; sirve para cualquier métrica.
**No se planifica por adelantado:** el usuario pide cada función cuando la necesita.
Descartado: jerarquía de simetrías, Killing a mano, descubrimiento de Killing, `lapse2`.
Recordatorio: el horizonte de eventos no es local (no encaja en el patrón); para callbacks usar
criterios locales (p. ej. Kretschmann / frecuencia para captura, radio de coordenadas para escape).

---

- [ ] (2026-11-13, para: sin fecha) **Usar `configuration.backend` para elegir el integrador** (CPU / GPU). Hoy el campo se guarda pero nada lo usa. Idea: `solve(ep, GPUTsit5(), DiffEqGPU.EnsembleGPUKernel(configuration.backend); ...)`.

- [ ] (2026-11-13, para: sin fecha) **Sección "Cómo citar" del README**: está oculta en un comentario HTML (`<!-- ... -->`). `CITATION.bib` ya corregido (versión `v0.1.0`, email `@gmail.com`); falta decidir cuándo mostrarla (descomentar).

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
