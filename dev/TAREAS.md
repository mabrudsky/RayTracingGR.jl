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

### Bloque 1 — Bugs
- [x] #1 Constructor posicional `KerrSchildCoordinates(1.0, 0.5)` siempre falla — hecho 2026-11-12 (opción B, sin `@with_kw`; se conservó el campo `topology` hasta #3)
- [ ] #2 Crear modelo físico concreto `struct Vacuum <: AbstractPhysicalModel end`
- [ ] #3 `AbstractCoordinate <: AbstractRayTracing` + trait `coordinates(...)` en vez de campo `topology`
- [ ] #4 Ligar `N` a la ecuación (`SVector{8}` / `SVector{16}` + `state_length`)
- [ ] #5 Evitar mezcla de precisión (`numtype` + chequeo)

### Bloque 2 — Dependencias
- [ ] #12 Reemplazar `DifferentialEquations` por `SciMLBase` (compat `3.50`)
- [ ] #13 `BenchmarkTools` solo en tests
- [ ] #14 Quitar `Parameters`; `import KernelAbstractions as KA`
  - Nota: tras #1, `KerrSchildCoordinates` ya no usa `@with_kw`; el `using Parameters` de `src/spacetime/Kerr/Kerr.jl` quedó sin uso. `Configuration`/`IntegrationParameters` (config_types.jl) todavía usan `@with_kw` → pasar a `Base.@kwdef`.

### Bloque 3 — Estructura
- [ ] #9 Unificar `_Ensemble_Problem` con `rhs(...)`; tipos a `propagate_types.jl`
- [ ] #10 Submódulo `kerr` → `Kerr`, exportar `KerrSchildCoordinates`, limpiar imports/exports
- [ ] #11 Fallbacks genéricos `christoffel` / `metric_inverse`; quitar `AbstractChristoffel`

### Bloque 4 — GPU
- [ ] #6 Contracción directa de Γ (`_metric_derivatives`, `_contract`) + test de comparación
- [ ] #7 `ntuple(f, Val(4))`
- [ ] #8 `save_everystep=false, save_start=false`; `initial_data` como `Matrix` de CPU

### Bloque 5 — Tests y orden
- [ ] #15 Benchmarks a `benchmark/`; tests isbits, `@allocated == 0`, Float32
- [ ] #16 Callback de terminación; mover `.md` de `src/` a `docs/`; sacar `utils_config_path.jl`; renombrar `k` → `st`

---

## Tareas agregadas por el usuario

<!-- Formato: - [ ] (YYYY-MM-DD, para: YYYY-MM-DD o "sin fecha") descripción -->

### Bloque 6 — Tests de validación (AD de Christoffel + métricas bien escritas)
Objetivo del usuario: el test de ∇g = 0 se hizo para verificar que la diferenciación automática de
Christoffel funcione y que las métricas estén bien escritas. Pero ∇g = 0 es una identidad algebraica:
solo prueba g·g⁻¹ = I y las convenciones de índices. Estos tests cubren lo que falta.
Detalle en REVISION_2026-11-12.md §17–21.

- [ ] (2026-11-12, para: sin fecha) #17 **AD vs diferencias finitas**: comparar `christoffel` con Γ armado a partir de ∂g por diferencias finitas centradas (paso h ~ 1e-6·escala, rtol ~ 1e-7, Float64). Es la prueba independiente de ForwardDiff. Objetivo: AD.
- [ ] (2026-11-12, para: sin fecha) #18 **Vector nulo de Kerr-Schild**: η^{μν} l_μ l_ν = 0, g^{μν} l_μ l_ν = 0 y g_{μν} lᵘ lᵛ = 0 (con lᵘ = η^{μν} l_ν). Valida `_H_l`. Objetivo: métrica.
- [ ] (2026-11-12, para: sin fecha) #19 **Límite a = 0**: comparar `metric` con Schwarzschild en Kerr-Schild escrito a mano (2H = 2M/r, l = (1, x/r, y/r, z/r)). Objetivo: métrica.
- [ ] (2026-11-12, para: sin fecha) #20 **Ricci = 0 (vacío)**: R_{μν} = ∂_λΓ^λ_{μν} − ∂_νΓ^λ_{μλ} + Γ^λ_{λσ}Γ^σ_{μν} − Γ^λ_{νσ}Γ^σ_{μλ} ≈ 0, derivando `christoffel` con Dual anidado (prueba también AD de segundo orden) o con diferencias finitas. Es la prueba más fuerte de que la métrica es solución de Einstein. Objetivo: métrica + AD.
- [ ] (2026-11-12, para: sin fecha) #21 **Constantes de movimiento** a lo largo de una geodésica integrada: g_{μν}kᵘkᵛ = 0, E = −k_t, L_z = x k_y − y k_x (índices bajos) y constante de Carter. Valida toda la cadena: métrica → Γ → ecuaciones → solver. Depende de #2 (Vacuum) y de tener `Ensemble_Problem`/ODE funcionando.
- Nota para todos: evaluar varios puntos, incluyendo dentro del horizonte (r < r₊), cerca del eje y en Float32 (tolerancias más laxas).

---

## Historial de sesiones

- 2026-11-12: Revisión completa (solo lectura). Creados `dev/REVISION_2026-11-12.md` y `dev/TAREAS.md`.
- 2026-11-12: Aplicado #1 en `src/spacetime/Kerr/KerrSchildCoordinates.jl` (constructores explícitos: `{T}(M,a)` con validación por `throw`, posicional `(M::T, a::T)`, keywords `(; M, a)`, fallback `(::Real, ::Real)` con error claro). Se mantiene `KerrSchildCoordinates(M=1.0, a=0.5)`. Test existente OK con `julia --project=. test/runtests.jl`: 3/3 pass, |∇g| máx ~1.7e-16, 0 alocaciones. Sin cambios en tests.
- 2026-11-12: README reescrito (estructura, diagramas de carpetas/includes/flujo/tipos, convenciones, sección Pendientes). Al cambiar tareas, mantener sincronizada la sección "Pendientes" del README con este archivo.
- 2026-11-12: Agregado Bloque 6 (#17–#21, tests de validación de AD y métricas) a pedido del usuario. Solo registrado, no implementado.
