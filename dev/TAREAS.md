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
- [x] #2 `Configuration` no se podía construir (no había `AbstractPhysicalModel` concreto) — hecho 2026-11-13 con **opción A**: se eliminó `model` de `Configuration` e `IntegrationParameters` y se borró `AbstractPhysicalModel`. La física la define el tipo de ecuación (`GeodesicEquationsVacuum`, ...). Si una ecuación futura necesita parámetros (p. ej. plasma `ωp`), van dentro de su struct y la ecuación se agrega como campo de `IntegrationParameters`.
- [x] #3 `AbstractCoordinate <: AbstractRayTracing` (antes colgaba de `AbstractSpacetime`) — hecho 2026-11-13. **Se conserva el campo `topology`** (decisión 2026-11-13): se usará cuando Kerr tenga otros sistemas de coordenadas (p. ej. Boyer-Lindquist) y es clave para los Callbacks (`radius(st.topology, st, x)`). Se descartó pasarlo a trait.
  - ~~Idea: extraer r de Kerr de `_H_l` para el callback~~ → descartada (contradice el principio genérico; ver "Funciones físicas genéricas — a pedido").
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
