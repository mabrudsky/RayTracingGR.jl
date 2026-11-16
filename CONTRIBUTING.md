# Contribuir a RayTracingGR.jl

Guía para colaboradores: cómo agregar un espacio-tiempo y cómo correr los tests. La guía de uso y la
estructura del proyecto (carpetas, carga de archivos, flujo de un rayo y jerarquía de tipos) están en el
[`README.md`](README.md#estructura-del-proyecto).

Las tareas pendientes, el historial de cambios y la revisión de código están en
[`dev/TAREAS.md`](dev/TAREAS.md) y [`dev/REVISION_2026-11-12.md`](dev/REVISION_2026-11-12.md).

---

# Agregar un nuevo espacio-tiempo (patrón actual)

1. Crear `src/spacetime/<Familia>/<Familia>.jl` con un `module <Familia>` (nombre en CamelCase, como
   `Kerr`), que importe desde `..RayTracingGR` solo los tipos y funciones que use, y con `import` las
   funciones `metric`, `metric_inverse`, `numtype` y `coordinate_radius` (para agregarles un método).
2. En un archivo aparte, definir el struct (isbits, parametrizado en `T<:Union{Float32,Float64}`) y
   los métodos `metric(st, x::SVector{4,T})`, `metric_inverse(...)`,
   `numtype(::MiEspacioTiempo{T}) where {T} = T` y `coordinate_radius(st, x)`, con el radio propio
   de ese sistema de coordenadas (por ejemplo, en Boyer-Lindquist es `x[2]`; en Kerr-Schild, el radio
   de Kerr $r$). `coordinate_radius` se usa para los cortes de la integración. `metric` tiene que aceptar un `T` genérico, porque
   recibe números `Dual`. `christoffel` no hace falta: el método genérico lo calcula derivando `metric`
   (solo se escribe uno propio si hay una fórmula analítica más rápida). `Ensemble_Problem` usa `numtype` para comprobar que la precisión coincida
   con la de `u0`.
3. Incluirlo en `src/spacetime/spacetime.jl` y exportar el **submódulo** desde `src/RayTracingGR.jl`
   (`export <Familia>`). El submódulo no exporta nada: los tipos se usan siempre con el nombre de la
   familia adelante, por ejemplo `Kerr.KerrSchildCoordinates(M = 1.0, a = 0.9)`.

---

# Tests

```julia
# desde la carpeta del proyecto
julia --project=. -e "using Pkg; Pkg.test()"
```

| Archivo | Qué comprueba |
|---|---|
| `test/metric/christoffel_tests_KerrSchildCoordinates.jl` | Que la derivada covariante de la métrica se anule (∇g = 0) en tres puntos, en Float64 y Float32. Con eso comprueba que `metric_inverse` sea la inversa de `metric` y las convenciones de índices de `christoffel`. |
| `test/metric/kerr_schild_tests.jl` | Constructores de `KerrSchildCoordinates` (y sus errores); que todo sea isbits y que `metric`, `metric_inverse`, `christoffel` y `coordinate_radius` no aloquen memoria, en Float64 y Float32; que `coordinate_radius` dé el radio de Kerr $r$ (eje, ecuador y punto general). |
| `test/metric/christoffel_tests_BoyerLindquistCoordinates.jl` | ∇g = 0 en Boyer-Lindquist (Float64 y Float32), con tolerancia relativa al tamaño de la métrica. |
| `test/metric/boyer_lindquist_tests.jl` | Constructores, isbits y cero alocaciones; $g\,g^{-1} = I$; $\det g = -\Sigma^2\sin^2\theta$; igual a Kerr-Schild transformada; mismo `coordinate_radius` en los dos sistemas; límite $a = 0$ (Schwarzschild). |
| `test/propagate/ensemble_tests.jl` | Los chequeos de `Ensemble_Problem` (largo de `u0`, precisión, filas y tipo de `initial_data`, `initial_data` en la CPU) y que las ecuaciones no aloquen memoria. |
| `test/propagate/callback_tests.jl` | `PulsarRegion` (constructores, isbits, condición en Kerr-Schild y Boyer-Lindquist) y el corte con `boundary_callback`: rayos integrados con `Tsit5` (CPU) y con `GPUTsit5` en `EnsembleGPUKernel(CPU())` (el código del kernel de GPU, ejecutado en la CPU); todos terminan en $R_\text{in}$ o $R_\text{out}$ y los dos dan el mismo estado final. |

Las alocaciones se miden dentro de una función: medidas en el ámbito global (o con un alias global
no constante) dan resultados falsos.

# Benchmarks

Los tiempos (`@btime`) no son parte de los tests: están en `benchmark/benchmarks.jl`, con su propio
entorno, y se corren a mano.

```julia
# desde la carpeta del proyecto (la primera vez: instantiate)
julia --project=benchmark -e "using Pkg; Pkg.instantiate()"
julia --project=benchmark benchmark/benchmarks.jl
```

En Julia 1.10, `benchmark/Project.toml` no reconoce la sección `[sources]`: la primera vez hay que
correr `Pkg.develop(path = ".")` con `--project=benchmark` en lugar de `Pkg.instantiate()`.
