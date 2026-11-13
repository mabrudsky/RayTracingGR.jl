# Contribuir a RayTracingGR.jl

Guía para colaboradores: cómo agregar un espacio-tiempo y cómo correr los tests. La guía de uso y la
estructura del proyecto (carpetas, carga de archivos, flujo de un rayo y jerarquía de tipos) están en el
[`README.md`](README.md#estructura-del-proyecto).

Las tareas pendientes, el historial de cambios y la revisión de código están en
[`dev/TAREAS.md`](dev/TAREAS.md) y [`dev/REVISION_2026-11-12.md`](dev/REVISION_2026-11-12.md).

---

# Agregar un nuevo espacio-tiempo (patrón actual)

1. Crear `src/spacetime/<Familia>/<Familia>.jl` con un `module`, que importe desde
   `..RayTracingGR` los tipos abstractos y las funciones `metric`, `metric_inverse` y `christoffel`.
2. En un archivo aparte, definir el struct (isbits, parametrizado en `T<:Union{Float32,Float64}`) y
   los métodos `metric(st, x::SVector{4,T})`, `metric_inverse(...)`, `christoffel(...)` y
   `numtype(::MiEspacioTiempo{T}) where {T} = T`. `metric` tiene que aceptar un `T` genérico, porque
   recibe números `Dual`. `Ensemble_Problem` usa `numtype` para comprobar que la precisión coincida
   con la de `u0`.
3. Incluirlo en `src/spacetime/spacetime.jl` y exportarlo desde `src/RayTracingGR.jl`.

---

# Tests

```julia
# desde la carpeta del proyecto
julia --project=. -e "using Pkg; Pkg.test()"
```

El test actual verifica que la derivada covariante de la métrica se anule (∇g = 0) en tres puntos.
Con eso comprueba que `metric_inverse` sea la inversa de `metric` y que las convenciones de índices
de `christoffel` sean correctas. También muestra benchmarks: todas las funciones corren sin alocar
memoria.
