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
| `test/propagate/callback_tests.jl` | `PulsarRegion` (constructores, isbits, condición en Kerr-Schild y Boyer-Lindquist) y el corte con `boundary_callback(region, backend)`: con `CPU()` devuelve un `ContinuousCallback` y con un backend de GPU un `DiscreteCallback` (isbits). Rayos integrados con `Tsit5` + la versión de CPU y con `GPUTsit5` en `EnsembleGPUKernel(CPU())` + la versión de GPU (el código del kernel de GPU, ejecutado en la CPU); todos terminan en $R_\text{in}$ o $R_\text{out}$ y los dos dan el mismo estado final; el kernel de GPU deja cada rayo a $\leq 3$ ulp del borde (paso de Newton). |
| `test/propagate/disk_region_tests.jl` | `AccretionDiskRegion`: constructores y errores; `coordinate_z` en Kerr-Schild y Boyer-Lindquist; signos de la condición y qué cruces detienen el rayo (`stops_at_boundary`); callback de GPU isbits. Cuatro rayos de resultado conocido (pega en el disco, cruza el plano afuera del disco y escapa por debajo, cae a $R_{\min}$, escapa por arriba) en CPU y en el kernel de GPU ejecutado en la CPU, Float64 y Float32. |
| `test/propagate/solve_tests.jl` | `propagate`: valores por defecto (`Vern9` en CPU, `GPUTsit5` en GPU, tolerancia $5\,\varepsilon_T$); CPU Float64 con `Vern9` (borde $< 10^{-10}$, condición nula $< 10^{-12}$); Float32 con `Tsit5`; cambio de algoritmo y opciones; errores por algoritmo incompatible con el backend u opciones internas. |
| `test/propagate/constants_of_motion_tests.jl` | Cadena completa (métrica → Christoffel → ecuaciones → solver → callback): $g_{\mu\nu}k^\mu k^\nu = 0$, $E = -k_t$, $L_z = x\,k_y - y\,k_x$ y la constante de Carter $Q$ se conservan entre el principio y el final de cada rayo ($a = 0.1$ y $0.9$; Float64 con `Vern9` y Float32 con `Tsit5`). |

Las alocaciones se miden dentro de una función: medidas en el ámbito global (o con un alias global
no constante) dan resultados falsos.

# Prueba en GPU

El CI no tiene GPU: `Pkg.test()` ejecuta el código del kernel de GPU en la CPU
(`EnsembleGPUKernel(CPU())`). La prueba en una GPU NVIDIA real está en `test/gpu/`, con su propio
entorno (CUDA, DiffEqGPU, ...), y se corre a mano:

```julia
# desde la carpeta del proyecto (la primera vez: instantiate)
julia --project=test/gpu -e "using Pkg; Pkg.instantiate()"
julia -t auto --project=test/gpu test/gpu/callback_gpu.jl
```

`callback_gpu.jl` prueba `boundary_callback(region, CUDABackend())` con `GPUTsit5` y
`EnsembleGPUKernel(CUDABackend())` en Float32. Con un backend de GPU, `boundary_callback` devuelve un
`DiscreteCallback` + bisección sobre la interpolación del integrador, todo `@inline`
(`ContinuousCallback` hace que el integrador escape a la memoria dinámica de la GPU: issue #219 de DiffEqGPU).
El script muestra:

1. si el kernel aloca memoria dentro de la GPU (`gpu_malloc`, causa del error "Out of dynamic GPU memory");
2. corridas con cada vez más rayos (variable `NRAYS`, por defecto `10000,100000,1000000`): cuántos rayos
   pegan en la estrella, cuántos escapan, distancia máxima al borde y tiempo;
3. comparación con la CPU (`Tsit5` + `EnsembleThreads`) en los primeros `NCPU` rayos.

Con `TRY_CONTINUOUS=true` prueba además `ContinuousCallback` en la GPU (al final, porque falla por el
#219). Sin una GPU NVIDIA disponible, ejecuta el kernel en la CPU.
Ejemplo con otros tamaños: `NRAYS=100000,5000000 julia -t auto --project=test/gpu test/gpu/callback_gpu.jl`.

En Julia 1.10, `test/gpu/Project.toml` no reconoce la sección `[sources]`: la primera vez hay que correr
`Pkg.develop(path = ".")` con `--project=test/gpu` en lugar de `Pkg.instantiate()`.

**`propagate_gpu.jl`** prueba `propagate` completo en una GPU real, con NVIDIA, AMD, Intel o Apple. El script
activa solo su entorno, así que se corre sin `--project`:

```bash
julia -t auto test/gpu/propagate_gpu.jl                # GPU por defecto: metal en macOS, cuda en el resto
GPU=amdgpu julia -t auto test/gpu/propagate_gpu.jl     # otra GPU: cuda, amdgpu, oneapi o metal
```

| `GPU` | Paquete | Entorno | Archivo de backend | Probado en GPU real |
|---|---|---|---|---|
| `cuda` | CUDA.jl | `test/gpu/` | `backend_cuda.jl` | Sí (A30) |
| `amdgpu` | AMDGPU.jl | `test/gpu/amdgpu/` | `backend_amdgpu.jl` | No |
| `oneapi` | oneAPI.jl | `test/gpu/oneapi/` | `backend_oneapi.jl` | No |
| `metal` | Metal.jl | `test/gpu/metal/` | `backend_metal.jl` | No |

Todo lo que depende de la GPU (el backend, el nombre del dispositivo, la memoria y el chequeo de memoria dentro
del kernel) está en el archivo de backend; el resto del script es igual para todas. Si una de esas funciones no
anda en una GPU, el script imprime "could not be checked" o "(unknown)" y sigue. Imprime si la GPU soporta
Float64: las de Apple y algunas de Intel no, y ahí todo se corre en Float32 (`propagate` da un error claro si se
le pasa Float64).

# Benchmarks

Los tiempos (`@btime`) no son parte de los tests: están en `benchmark/benchmarks.jl`, con su propio
entorno, y se corren a mano.

```julia
# desde la carpeta del proyecto (la primera vez: instantiate)
julia --project=benchmark -e "using Pkg; Pkg.instantiate()"
julia --project=benchmark benchmark/benchmarks.jl
julia --project=benchmark benchmark/integration_step.jl   # costo de un paso u_n → u_{n+1} + gráfico
```

`benchmark/integration_step.jl` mide con `@benchmark` cuánto cuesta, dado un estado $u_n$, obtener $u_{n+1}$ en CPU (Float64),
y regenera `benchmark/integration_step.png`, el gráfico de la sección "Rendimiento" del
README.

En Julia 1.10, `benchmark/Project.toml` no reconoce la sección `[sources]`: la primera vez hay que
correr `Pkg.develop(path = ".")` con `--project=benchmark` en lugar de `Pkg.instantiate()`.
