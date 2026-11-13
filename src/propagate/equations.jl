function geodesic_equations_Vacuum(u::SVector{8,T}, p, t) where {T}           # u = (x⁰..x³, k⁰..k³)
    @inbounds begin
        position = SVector{4,T}(u[1], u[2], u[3], u[4])
        k = SVector{4,T}(u[5], u[6], u[7], u[8])
    end
    kkT = k * k'   
    Γ = christoffel(p.spacetime, position) 
    a = @SVector T[-sum(Γ[:, :, λ] .* kkT) for λ in 1:4]    # Geodesic equation kᵃ∇ₐkᵇ=0 where k is the momentum of the photon.

    du = vcat(k, a)
    return du
end 

function parallel_transport_equations_Vacuum(u::SVector{16,T}, p, t) where {T}  # u = (x, k, ex, ey)
    @inbounds begin
        position = SVector{4,T}(u[1], u[2], u[3], u[4])
        k  = SVector{4,T}(u[5], u[6], u[7], u[8])
        ex = SVector{4,T}(u[9], u[10], u[11], u[12])
        ey = SVector{4,T}(u[13], u[14], u[15], u[16])
    end

    _kkT  = k * k'    
    _k_ex = k * ex'
    _k_ey = k * ey'
    
    Γ = christoffel(p.spacetime, position)
    
    a   = @SVector T[-sum(Γ[:, :, λ] .* _kkT)  for λ in 1:4]  # Geodesic equation kᵃ∇ₐkᵇ=0 where k is the momentum of the photon. 
    dex = @SVector T[-sum(Γ[:, :, λ] .* _k_ex) for λ in 1:4]  # Parallel transport equation kᵃ∇ₐexᵇ=0 where ex is the rectangular coordinates of the image plane.            
    dey = @SVector T[-sum(Γ[:, :, λ] .* _k_ey) for λ in 1:4]  # Parallel transport equation kᵃ∇ₐeyᵇ=0 where ey is the rectangular coordinates of the image plane. 

    du = vcat(k, a, dex, dey)
    return du
end