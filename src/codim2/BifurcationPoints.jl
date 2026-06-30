type(::Cusp) = :Cusp
type(::Bautin) = :Bautin
type(::ZeroHopf) = :ZeroHopf
type(::HopfHopf) = :HopfHopf
type(bp::BogdanovTakens) = :BogdanovTakens

function __get_lenses_and_params_codim2(bp)
    lens1, lens2 = bp.lens
    p1 = _get(bp.params, lens1)
    p2 = _get(bp.params, lens2)
    return lens1, lens2, p1, p2
end

function Base.show(io::IO, bp::Cusp)
    lens1, lens2, p1, p2 = __get_lenses_and_params_codim2(bp)
    printstyled(io, "Cusp", color=:cyan, bold = true)
    print(io, " bifurcation point at ", get_lens_symbol(lens1, lens2)," ≈ ($p1, $p2).\n")
    # avoid aliasing with user defined parameters
    p1 = :β1 == get_lens_symbol(lens1) ? :p1 : :β1
    p2 = :β2 == get_lens_symbol(lens2) ? :p2 : :β2
    println(io, "Normal form: $p1 + $p2⋅A + c⋅A³")
    c = bp.nf.c
    println(io, "Normal form coefficient:\n c = $c")
end

function Base.show(io::IO, bp::Bautin; prefix = "", detailed = false)
    lens1, lens2, p1, p2 = __get_lenses_and_params_codim2(bp)
    printstyled(io, "Bautin", color=:cyan, bold = true)
    print(io, " bifurcation point at ", get_lens_symbol(lens1, lens2)," ≈ ($p1, $p2).\n")
    println(io, prefix*"ω = ", bp.nf.ω)
    println(io, prefix*"Second lyapunov coefficient l₂ = ", bp.nf.l2)
    println(io, prefix*"Normal form: i⋅ω⋅z + l₂⋅z⋅|z|⁴")
    detailed && println(io, prefix*"Normal form coefficients (detailed):")
    detailed && println(io, bp.nf)
    nothing
end

function Base.show(io::IO, bp::ZeroHopf)
    lens1, lens2, p1, p2 = __get_lenses_and_params_codim2(bp)
    printstyled(io, "Zero-Hopf", color=:cyan, bold = true)
    print(io, " bifurcation point at ", get_lens_symbol(lens1, lens2)," ≈ ($p1, $p2).\n")
    println(io, "null eigenvalue ≈ ", bp.nf.λ0)
    println(io, "ω = ", bp.nf.ω)
    hasnf = get(bp.nf, :hasNS, nothing)
    if ~isnothing(hasnf)
        println(io, "There is a curve of NS of periodic orbits: ", hasnf)
    end
end

function Base.show(io::IO, bp::HopfHopf)
    lens1, lens2, p1, p2 = __get_lenses_and_params_codim2(bp)
    printstyled(io, "Hopf-Hopf", color=:cyan, bold = true)
    println(io, " bifurcation point at ", get_lens_symbol(lens1, lens2)," ≈ ($p1, $p2).")
    println(io, "Eigenvalues:\nλ1 = ", bp.nf.λ1, "\nλ2 = ", bp.nf.λ2)
    println(io, bp.nf)
end

function Base.show(io::IO, bp::BogdanovTakens)
    lens1, lens2, p1, p2 = __get_lenses_and_params_codim2(bp)
    printstyled(io, "Bogdanov-Takens", color=:cyan, bold = true)
    println(io, " bifurcation point at ", get_lens_symbol(lens1, lens2)," ≈ ($p1, $p2).")
    # avoid aliasing with user defined parameters
    p1 = :β1 == get_lens_symbol(lens1) ? :p1 : :β1
    p2 = :β2 == get_lens_symbol(lens2) ? :p2 : :β2
    println(io, "Normal form (B, $p1 + $p2⋅B + b⋅A⋅B + a⋅A²)")
    (;a, b) = bp.nf
    println(io, "Normal form coefficients:\n a = $a\n b = $b")
    println(io, "\nYou can call various predictors:\n - predictor(::BogdanovTakens, ::Val{:HopfCurve}, ds)\n - predictor(::BogdanovTakens, ::Val{:FoldCurve}, ds)\n - predictor(::BogdanovTakens, ::Val{:HomoclinicCurve}, ds)")
end
