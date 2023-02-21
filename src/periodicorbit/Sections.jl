abstract type AbstractSection end

update!(sh::AbstractSection) = error("Not yet implemented. You can use the dummy function `sh->true`.")

####################################################################################################
function sectionShooting(x::AbstractVector, T,
						normal::AbstractVector,
						center::AbstractVector)
	N = length(center)
	# we only constrain the first point to lie on a specific hyperplane
	# this avoids the temporary xc - centers
	return (dot(x, normal) - dot(center, normal)) * T
end

# section for Standard Shooting
"""
$(TYPEDEF)

This composite type (named for Section Standard Shooting) encodes a type of section implemented by a single hyperplane. It can be used in conjunction with [`ShootingProblem`](@ref). The hyperplane is defined by a point `center` and a `normal`.

$(TYPEDFIELDS)

"""
struct SectionSS{Tn, Tc}  <: AbstractSection
	"Normal to define hyperplane"
	normal::Tn

	"Representative point on hyperplane"
	center::Tc
end

(sect::SectionSS)(u, T) = sectionShooting(u, T, sect.normal, sect.center)

# matrix-free jacobian
function (sect::SectionSS)(u, T::Ty, du, dT::Ty) where Ty
	return sect(u, one(Ty)) * dT + dot(du, sect.normal) * T
end

isEmpty(sect::SectionSS{Tn, Tc}) where {Tn, Tc} = (Tn == Nothing) || (Tn == Nothing)

# we update the field of Section, useful during continuation procedure for updating the section
function update!(sect::SectionSS, normal, center)
	copyto!(sect.normal, normal)
	copyto!(sect.center, center)
	sect
end

####################################################################################################
# Poincare shooting based on Sánchez, J., M. Net, B. Garcı́a-Archilla, and C. Simó. “Newton–Krylov Continuation of Periodic Orbits for Navier–Stokes Flows.” Journal of Computational Physics 201, no. 1 (November 20, 2004): 13–33. https://doi.org/10.1016/j.jcp.2004.04.018.

function sectionHyp!(out, x, normals, centers)
	for ii in eachindex(normals)
		out[ii] = dot(normals[ii], x) - dot(normals[ii], centers[ii])
	end
	out
end

"""
$(TYPEDEF)

This composite type (named for SectionPoincaréShooting) encodes a type of Poincaré sections implemented by hyperplanes. It can be used in conjunction with [`PoincareShootingProblem`](@ref). Each hyperplane is defined par a point (one example in `centers`) and a normal (one example in `normals`).

$(TYPEDFIELDS)

# Constructor(s)
    SectionPS(normals::Vector{Tv}, centers::Vector{Tv})

"""
struct SectionPS{Tn, Tc, Tnb, Tcb} <: AbstractSection
	M::Int64				# number of hyperplanes
	normals::Tn 			# normals to define hyperplanes
	centers::Tc 			# representative point on each hyperplane
	indices::Vector{Int64} 	# indices to be removed in the operator Ek

	normals_bar::Tnb
	centers_bar::Tcb

	function SectionPS(normals, centers)
		@assert length(normals) == length(centers)
		M = length(normals)
		indices = zeros(Int64, M)
		for ii in 1:M
			indices[ii] = argmax(abs.(normals[ii]))
		end
		nbar = [R(normals[ii], indices[ii]) for ii in 1:M]
		cbar = [R(centers[ii], indices[ii]) for ii in 1:M]

		return new{typeof(normals), typeof(centers), typeof(nbar), typeof(cbar)}(M, normals, centers, indices, nbar, cbar)
	end

	SectionPS(M = 0) = new{Nothing, Nothing, Nothing, Nothing}(M, nothing, nothing, Int64[], nothing, nothing)
end

(hyp::SectionPS)(out, u) = sectionHyp!(out, u, hyp.normals, hyp.centers)
isEmpty(sect::SectionPS{Tn, Tc, Tnb, Tcb}) where {Tn, Tc, Tnb, Tcb} = (Tn == Nothing) || (Tc == Nothing)

_selectIndex(v) = argmax(abs.(v))
# ==================================================================================================
function _duplicate!(x::AbstractVector)
	n = length(x)
	for ii in 1:n
		push!(x, copy(x[ii]))
	end
	x
end
_duplicate(x::AbstractVector) = _duplicate!(copy(x))
_duplicate(hyp::SectionPS) = SectionPS(_duplicate(hyp.normals), _duplicate(hyp.centers))
_duplicate(hyp::SectionSS) = SectionSS(_duplicate(hyp.normals), _duplicate(hyp.centers))
# ==================================================================================================
"""
	update!(hyp::SectionPS, normals, centers)

Update the hyperplanes saved in `hyp`.
"""
function update!(hyp::SectionPS, normals, centers)
	M = hyp.M
	@assert length(normals) == M "Wrong number of normals"
	@assert length(centers) == M "Wrong number of centers"
	for ii in 1:M
		hyp.normals[ii] .= normals[ii]
		hyp.centers[ii] .= centers[ii]
		k = _selectIndex(normals[ii])
		hyp.indices[ii] = k
		R!(hyp.normals_bar[ii], normals[ii], k)
		R!(hyp.centers_bar[ii], centers[ii], k)
	end
	return hyp
end

# Operateur Rk from the paper above
@views function R!(out, x::AbstractVector, k::Int)
	out[1:k-1] .= x[1:k-1]
	out[k:end] .= x[k+1:end]
	return out
end

R!(hyp::SectionPS, out, x::AbstractVector, k::Int) = R!(out, x, hyp.indices[k])
R(x::AbstractVector, k::Int) = R!(similar(x, length(x) - 1), x, k)
R(hyp::SectionPS, x::AbstractVector, k::Int) = R!(hyp, similar(x, length(x) - 1), x, k)

# differential of R
dR!(hyp::SectionPS, out, dx::AbstractVector, k::Int) = R!(hyp, out, dx, k)
dR(hyp::SectionPS, dx::AbstractVector, k::Int) = R(hyp, dx, k)

# Operateur Ek from the paper above
function E!(hyp::SectionPS, out, xbar::AbstractVector, ii::Int)
	@assert length(xbar) == length(hyp.normals[1]) - 1 "Wrong size for the projector / expansion operators, length(xbar) = $(length(xbar)) and length(normal) = $(length(hyp.normals[1]))"
	k = hyp.indices[ii]
	nbar  = hyp.normals_bar[ii]
	xcbar = hyp.centers_bar[ii]
	coord_k = hyp.centers[ii][k] - (dot(nbar, xbar) - dot(nbar, xcbar)) / hyp.normals[ii][k]

	@views out[1:k-1] .= xbar[1:k-1]
	@views out[k+1:end] .= xbar[k:end]
	out[k] = coord_k
	return out
end

function E(hyp::SectionPS, xbar::AbstractVector, ii::Int)
	out = similar(xbar, length(xbar) + 1)
	E!(hyp, out, xbar, ii)
end

# differential of E!
function dE!(hyp::SectionPS, out, dxbar::AbstractVector, ii::Int)
	k = hyp.indices[ii]
	nbar = hyp.normals_bar[ii]
	xcbar = hyp.centers_bar[ii]
	coord_k = - dot(nbar, dxbar) / hyp.normals[ii][k]

	@views out[1:k-1]   .= dxbar[1:k-1]
	@views out[k+1:end] .= dxbar[k:end]
	out[k] = coord_k
	return out
end

function dE(hyp::SectionPS, dxbar::AbstractVector, ii::Int)
	out = similar(dxbar, length(dxbar) + 1)
	dE!(hyp, out, dxbar, ii)
end
