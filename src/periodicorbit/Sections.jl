abstract type AbstractSection end

update!(sh::AbstractSection) = error("Not yet implemented. You can use the dummy function `sh->true`.")

####################################################################################################
@views function sectionShooting(x::AbstractVector, normals::AbstractVector, centers::AbstractVector)
	res = eltype(x)(1)
	M = length(centers); N = div(length(x) - 1, M)
	xv = x[1:end-1]
	xc = reshape(xv, N, M)
	for ii in 1:M
		# this avoids the temporary xc - centers
		res *= dot(xc[:, ii], normals[ii]) - dot(centers[ii], normals[ii])
	end
	res
end

# section for Standard Shooting
"""
$(TYPEDEF)

This composite type (named for Section Standard Shooting) encodes a type of sections implemented by hyperplanes. It can be used in conjunction with [`ShootingProblem`](@ref). Each hyperplane is defined by a point (one example in `centers`) and a normal (one example in `normals`).

$(TYPEDFIELDS)

"""
struct SectionSS{Tn, Tc}  <: AbstractSection
	"Normals to define hyperplanes"
	normals::Tn

	"Representative point on each hyperplane"
	centers::Tc
end

(sect::SectionSS)(u) = sectionShooting(u, sect.normals, sect.centers)

# we update the field of Section, useful during continuation procedure for updating the section
function update!(sect::SectionSS, normals, centers)
	copyto!(sect.normals, normals)
	copyto!(sect.centers, centers)
	sect
end

####################################################################################################
# Poincare shooting based on Sánchez, J., M. Net, B. Garcı́a-Archilla, and C. Simó. “Newton–Krylov Continuation of Periodic Orbits for Navier–Stokes Flows.” Journal of Computational Physics 201, no. 1 (November 20, 2004): 13–33. https://doi.org/10.1016/j.jcp.2004.04.018.

function sectionHyp!(out, x, normals, centers)
	for ii = 1:length(normals)
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
		nbar = [R(normals[ii], indices[ii]) for ii=1:M]
		cbar = [R(centers[ii], indices[ii]) for ii=1:M]

		return new{typeof(normals), typeof(centers), typeof(nbar), typeof(cbar)}(M, normals, centers, indices, nbar, cbar)
	end

	SectionPS(M = 0) = new{Nothing, Nothing, Nothing, Nothing}(M, nothing, nothing, Int64[], nothing, nothing)
end

(hyp::SectionPS)(out, u) = sectionHyp!(out, u, hyp.normals, hyp.centers)

"""
	update!(hyp::SectionPS, normals, centers)

Update the hyperplanes saved in `hyp`.
"""
function update!(hyp::SectionPS, normals, centers)
	M = hyp.M
	@assert length(normals) == M "Wrong number of normals"
	@assert length(centers) == M "Wrong number of centers"
	hyp.normals .= normals
	hyp.centers .= centers
	for ii in 1:M
		hyp.indices[ii] = argmax(abs.(normals[ii]))
		R!(hyp.normals_bar[ii], normals[ii], hyp.indices[ii])
		R!(hyp.centers_bar[ii], centers[ii], hyp.indices[ii])
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

# Operateur Ek from the paper above
function E!(hyp::SectionPS, out, xbar::AbstractVector, ii::Int)
	@assert length(xbar) == length(hyp.normals[1]) - 1 "Wrong size for the projector / expansion operators, length(xbar) = $(length(xbar)) and length(normal) = $(length(hyp.normals[1]))"
	k = hyp.indices[ii]
	nbar = hyp.normals_bar[ii]
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
