@with_kw Jet{Tf, Tdf, Td2f, Td3f, Tj, Td}
	δ::Td = 1e-8
	F::Tf
	dF::Tdf = (x,p,dx) -> (F(x .+ δ .* dx, p) - F(x, p)) / δ
	d2F::Td2f = (x,p,dx1,dx2) -> (dF(x .+ δ .* dx2, p, dx1) - dF(x, p, dx1)) / δ
	d3F::Td3f = (x,p,dx1,dx2) -> (d2F(x .+ δ .* dx3, p, dx1, dx2) - d2F(x, p, dx1, dx2)) / δ
	J::Tj = nothing
end

Jet(f) = Jet(F = f)
(j::Jet)(x, p) = j.F(x, p)
(j::Jet)(::Val{Jacobian}, x, p) = j.J(x, p)
(j::Jet)(x, p, dx) = j.dF(x, p, dx)
(j::Jet)(x, p, dx1, dx2) = j.d2F(x, p, dx1, dx2)
(j::Jet)(x, p, dx1, dx2, dx3) = j.d2F(x, p, dx1, dx2, dx3)
