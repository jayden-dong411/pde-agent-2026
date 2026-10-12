import pytest

from pde_agent.equations import builtin_equations, manufactured_problem
from pde_agent.problem import PDEProblem
from pde_agent.solvers import builtin_registry
from pde_agent.solvers.base import SolverDiverged
from pde_agent.verify import convergence_study

CASES = [
    ("poisson", "sin(pi*x)*exp(x)", [(0, 1)], None, {"k": 1.5}, "dirichlet"),
    ("poisson", "sin(pi*x)*sin(pi*y)*exp(x)", [(0, 1), (0, 2)], None, {}, "dirichlet"),
    ("heat", "sin(pi*x)*exp(-t) + x*t", [(0, 1)], 0.5, {"alpha": 0.5}, "dirichlet"),
    ("advection", "sin(2*pi*x)*cos(t) + 0.5*cos(4*pi*x)*exp(-t)", [(0, 1)], 0.5, {"c": 1.0}, "periodic"),
    ("advection", "sin(2*pi*(x + 0.5*t))", [(0, 1)], 1.0, {"c": -0.5}, "periodic"),
    ("wave", "exp(x)*cos(2*t) + sin(3*x)*t", [(0, 1)], 1.0, {"c": 1.3}, "dirichlet"),
]


def _studies():
    eqs, registry = builtin_equations(), builtin_registry()
    for kind, exact, domain, t_final, coeffs, boundary in CASES:
        problem = manufactured_problem(eqs.get(kind), exact, domain, t_final, coeffs, boundary)
        for spec in registry.for_problem(problem):
            yield pytest.param(spec, problem, id=f"{spec.name}-{problem.dim}d-{coeffs}")


@pytest.mark.parametrize("spec,problem", list(_studies()))
def test_builtin_solvers_converge_at_their_order(spec, problem):
    ns = [16, 32, 64] if problem.dim == 1 else [8, 16, 32]
    study = convergence_study(spec, problem, ns)
    assert study.errors[-1] < 0.1
    assert study.final_order == pytest.approx(spec.order, abs=0.35)


def test_every_builtin_kind_has_a_solver():
    kinds = {spec.kind for spec in builtin_registry()}
    assert kinds == {eq.kind for eq in builtin_equations()}


def test_ftcs_reports_instability_beyond_its_limit():
    eq = builtin_equations().get("heat")
    problem = manufactured_problem(eq, "sin(pi*x)*exp(-t)", [(0, 1)], 1.0)
    with pytest.raises(SolverDiverged):
        builtin_registry().get("heat.ftcs").run(problem, 64, {"r": 0.7})


def test_self_convergence_without_exact_solution():
    problem = PDEProblem(kind="poisson", domain=[(0, 1)], source="exp(x)*sin(3*x)")
    study = convergence_study(builtin_registry().get("poisson.fd2"), problem, [16, 32, 64, 128])
    assert study.kind == "difference"
    assert study.final_order == pytest.approx(2.0, abs=0.2)
