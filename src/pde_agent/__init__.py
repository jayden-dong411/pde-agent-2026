"""pde-agent: an agent that solves PDEs and keeps getting better at it."""

from .agent import PDEAgent, SolveReport
from .equations import Equation, builtin_equations, manufactured_problem
from .ode import ODEProblem
from .problem import PDEProblem

__version__ = "1.0.0"

__all__ = ["Equation", "ODEProblem", "PDEAgent", "PDEProblem", "SolveReport", "builtin_equations", "manufactured_problem"]
