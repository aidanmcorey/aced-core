"""
symmech.solver
==============
Energy Methods Engine (Castigliano's Method).
"""

import sympy as sp
from sympy import Symbol, Matrix, integrate, diff, simplify, S
from .beam import Beam

class PointLoad:
    """A concentrated force and/or moment applied at a specific location."""
    def __init__(self, node, force_vec=None, moment_vec=None):
        self.node = node
        self.F = Matrix(force_vec) if force_vec else Matrix([0, 0, 0])
        self.M = Matrix(moment_vec) if moment_vec else Matrix([0, 0, 0])

    def __repr__(self):
        return f"Load@{self.node.label}(F={self.F.T}, M={self.M.T})"


class CastiglianoSolver:
    """
    Solves for unknown reactions or displacements using Strain Energy.
    """
    def __init__(self):
        self.beams = []
        self.loads = []
        self.reaction_vars = [] 

    def add_beam(self, beam: Beam):
        self.beams.append(beam)

    def add_load(self, load: PointLoad):
        self.loads.append(load)
    
    def create_reaction(self, name: str):
        var = Symbol(name)
        self.reaction_vars.append(var)
        return var

    def _get_internal_loads(self, beam: Beam, t):
        """
        Calculates internal forces/moments at parameter t on the given beam.
        Includes contribution from:
        1. Discrete Point Loads (Summation)
        2. Distributed Loads (Integration)
        """
        # 1. Current Position r(t) (The 'Cut')
        r_cut = beam.edge.r_func.subs(beam.edge.t, t)
        
        # Initialize Totals (Global Frame)
        F_total = Matrix([0, 0, 0])
        M_total = Matrix([0, 0, 0])
        
        # --- A. Process Point Loads ---
        for load in self.loads:
            # Assumption: All loads are downstream (cantilever logic for V0.1)
            
            # Force
            F_total += load.F
            
            # Moment: M_load + r_arm x F_load
            r_load = load.node.pos
            r_arm = r_load - r_cut
            M_total += load.M + r_arm.cross(load.F)

        # --- B. Process Distributed Loads ---
        if hasattr(beam, 'distributed_load') and beam.distributed_load:
            # We must integrate from current 't' to end of beam '1'
            tau = Symbol('tau') # Dummy integration variable
            
            # 1. Get Load Vector q(tau) at integration point
            q_vec = beam.distributed_load(tau)
            
            # 2. Get Position r(tau) at integration point
            r_tau = beam.edge.r_func.subs(beam.edge.t, tau)
            
            # 3. Differential arc length ds/dtau
            # We need the magnitude of dr/dtau
            dr_dtau = diff(r_tau, tau)
            ds_dtau = sqrt(dr_dtau.dot(dr_dtau))
            
            # 4. Integrate Force: Integral( q(tau) * ds )
            # We integrate each component of the vector
            F_dist = integrate(q_vec * ds_dtau, (tau, t, 1))
            F_total += F_dist
            
            # 5. Integrate Moment: Integral( (r(tau) - r(t)) x q(tau) * ds )
            r_arm_dist = r_tau - r_cut
            M_integrand = r_arm_dist.cross(q_vec) * ds_dtau
            
            M_dist = integrate(M_integrand, (tau, t, 1))
            M_total += M_dist

        # --- C. Project into Local Beam Frame ---
        # Get frame vectors at t
        T_vec = beam.t_vec.subs(beam.edge.t, t)
        N_vec = beam.n_vec.subs(beam.edge.t, t)
        B_vec = beam.b_vec.subs(beam.edge.t, t)
        
        return {
            'N': simplify(F_total.dot(T_vec)), 
            'Vn': simplify(F_total.dot(N_vec)), 
            'Vb': simplify(F_total.dot(B_vec)),
            'T': simplify(M_total.dot(T_vec)),
            'Mn': simplify(M_total.dot(N_vec)), 
            'Mb': simplify(M_total.dot(B_vec))
        }

    def compute_total_energy(self):
        """Integrates strain energy density over all beams."""
        U_total = S(0)
        
        for beam in self.beams:
            # Internal loads as function of t
            loads = self._get_internal_loads(beam, beam.edge.t)
            ds = beam.edge.arc_length_expr
            
            # Energy Densities (Standard Beam Theory)
            u_axial = (loads['N']**2) / (2 * beam.props.EA)
            u_torsion = (loads['T']**2) / (2 * beam.props.GJ)
            u_bend_n = (loads['Mn']**2) / (2 * beam.props.EIyy)
            u_bend_b = (loads['Mb']**2) / (2 * beam.props.EIxx)
            
            density = u_axial + u_torsion + u_bend_n + u_bend_b
            
            # Integrate over beam domain [0, 1]
            segment_energy = integrate(density * ds, (beam.edge.t, 0, 1))
            U_total += segment_energy
            
        return simplify(U_total)

    def solve_statically_indeterminate(self):
        """Solves for unknown reaction variables (dL/dR = 0)."""
        U = self.compute_total_energy()
        equations = [diff(U, R) for R in self.reaction_vars]
        return sp.solve(equations, self.reaction_vars)

    def get_deflection(self, force_var):
        """Calculates deflection (delta = dU/dP)."""
        U = self.compute_total_energy()
        return simplify(diff(U, force_var))
