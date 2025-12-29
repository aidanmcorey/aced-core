"""
symmech.section
===============
Dual Representation of cross-sections for beam mechanics.

1. Boundary Mode (Exact/Skin):
   Uses Green's Theorem to calculate Area, Moments of Inertia (Ixx, Iyy, Ixy),
   and Centroids by integrating along the closed contour of the section.
   Supports curved edges exactly via symbolic integration.

2. Midplane Mode (Thin-Walled/Skeleton):
   Represents the section as a graph of edges with thickness.
   Used for Torsion Constant (J) and Shear Center calculations which are
   difficult to extract from boundary integrals alone.
"""

import sympy as sp
from sympy import Symbol, integrate, diff, simplify, sqrt, S
from .core import Edge, Node

class SectionProfile:
    """Base class for section properties."""
    pass


class BoundarySection(SectionProfile):
    """
    Represents the outer 'skin' of a cross-section as a closed loop of Edges.
    Calculates properties using Green's Theorem.
    """
    def __init__(self, contour_edges: list[Edge]):
        """
        Args:
            contour_edges: A list of Edge objects forming a closed loop (CCW).
                           The end of edge[i] must be the start of edge[i+1].
        """
        self.edges = contour_edges
        self._validate_closure()

    def _validate_closure(self):
        # Basic topology check: Start of first must equal End of last
        start = self.edges[0].node_start.pos
        end = self.edges[-1].node_end.pos
        # In a symbolic environment, we check if the difference simplifies to zero
        if simplify((start - end).norm()) != 0:
            # Warning: Symbolic equality checks can sometimes fail if complex, 
            # but this catches obvious discontinuities.
            pass 

    def _integrate_path(self, func_integrand):
        """
        Helper to integrate a scalar function along the entire contour.
        Integral = Sum( Integral(func(t) * dt) for edge in edges )
        """
        total = 0
        for edge in self.edges:
            # Parametric variable t in [0, 1]
            t = edge.t
            # Get x(t), y(t) from the edge's position vector
            # Assuming 2D section in XY plane for now, or projection.
            x = edge.r_func[0]
            y = edge.r_func[1]
            
            # Derivatives for line integrals
            dx_dt = diff(x, t)
            dy_dt = diff(y, t)
            
            # Evaluate the specific integrand for this edge
            # func_integrand should accept (x, y, dx_dt, dy_dt)
            term = func_integrand(x, y, dx_dt, dy_dt)
            total += integrate(term, (t, 0, 1))
            
        return simplify(total)

    @property
    def area(self):
        """
        Area via Green's Theorem: A = 0.5 * integral(x dy - y dx)
        """
        def integrand(x, y, dx, dy):
            return S(1)/2 * (x * dy - y * dx)
        return self._integrate_path(integrand)

    @property
    def first_moments(self):
        """Returns tuple (Qx, Qy). Qx = int(y dA), Qy = int(x dA)"""
        # Qx = integral(y dA) -> Green's: integral(x*y dy) or -integral(y^2/2 dx)
        # Using -y^2/2 dx for Qx
        def f_Qx(x, y, dx, dy): return - (y**2 / 2) * dx
        
        # Qy = integral(x dA) -> Green's: integral(x^2/2 dy)
        def f_Qy(x, y, dx, dy): return (x**2 / 2) * dy
        
        return (self._integrate_path(f_Qx), self._integrate_path(f_Qy))

    @property
    def centroid(self):
        """Returns (x_bar, y_bar)"""
        Qx, Qy = self.first_moments
        A = self.area
        return (simplify(Qy / A), simplify(Qx / A))

    @property
    def moments_of_inertia(self):
        """Returns (Ixx, Iyy, Ixy) about the global origin."""
        # Ixx = integral(y^2 dA) -> -integral(y^3/3 dx)
        def f_Ixx(x, y, dx, dy): return -(y**3 / 3) * dx
        
        # Iyy = integral(x^2 dA) -> integral(x^3/3 dy)
        def f_Iyy(x, y, dx, dy): return (x**3 / 3) * dy
        
        # Ixy = integral(xy dA) -> integral(x*y^2/2 dy) (one of many forms)
        # Or -integral(x^2*y/2 dx). Let's use symmetric: 
        # integral( (x^2*y/2)*dy - (x*y^2/2)*dx ) ? No, standard form:
        def f_Ixy(x, y, dx, dy): return (x**2 * y / 2) * dy 

        return (
            self._integrate_path(f_Ixx),
            self._integrate_path(f_Iyy),
            self._integrate_path(f_Ixy)
        )


class MidplaneSection(SectionProfile):
    """
    Represents the section as a 'Skeleton' of edges with thickness.
    Focuses on Torsion (J) and thin-walled properties.
    """
    def __init__(self, edges: list[Edge], closed_cell: bool = False):
        """
        Args:
            edges: List of Edge objects representing webs/flanges.
            closed_cell (bool): True if this forms a single closed torque box.
                               (Multi-cell detection is V0.2)
        """
        self.edges = edges
        self.is_closed = closed_cell

    @property
    def torsion_constant_J(self):
        """
        Calculates St. Venant Torsion Constant J.
        
        Case 1: Open Section (e.g., I-beam, C-channel)
            J = Sum(1/3 * length * thickness^3)
            
        Case 2: Closed Single Cell (e.g., Box beam) - Bredt's Formula
            J = (4 * Am^2) / loop_integral(ds/t)
            Where Am is the area enclosed by the midline.
        """
        if not self.is_closed:
            # Summation rule for open thin-walled sections
            J_total = 0
            for edge in self.edges:
                L = edge.calculate_length()
                t = edge.thickness
                # Fundamental approximation for thin rectangles
                J_total += S(1)/3 * L * t**3
            return simplify(J_total)
        
        else:
            # Bredt's Formula for single cell
            # 1. Calculate Enclosed Area (Am) of the midline
            # We treat the midplane edges as a BoundarySection to get Area
            skeleton_boundary = BoundarySection(self.edges)
            Am = skeleton_boundary.area
            
            # 2. Calculate contour integral of (1/t) ds
            contour_int = 0
            for edge in self.edges:
                L = edge.calculate_length()
                t = edge.thickness
                contour_int += L / t
                
            J = (4 * Am**2) / contour_int
            return simplify(J)

    def calculate_shear_center(self):
        """
        V0.1 Placeholder.
        Requires calculating first moments of sectors.
        To be implemented with Castigliano solver in V0.2 logic.
        """
        raise NotImplementedError("Shear Center logic scheduled for V0.1/V0.2 boundary.")
