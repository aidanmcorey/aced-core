"""
symmech.core
============
The Analytic Core of the SymMech library.

This module defines the fundamental geometric primitives:
1. Node: A symbolic point in 3D space.
2. Edge: A parametric curve connecting two Nodes with associated thickness.

Philosophy:
    All calculations are performed symbolically using SymPy. 
    Numerical evaluation happens only when explicitly requested (e.g., for meshing or plotting).
"""

import sympy as sp
from sympy import Symbol, Matrix, diff, integrate, sqrt, simplify, S

class Node:
    """
    Represents a point in 3D space (x, y, z).
    Coordinates can be numeric or symbolic variables.
    """
    def __init__(self, x, y, z, label: str = None):
        """
        Args:
            x, y, z: Coordinates (float, int, or sp.Symbol).
            label (str): Optional identifier for the node.
        """
        self.x = sp.sympify(x)
        self.y = sp.sympify(y)
        self.z = sp.sympify(z)
        self.label = label
        
        # Vector representation
        self.pos = Matrix([self.x, self.y, self.z])

    def distance_to(self, other: 'Node'):
        """Calculates Euclidean distance to another node."""
        return sqrt((self.x - other.x)**2 + 
                    (self.y - other.y)**2 + 
                    (self.z - other.z)**2)

    def __repr__(self):
        coord_str = f"({self.x}, {self.y}, {self.z})"
        if self.label:
            return f"Node<{self.label}: {coord_str}>"
        return f"Node{coord_str}"


class Edge:
    """
    Represents a parametric curve r(t) connecting two Nodes.
    
    Attributes:
        node_start (Node): The starting point (t=0).
        node_end (Node): The ending point (t=1).
        thickness (sp.Expr): Thickness of the edge (used for Section definition).
        r_func (Matrix): The symbolic position vector r(t).
    """
    def __init__(self, node_start: Node, node_end: Node, thickness=0, curve_func=None):
        """
        Args:
            node_start (Node): Start point.
            node_end (Node): End point.
            thickness: Physical thickness (float or Symbol).
            curve_func (Matrix, optional): A SymPy Matrix([x(t), y(t), z(t)]) dependent 
                                         on symbolic variable 't'. 
                                         If None, defaults to a straight line.
        """
        self.node_start = node_start
        self.node_end = node_end
        self.thickness = sp.sympify(thickness)
        
        # Standard parametric variable
        self.t = Symbol('t')

        if curve_func is None:
            # Default: Linear interpolation (Straight Line)
            # r(t) = P_start + t * (P_end - P_start)
            d_vec = self.node_end.pos - self.node_start.pos
            self.r_func = self.node_start.pos + self.t * d_vec
        else:
            # Custom curved geometry
            self.r_func = curve_func
            # Validation: We could assert that curve_func.subs(t, 0) == node_start 
            # but we assume user intent for advanced geometries.

    @property
    def tangent_vector(self):
        """
        Symbolic Tangent Vector T(t).
        T = dr/dt / |dr/dt|
        """
        dr_dt = diff(self.r_func, self.t)
        mag = sqrt(dr_dt.dot(dr_dt))
        return simplify(dr_dt / mag)

    @property
    def normal_vector(self):
        """
        Symbolic Principal Normal Vector N(t).
        N = dT/dt / |dT/dt|
        
        Note: For straight lines, dT/dt is 0, making N undefined (Singularity).
        In Beam theory, if curvature is 0, the orientation is defined by a 
        reference vector, not the Frenet normal.
        """
        T = self.tangent_vector
        dT_dt = diff(T, self.t)
        mag = sqrt(dT_dt.dot(dT_dt))
        
        # Handle the straight line case gracefully
        if simplify(mag) == 0:
            return Matrix([0, 0, 0]) # Undefined/Zero curvature
            
        return simplify(dT_dt / mag)

    @property
    def curvature(self):
        """
        Symbolic Curvature kappa(t).
        kappa = |r' x r''| / |r'|^3
        """
        r_prime = diff(self.r_func, self.t)
        r_double_prime = diff(r_prime, self.t)
        
        cross_prod = r_prime.cross(r_double_prime)
        numerator = sqrt(cross_prod.dot(cross_prod))
        denominator = (r_prime.dot(r_prime))**(S(3)/2)
        
        return simplify(numerator / denominator)

    @property
    def arc_length_expr(self):
        """
        The differential arc length 'ds' expression in terms of 'dt'.
        ds/dt = |dr/dt|
        """
        dr_dt = diff(self.r_func, self.t)
        return simplify(sqrt(dr_dt.dot(dr_dt)))

    def calculate_length(self, t0=0, t1=1):
        """
        Performs the symbolic integration to find total length.
        L = integral(|dr/dt|) from t0 to t1.
        """
        ds = self.arc_length_expr
        return integrate(ds, (self.t, t0, t1))

    def __repr__(self):
        return f"Edge(Start={self.node_start.label}, End={self.node_end.label}, thk={self.thickness})"
