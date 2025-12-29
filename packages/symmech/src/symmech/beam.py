"""
symmech.beam
============
Physical definition of structural elements.

This module combines:
1. Geometry (from symmech.core.Edge)
2. Section Properties (from symmech.section)
3. Material Properties (E, G)
4. Loads (Symbolic functions)

It prepares the mathematical objects required for Energy Methods (Castigliano).
"""

import sympy as sp
from sympy import Symbol, Matrix, simplify, sqrt, S
from .core import Node, Edge
from .section import SectionProfile, BoundarySection, MidplaneSection

class Material:
    """Isotropic material properties."""
    def __init__(self, name: str, E, G, rho=0):
        """
        Args:
            name: Identifier.
            E: Young's Modulus (symbolic or numeric).
            G: Shear Modulus (symbolic or numeric).
            rho: Density (optional, for mass/dynamics).
        """
        self.name = name
        self.E = sp.sympify(E)
        self.G = sp.sympify(G)
        self.rho = sp.sympify(rho)

    def __repr__(self):
        return f"Material({self.name}, E={self.E}, G={self.G})"


class BeamProperties:
    """
    Aggregates Section and Material data into stiffness terms.
    Can be initialized with a computed SectionProfile OR manual values (A, Ixx, etc).
    """
    def __init__(self, material: Material, section_obj=None, 
                 A=None, Ixx=None, Iyy=None, J=None, Ip=None):
        
        self.material = material
        
        # If a computed section object is provided, extract properties
        if section_obj:
            if isinstance(section_obj, BoundarySection):
                self.A = section_obj.area
                # Assuming Principal Axes alignment for V0.1
                ixx, iyy, _ = section_obj.moments_of_inertia 
                self.Ixx = ixx
                self.Iyy = iyy
                self.J = S(0) # Boundary doesn't calculate J well
            elif isinstance(section_obj, MidplaneSection):
                # Midplane is bad at Area/Inertia, good at J
                self.J = section_obj.torsion_constant_J
                self.A = S(0) 
                self.Ixx = S(0)
                self.Iyy = S(0)
            
            # Hybrid (User usually passes combined data or we merge later)
            # For V0.1, we allow manual overrides which is common in symbolic work
        
        # Manual overrides take precedence (or fill gaps)
        if A is not None: self.A = sp.sympify(A)
        if Ixx is not None: self.Ixx = sp.sympify(Ixx)
        if Iyy is not None: self.Iyy = sp.sympify(Iyy)
        if J is not None: self.J = sp.sympify(J)
        
        # Polar Moment (approx J for circular, Ixx+Iyy for others)
        self.Ip = Ip if Ip is not None else (self.Ixx + self.Iyy)

    @property
    def EIxx(self): return self.material.E * self.Ixx
    
    @property
    def EIyy(self): return self.material.E * self.Iyy
    
    @property
    def GJ(self): return self.material.G * self.J
    
    @property
    def EA(self): return self.material.E * self.A


class Beam:
    """
    A physical beam element connecting two nodes.
    
    Key Logic:
    - Orients the cross-section relative to the curve centerline using a 'Ref Vector'.
    - Defines the Local Coordinate System (t, n, b) along the curve.
    """
    def __init__(self, edge: Edge, props: BeamProperties, ref_vector: list = None):
        """
        Args:
            edge: The geometric curve.
            props: Stiffness properties.
            ref_vector: A vector [vx, vy, vz] that defines the 'Up' direction 
                        (local Y) for the cross-section. 
                        If None, defaults to Global Z (or Y if curve is vertical).
        """
        self.edge = edge
        self.props = props
        
        # Define Orientation
        # Tangent is local x
        self.t_vec = edge.tangent_vector
        
        # Handle Reference Vector (for local y/z orientation)
        if ref_vector is None:
            # Default logic: Use Global Z to define 'Up'
            v_ref = Matrix([0, 0, 1])
            # If beam is vertical (Tangent || Z), use Global Y
            if simplify(self.t_vec.cross(v_ref).norm()) == 0:
                v_ref = Matrix([0, 1, 0])
        else:
            v_ref = Matrix(ref_vector)
            
        # Normal (n) - Local Y
        # n = (ref x t) x t  <-- Projects ref vector perpendicular to tangent
        # Or more simply: n = unit(ref x t) ?? 
        # Standard Aerospace convention: 
        # local_y (n) is perpendicular to tangent, in plane of ref_vector.
        # local_z (b) = t x n
        
        # Let's calculate Binormal first using cross product
        b_unsized = self.t_vec.cross(v_ref)
        self.b_vec = simplify(b_unsized / sqrt(b_unsized.dot(b_unsized))) # Local Z
        
        self.n_vec = simplify(self.b_vec.cross(self.t_vec)) # Local Y

    def get_local_frame(self, t_param):
        """Returns rotation matrix [t, n, b] at parameter t."""
        T = self.t_vec.subs(self.edge.t, t_param)
        N = self.n_vec.subs(self.edge.t, t_param)
        B = self.b_vec.subs(self.edge.t, t_param)
        return Matrix([T, N, B])

    def apply_distributed_load(self, q_func):
        """
        q_func: Function taking (t) -> Matrix([qx, qy, qz]) in Global Frame.
        Stores load for solver integration.
        """
        self.distributed_load = q_func

    def __repr__(self):
        return f"Beam(Edge={self.edge}, Material={self.props.material.name})"
