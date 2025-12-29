"""
symmech.beam
============
Physical definition of structural elements.
"""

import sympy as sp
from sympy import Symbol, Matrix, simplify, sqrt, S
from .core import Node, Edge
from .section import SectionProfile, BoundarySection, MidplaneSection

class Material:
    """Isotropic material properties."""
    def __init__(self, name: str, E, G, rho=0):
        self.name = name
        self.E = sp.sympify(E)
        self.G = sp.sympify(G)
        self.rho = sp.sympify(rho)

    def __repr__(self):
        return f"Material({self.name}, E={self.E}, G={self.G})"


class BeamProperties:
    """Aggregates Section and Material data into stiffness terms."""
    def __init__(self, material: Material, section_obj=None, 
                 A=None, Ixx=None, Iyy=None, J=None, Ip=None):
        
        self.material = material
        
        # Extract properties from Section Object if provided
        if section_obj:
            if isinstance(section_obj, BoundarySection):
                self.A = section_obj.area
                ixx, iyy, _ = section_obj.moments_of_inertia 
                self.Ixx = ixx
                self.Iyy = iyy
                self.J = S(0) 
            elif isinstance(section_obj, MidplaneSection):
                self.J = section_obj.torsion_constant_J
                self.A = S(0) 
                self.Ixx = S(0)
                self.Iyy = S(0)
            
        # Manual overrides
        if A is not None: self.A = sp.sympify(A)
        if Ixx is not None: self.Ixx = sp.sympify(Ixx)
        if Iyy is not None: self.Iyy = sp.sympify(Iyy)
        if J is not None: self.J = sp.sympify(J)
        
        self.Ip = Ip if Ip is not None else (self.Ixx + self.Iyy)

    @property
    def EIxx(self): return self.material.E * self.Ixx
    @property
    def EIyy(self): return self.material.E * self.Iyy
    @property
    def GJ(self): return self.material.G * self.J
    @property
    def EA(self): return self.material.E * self.A


class DistributedLoad:
    """
    Represents a continuous force distribution q(t) applied along the beam.
    """
    def __init__(self, q_func, label="q(t)"):
        """
        Args:
            q_func: A callable taking parameter 't' and returning a SymPy Matrix([qx, qy, qz])
                    in the Global Coordinate System.
            label:  Optional string identifier.
        """
        self.q_func = q_func
        self.label = label

    def value_at(self, t):
        """Returns the global force vector at parameter t."""
        return self.q_func(t)

    def __repr__(self):
        return f"DistributedLoad<{self.label}>"


class Beam:
    """
    A physical beam element connecting two nodes.
    """
    def __init__(self, edge: Edge, props: BeamProperties, ref_vector: list = None):
        self.edge = edge
        self.props = props
        
        # Container for distributed loads
        self.distributed_loads: list[DistributedLoad] = []
        
        # Define Orientation
        self.t_vec = edge.tangent_vector
        
        # Handle Reference Vector (for local y/z orientation)
        if ref_vector is None:
            v_ref = Matrix([0, 0, 1])
            if simplify(self.t_vec.cross(v_ref).norm()) == 0:
                v_ref = Matrix([0, 1, 0])
        else:
            v_ref = Matrix(ref_vector)
            
        b_unsized = self.t_vec.cross(v_ref)
        self.b_vec = simplify(b_unsized / sqrt(b_unsized.dot(b_unsized))) # Local Z
        self.n_vec = simplify(self.b_vec.cross(self.t_vec)) # Local Y

    def add_distributed_load(self, dist_load: DistributedLoad):
        """Attaches a distributed load object to this beam."""
        self.distributed_loads.append(dist_load)

    def __repr__(self):
        return f"Beam(Edge={self.edge}, Loads={len(self.distributed_loads)})"
