"""
symmech.viz
===========
Visualization Engine using Plotly.
"""

import sympy as sp
import numpy as np
import plotly.graph_objects as go
from .beam import Beam
from .core import Node
from .section import BoundarySection, MidplaneSection

class SectionVisualizer:
    """
    Visualizes 2D Cross-Sections.
    """
    def __init__(self, section, title="Section Profile"):
        self.section = section
        self.fig = go.Figure()
        self.fig.update_layout(
            title=title,
            yaxis=dict(scaleanchor="x", scaleratio=1), # Equal Aspect Ratio 2D
            xaxis_title="Y (Width)",
            yaxis_title="Z (Height)"
        )

    def plot_2d(self):
        """Generates the 2D plot of the cross-section."""
        
        edges_to_plot = []
        
        # Determine edges based on section type
        if isinstance(self.section, BoundarySection):
            edges_to_plot = self.section.edges
        elif isinstance(self.section, MidplaneSection):
            edges_to_plot = self.section.edges
            
        for edge in edges_to_plot:
            # Parametric variable t
            # r_func is likely 3D [x, y, z], but for section in Y-Z plane:
            # We assume section is defined in local frame where x=0.
            
            # Lambdify geometry
            r_num = sp.lambdify(edge.t, edge.r_func, modules='numpy')
            t_vals = np.linspace(0, 1, 20)
            
            coords = []
            for t in t_vals:
                pt = r_num(t) # [x, y, z]
                coords.append(np.array(pt).flatten())
            coords = np.array(coords)
            
            # For a section, we usually plot Y vs Z (or Index 1 vs Index 2)
            # assuming the profile was defined in the YZ plane.
            ys = coords[:, 0] # Standard Convention: Profile defined in X-Y or Y-Z?
            zs = coords[:, 1] # Let's assume input was 2D [y, z] or 3D [0, y, z]
            
            # If coordinates are 3D and x is constant 0
            if coords.shape[1] == 3:
                ys = coords[:, 1]
                zs = coords[:, 2]
            
            self.fig.add_trace(go.Scatter(
                x=ys, y=zs,
                mode='lines',
                line=dict(color='blue', width=2)
            ))
            
    def show(self):
        self.plot_2d()
        self.fig.show()


class BeamVisualizer:
    """
    Manager for 3D rendering of the structural assembly.
    """
    def __init__(self, title="SymMech Model"):
        self.fig = go.Figure()
        self.fig.update_layout(
            title=title,
            scene=dict(
                aspectmode='data',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )

    def add_beam(self, beam: Beam, samples=20, show_frames=True, show_thickness=False):
        """
        Args:
            show_thickness: If True, attempts to render a simplified extrusion 
                            of the cross-section along the path.
        """
        # ... (Existing Centerline Logic from previous step) ...
        r_numeric = sp.lambdify(beam.edge.t, beam.edge.r_func, modules='numpy')
        t_vals = np.linspace(0, 1, samples)
        
        centerline_coords = []
        for t in t_vals:
            pt = r_numeric(t)
            centerline_coords.append(np.array(pt).flatten())
        centerline_coords = np.array(centerline_coords)
        
        # 1. Plot Centerline
        self.fig.add_trace(go.Scatter3d(
            x=centerline_coords[:, 0], y=centerline_coords[:, 1], z=centerline_coords[:, 2],
            mode='lines',
            name=f"Beam {beam.edge.node_start.label}",
            line=dict(width=5, color='black')
        ))

        # 2. (Optional) Extrude Section
        # This requires transforming the 2D section shape into the 3D local frame [T, N, B]
        # at every step 't' along the curve.
        if show_thickness and beam.props and hasattr(beam.props, 'A'): # Basic check
             # Placeholder for V0.2: Full extrusion is computationally heavy.
             # V0.1: We just show the frames.
             pass

        # 3. Plot Frames (Local Axes)
        if show_frames:
            t_num = sp.lambdify(beam.edge.t, beam.t_vec, modules='numpy')
            n_num = sp.lambdify(beam.edge.t, beam.n_vec, modules='numpy')
            b_num = sp.lambdify(beam.edge.t, beam.b_vec, modules='numpy')
            
            frame_indices = np.linspace(0, samples-1, 5, dtype=int)
            
            for idx in frame_indices:
                t_val = t_vals[idx]
                origin = centerline_coords[idx]
                
                T = np.array(t_num(t_val)).flatten()
                N = np.array(n_num(t_val)).flatten()
                B = np.array(b_num(t_val)).flatten()
                
                scale = (np.max(centerline_coords) - np.min(centerline_coords)) * 0.1
                if scale == 0: scale = 1.0

                self._add_arrow(origin, T, 'red', scale, "t")   # Tangent
                self._add_arrow(origin, N, 'green', scale, "n") # Normal (Local Y)
                self._add_arrow(origin, B, 'blue', scale, "b")  # Binormal (Local Z)

    def add_node(self, node: Node):
        x, y, z = float(node.x), float(node.y), float(node.z)
        self.fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers+text',
            text=[node.label],
            marker=dict(size=4, color='black')
        ))

    def _add_arrow(self, origin, vector, color, scale, name):
        end = origin + vector * scale
        self.fig.add_trace(go.Scatter3d(
            x=[origin[0], end[0]], y=[origin[1], end[1]], z=[origin[2], end[2]],
            mode='lines', showlegend=False, line=dict(color=color, width=3)
        ))
        
    def show(self):
        self.fig.show()
