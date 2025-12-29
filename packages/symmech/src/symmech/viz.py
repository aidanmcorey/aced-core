"""
symmech.viz
===========
Visualization Engine using Plotly.

Functionality:
1. Renders symbolic Beam centerlines in 3D.
2. Visualizes Local Coordinate Systems (Tangent, Normal, Binormal).
3. (Future) Plots stress diagrams and deflected shapes.

Note:
    This module performs 'Discretization on Demand'. It converts exact 
    symbolic curves into arrays of points solely for rendering.
"""

import sympy as sp
import numpy as np
import plotly.graph_objects as go
from .beam import Beam
from .core import Node

class BeamVisualizer:
    """
    Manager for 3D rendering of the structural assembly.
    """
    def __init__(self, title="SymMech Model"):
        self.fig = go.Figure()
        self.title = title
        
        # Layout settings for engineering view (Equal aspect ratio)
        self.fig.update_layout(
            title=title,
            scene=dict(
                aspectmode='data', # Vital for true-to-scale structural viewing
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        self.beams = []

    def add_beam(self, beam: Beam, samples=20, show_frames=True):
        """
        Adds a beam to the plot.
        
        Args:
            beam: The Beam object.
            samples: Number of points to evaluate along the curve.
            show_frames: If True, draws RGB arrows for local (t,n,b) axes.
        """
        # 1. Lambdify the Position Vector r(t)
        # Convert symbolic matrix to a fast numeric function
        # r_func is Matrix([x, y, z])
        r_numeric = sp.lambdify(beam.edge.t, beam.edge.r_func, modules='numpy')
        
        # 2. Generate parametric domain
        t_vals = np.linspace(0, 1, samples)
        
        # 3. Evaluate points
        # If r_numeric returns a single list/array, stack them
        # Note: lambdify behavior varies if expressions are constant. 
        # We wrap in a safe evaluator.
        coords = []
        for t in t_vals:
            # Result is a 3x1 array or list
            pt = r_numeric(t)
            # Handle potential shape nuances from sympy
            coords.append(np.array(pt).flatten())
            
        coords = np.array(coords) # Shape (samples, 3)
        xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
        
        # 4. Plot Centerline
        self.fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='lines',
            name=f"Beam: {beam.edge.node_start.label}-{beam.edge.node_end.label}",
            line=dict(width=5, color='blue')
        ))
        
        # 5. Plot Orientation Frames (Local Axes)
        if show_frames:
            # Create numeric functions for T, N, B
            t_num = sp.lambdify(beam.edge.t, beam.t_vec, modules='numpy')
            n_num = sp.lambdify(beam.edge.t, beam.n_vec, modules='numpy')
            b_num = sp.lambdify(beam.edge.t, beam.b_vec, modules='numpy')
            
            # Subsample for frames (don't draw 20 frames, maybe just 3 or 5)
            frame_indices = np.linspace(0, samples-1, 5, dtype=int)
            
            for idx in frame_indices:
                t_val = t_vals[idx]
                origin = coords[idx]
                
                # Evaluate vectors
                # Note: If curvature is 0, N/B might be constant vectors
                T = np.array(t_num(t_val)).flatten()
                N = np.array(n_num(t_val)).flatten()
                B = np.array(b_num(t_val)).flatten()
                
                scale = (np.max(coords) - np.min(coords)) * 0.1 # Dynamic scaling
                if scale == 0: scale = 1.0

                self._add_arrow(origin, T, 'red', scale, "t")
                self._add_arrow(origin, N, 'green', scale, "n")
                self._add_arrow(origin, B, 'blue', scale, "b")

    def add_node(self, node: Node):
        """Adds a marker for a specific node."""
        # Evaluate if symbolic
        x = float(node.x.evalf())
        y = float(node.y.evalf())
        z = float(node.z.evalf())
        
        self.fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers+text',
            name=node.label,
            text=[node.label],
            textposition="top center",
            marker=dict(size=5, color='black')
        ))

    def _add_arrow(self, origin, vector, color, scale, name):
        """Helper to draw a 3D arrow (Cone + Line)."""
        end = origin + vector * scale
        
        # Draw Line
        self.fig.add_trace(go.Scatter3d(
            x=[origin[0], end[0]],
            y=[origin[1], end[1]],
            z=[origin[2], end[2]],
            mode='lines',
            showlegend=False,
            line=dict(color=color, width=3)
        ))
        
        # Cone for arrowhead (optional, keeping it simple with lines for performance)

    def show(self):
        """Render the interactive plot."""
        self.fig.show()
