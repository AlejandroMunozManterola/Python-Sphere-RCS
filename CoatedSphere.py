import numpy as np
from DielectricMaterial import DielectricMaterial as DM


class CoatedSphere:
    """
    Represents a dielectric-coated sphere (concentric core + shell).

    The geometry is:
        Core:  radius = core_radius,   material = core_material
        Shell: inner radius = core_radius,
               outer radius = core_radius + shell_width,
               material = shell_material

    Total outer radius = core_radius + shell_width
    """

    def __init__(self, core_material, shell_material, core_radius, shell_width):
        """
        core_material:   DielectricMaterial — the inner sphere
        shell_material:  DielectricMaterial — the coating layer
        core_radius:     float — radius of the core (m)
        shell_width:     float — thickness of the shell layer (m)
        """
        self.core_material = core_material
        self.shell_material = shell_material
        self.core_radius = core_radius
        self.shell_width = shell_width
        self.radius = core_radius + shell_width  # outer radius
