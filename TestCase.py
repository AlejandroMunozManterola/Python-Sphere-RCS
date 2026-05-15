import numpy as np
from DielectricMaterial import *
from CoatedSphere import *
from src import *

class TestCase:
    '''
    Defines a sphere (or coated sphere) for RCS testing.

    For a simple sphere:
        TestCase(radius, sphere_material, background_material)

    For a coated sphere:
        TestCase(coated_sphere, background_material)
        where coated_sphere is a CoatedSphere object
    '''

    def __init__(self, *args, **kwargs):
        # Detect whether first arg is a CoatedSphere or a radius (numeric)
        if len(args) >= 1 and isinstance(args[0], CoatedSphere):
            # Coated sphere form: TestCase(coated_sphere, background_material)
            self.coated_sphere = args[0]
            self.background_material = args[1] if len(args) > 1 else None
            self.radius = self.coated_sphere.radius
            self.sphere_material = None
        else:
            # Simple sphere form: TestCase(radius, sphere_material, background_material)
            self.radius = args[0]
            self.sphere_material = args[1]
            self.background_material = args[2] if len(args) > 2 else None
            self.coated_sphere = None

class TestParameters:
    '''
    Groups together sensor location and frequencies for testing.
    '''

    def __init__(self, sensor_location, frequency):
        self.sensor_location = sensor_location
        self.frequency = frequency
