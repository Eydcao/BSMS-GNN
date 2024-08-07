from .airfoil import airfoilDataPipe
from .cylinder_flow import cylinderDataPipe

DATSET_HANDLER = {"airfoil": airfoilDataPipe, "cylinder_flow": cylinderDataPipe}
