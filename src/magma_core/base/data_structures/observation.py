from dataclasses import dataclass, field
from typing import Dict

@dataclass
class Observation:
    """ A dataclass to contain the observation data. """
    selected_robot_name: str
    task_attributes:Dict
    maniskill_obs:Dict
    add_constants: Dict = field(default_factory = lambda: ({}))