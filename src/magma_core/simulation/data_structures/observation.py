from dataclasses import dataclass, field
from typing import Dict

from magma_core.simulation.data_structures.tools import ToolBatchContext

@dataclass
class Observation:
    """ A dataclass to contain the observation data. """
    selected_robot_name: str
    task_attributes:Dict
    maniskill_obs:Dict
    add_constants: Dict = field(default_factory = lambda: ({}))
    tool_batch_context: ToolBatchContext = field(default_factory=ToolBatchContext)
