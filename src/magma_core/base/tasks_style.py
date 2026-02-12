from enum import Enum

class TaskStyle(Enum):
    """
    TaskStyle allows to define custom criteria that represents what tasks are evaluating.

    This serves to our curriculum agents to foud close tasks to the current ones or to cible the weakpoint or the current model.
    """

    LONG_STAGE = "long_stage" #  Tasks where there is one instruction covering multiple stage in a row.
    
    MEMORY_COHERENCE = "memory_coherence" # Tasks where we are manipulating constraint to verify the persistent state tracking of model.

    CONSTRAINED = "constrained" # Tasks where we give a lot of constraints on the execution of the task (execution, orders, user preference, safety).

    MULTI_ROBOT = "multi_robot" # Tasks with multiple robots.

    DETECTION_TASK = "detection_task" # Tasks where the robot has not directly access to objects in the scene, but must use a detection function.

    COLOR_REASONING = "color_reasoning" # Tasks where the model must reason about colors.

    INTERUPTION = "interuption" # Tasks where the user interupt an ongoing task to ask for an urgent action, before continuing the other one.