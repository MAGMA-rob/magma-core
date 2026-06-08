from ..data_structures import ToolStatus, ToolExecution, Situation
from typing import Dict, List, Optional, Tuple, Union

from .runtime_randomizer import RuntimeRandomizer
from .spec_generator import SpecGenerator
from ..tasks import BaseTask

class Randomizer():
    """
    The Randomizer is reponsible to manage different runtime randomizer allowing to switch betweens different randomization instances.

    In MAGMA-GEN, there is generally one instance computed at the very beginning of the task.
    In MAGMA-BENCH, there are multiples randomizer instances depending on each task variation.
    """

    variations : List[RuntimeRandomizer]
    variation_idx : int
    nb_variation : int

    def __init__(self, nb_variation : int, seed : Optional[int]) -> None:
        if nb_variation < 1:
            raise ValueError("Nb variation must be >= 1")
        self.variation_idx = 0
        self.generator = SpecGenerator(seed)
        self.nb_variation = nb_variation

    def initialize_randomizer(self, task_ref : BaseTask):
        if self.nb_variation == 1:
            specs = [self.generator.generate_random_spec(task_ref)]
        else:
            specs = self.generator.generate_N_spec(task_ref, self.nb_variation)
        
        self.variations = []
        for spec in specs:
            self.variations.append(
                RuntimeRandomizer(spec)
            )
        self.variation_idx = 0

    def set_variation_index(self,idx : int):
        if idx < 0 or idx > self.nb_variation:
            raise RuntimeError(f"The passed idx ({idx}) is outside the variation range (0,{self.nb_variation}-1)")
        
        self.variation_idx = idx

    # Map to Runtime Randomizer

    def traduce_end_eval(self, out: Dict[int, Dict]) -> Dict[int, Dict]:
        return self.variations[self.variation_idx].traduce_end_eval(out)
    
    def traduce_end_gen(self, out: Dict[int, ToolStatus]) -> Dict[int, ToolStatus]:
        return self.variations[self.variation_idx].traduce_end_gen(out)
    
    def traduce_attributes_to_env(self, text: str) -> str:
        return self.variations[self.variation_idx].traduce_attributes_to_env(text)

    def traduce_attributes_to_llm(self, text: str) -> str:
        """Replace real attribute names with the randomized public names."""
        return self.variations[self.variation_idx].traduce_attributes_to_llm(text)
    
    def map_tool_call(
        self,
        func_name: str,
        params: Dict,
    ) -> Union[ToolExecution, Tuple[str, Dict]]:
        return self.variations[self.variation_idx].map_tool_call(func_name,params)
    
    def get_randomized_situation(self, original_situation: Situation) -> Situation:
        return self.variations[self.variation_idx].get_randomized_situation(original_situation)
    
    def get_tools(self) -> List[Dict]:
        return self.variations[self.variation_idx].get_tools()
    
    def get_randomized_attributes(self, attributes: Dict) -> Dict:
        return self.variations[self.variation_idx].get_randomized_attributes(attributes)
    
    def get_tmp_translation(self, node_id : int):
        return self.variations[self.variation_idx].tmp_translation[node_id]
    
    def set_tmp_translation(self, node_id : int, name):
        self.variations[self.variation_idx].tmp_translation[node_id] = name