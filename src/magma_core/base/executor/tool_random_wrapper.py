# Author : Loan BERNAT
# 2 BSD

from ..data_structures import (
    ToolExecution, ToolInfos, ToolStatus,
    Situation, EmptyInstruction, StatusReturn,
    UserInstruction, TemplateInstruction
)
from ..tasks import BaseTask

import re, random, yaml, json
from typing import List, Dict, Tuple, Union, Optional

class ToolRandomizerWrapper():
    # news tools
    tools : List[Dict]
    tools_equivalence : Dict # randomized : real

    # news attributes
    attributes_equivalence : Dict # Stock pre-computed attributes element associations  real : randomized
    reversed_attributes_equivalence: Dict # randomized : real

    # Updated memory and instruction
    memory : List[str]
    instruction : str

    # temp storing
    tmp_translation : Dict

    def __init__(self, seed : Optional[int] = None) -> None:
        if seed:
            random.seed(seed)

        self.attributes_config = {}
        self.tools_equivalence = {}
        self.attributes_equivalence = {}
        self.reversed_attributes_equivalence = {}
        self.original_attributes_listed = []


    def initialize_randomizer(self, task_ref : BaseTask):
        if task_ref.randomized_config_path == "":
            raise TypeError(f"The {task_ref.name} has no randomized file path set")
        with open(task_ref.randomized_config_path, "r") as f:
            config = yaml.safe_load(f)

        rt = config.get("tools",None)
        ra = config.get("attributes",{})
        
        if rt:
            self._randomize_tool(rt, task_ref.get_tools())
        else:
            raise ValueError("No tool randomization detected.")

        if ra:
            self._randomize_attributes(ra, task_ref.all_task_attributes)
        else:
            print("[RANDOMIZER] No config attributes detected. Shuffle as default.")
        
        self.tmp_translation = {}

    ######## RANDOMIZE

    def _randomize_tool(self, tool_config : Dict, original_tools : List):
        self.tools = []
        self._param_patterns = {}
        for tool in original_tools:
            tool_augmentation = tool_config.get(tool['name'],None)

            if not tool_augmentation:
                print(f"[RANDOMIZER] Unknow function {tool['name']}")
                self.tools.append(tool)
                self.tools_equivalence[tool['name']] = {"name": tool["name"], "parameters":{k:k for k in tool['parameters']}}
                continue

            name = random.choice(tool_augmentation['name'])
            description = random.choice(tool_augmentation['description'])

            parameters_dict = {}
            param_equivalence = {}
            for param, param_val in tool['parameters'].items():
                possible_param_val = tool_augmentation['parameters'].get(param,[])
                if not possible_param_val: #Si jamais le paramètre n'a pas été definit
                    print(f"[RANDOMIZER] Unknow parameter {param} for function {tool['name']}")
                    p_name = param
                    p_desc = param_val['description']
                else:
                    p_name = random.choice(possible_param_val['name'])
                    p_desc = random.choice(possible_param_val['description'])

                parameters_dict[p_name] = {"description" : p_desc, "type": param_val["type"]}
                param_equivalence[p_name] = param

            self.tools.append({"name":name, "description" : description, "parameters" : parameters_dict})
            self.tools_equivalence[name] = {"name" : tool['name'], "parameters":param_equivalence}

            reversed_p = {
                v: k
                for k, v in self.tools_equivalence[name]['parameters'].items()
            }
            pattern = re.compile(
                r'\b(' + '|'.join(map(re.escape, reversed_p.keys())) + r')\b'
            )
            self._param_patterns[name] = (pattern, reversed_p)

    def get_randomized_attributes(self, attributes : Dict):
        out = {}
        for att_name, att_val in attributes.items():
            if att_name == "known_robots" or not isinstance(att_val, List): # How can i handle like str value (like state, where the state name can be randomized)
                out[att_name] = att_val
                continue # We do not randomize if attributes are dict, int
            out[att_name] = []
            for val in att_val:
                if val in self.original_attributes_listed:   
                    randomized_val = self.attributes_equivalence.get(val, None)
                else:
                    randomized_val = val

                out[att_name].append(randomized_val)

            random.shuffle(out[att_name])

        keys = list(out.keys())
        random.shuffle(keys)
        return {k: out[k] for k in keys}


    def _randomize_attributes(self, attributes_config : Dict, task_attributes : Dict):        
        for att_name, att_val in task_attributes.items():
            if att_name == "known_robots": continue
            possible_val = attributes_config.get(att_name,None)
            if not possible_val:
                print(f"[RANDOMIZER] Warn : No randomization for attributes {att_name}")
                continue
            if not isinstance(att_val, List): # How can i handle like str value (like state, where the state name can be randomized)
                continue # We do not randomize if attributes are dict, int
            for i, config_val in enumerate(possible_val):
                if not isinstance(config_val, List):
                    raise TypeError(f"There is a config value ({i}) for attributes {att_name} which is not a List ({type(config_val)})")
                if len(config_val) < len(att_val):
                    raise TypeError(f"There is a config value ({i}) for attributes {att_name} with a lenght ({len(config_val)}) < to task attributes ({len(att_val)})")
                               

            config_values = random.choice(possible_val)

            for i, val in enumerate(att_val):
                self.original_attributes_listed.append(val)
                self.attributes_equivalence[val] = config_values[i]


        self.reversed_attributes_equivalence = {v: k for k, v in self.attributes_equivalence.items()}
        self._pattern_to_env = re.compile(
            r'\b(' + '|'.join(map(re.escape, self.reversed_attributes_equivalence.keys())) + r')\b'
        )
        self._pattern_to_llm = re.compile(
            r'\b(' + '|'.join(map(re.escape, self.attributes_equivalence.keys())) + r')\b'
        )
        print(self.attributes_equivalence)

    ########### getters

    def get_tools(self) -> List[Dict]:
        return self.tools
    
    def get_randomized_situation(self, original_situation: Situation) -> Situation:
        """
        Ceeate a new situation instance with randomized elements
        """
        attributes = self.get_randomized_attributes(original_situation.attributes)

        if isinstance(original_situation.instruction, UserInstruction):
            new_instruction_content = self.traduce_attributes_to_llm(original_situation.instruction.get_content())
            new_instruction = UserInstruction(new_instruction_content)
        elif isinstance(original_situation.instruction, StatusReturn):
            new_instruction_content = self.traduce_attributes_to_llm(original_situation.instruction.get_content())
            # AJOUTER FONCTION ICI
            new_instruction = StatusReturn(json.loads(new_instruction_content))
        elif isinstance(original_situation.instruction, TemplateInstruction):
            print("===========")
            print(original_situation.instruction.template)
            new_template_str = self.traduce_attributes_to_llm(json.dumps(original_situation.instruction.template))
            print(new_template_str)
            new_context_str = self.traduce_attributes_to_llm(original_situation.instruction.context)
            new_instruction = TemplateInstruction(json.loads(new_template_str), new_context_str)
        else:
            new_instruction = EmptyInstruction()

        new_memory = [self.traduce_attributes_to_llm(mem) for mem in original_situation.memory]
        new_user_scenario = self.traduce_attributes_to_llm(original_situation.user_scenario)

        return original_situation.copy_with(
            memory=new_memory,
            attributes=attributes,
            instruction=new_instruction,
            user_scenario=new_user_scenario
        )
    
    def map_tool_call(self, func_name : str, params : Dict) -> Union[ToolExecution, Tuple[str, Dict]]:
        """Handles validation, parameter remapping, and ToolInfos creation for one tool."""
        # Unknown tool
        if func_name not in self.tools_equivalence:
            return ToolExecution([], None, reason=f"{func_name} is not a known tool.")

        # Map parameters
        mapping = self.tools_equivalence[func_name]["parameters"]
        new_func_name = self.tools_equivalence[func_name]["name"]

        new_params, err_param_names, err_param_values = {}, [], []
        for p, val in params.items():
            if p in mapping:
                param_mapped, unk = self._recursive_arguments_replacement(val)
                if len(unk)>0:
                    err_param_values.extend(unk)
                else:
                    new_params[mapping[p]] = param_mapped
            else:
                err_param_names.append(p)

        if err_param_names:
            invalid = ", ".join(err_param_names)
            return ToolExecution([], None, reason=f"{invalid} are not valid arguments")

        if err_param_values:
            invalid = ", ".join(err_param_values)
            return ToolExecution([], None, reason=f"Unknow attributes: {invalid} are not valid attributes")

        return new_func_name, new_params
    
    def traduce_attributes_to_env(self, text: str) -> str:
        """Replace randomized attributes names in a string with their official task names."""
        if self.reversed_attributes_equivalence:
            a =  self._pattern_to_env.sub(
                lambda m: self.reversed_attributes_equivalence[m.group(0)],
                text
            )
            return a
        return text
    
    def traduce_attributes_to_llm(self, text: str) -> str:
        """Replace attributes real names in a string with their randomized names."""
        if self.attributes_equivalence:
            a =  self._pattern_to_llm.sub(
                lambda m: self.attributes_equivalence[m.group(0)],
                text
            )
            return a
        return text
    
    def _replace_missing_arg(self, text : str, func_name : str) -> str:
        """
        Replace parameters name and description of a given function
        """
        pattern, reversed_p = self._param_patterns[func_name]

        return pattern.sub(lambda m: reversed_p[m.group(0)], text)

    def _recursive_arguments_replacement(self, argument):
        """
        Recursive replace of arguments from a function call
        """
        unknown_arg = []

        def inner(argument):
            if isinstance(argument, List):
                out = []
                for a in argument:
                    out.append(inner(a))
                return out
            elif isinstance(argument, Dict):
                out = {}
                for k, v in argument.items():
                    k_a = inner(k)
                    k_v = inner(v)
                    out[k_a] = k_v
                return out

            att = self.reversed_attributes_equivalence.get(argument,None)
            if not att:
                if argument in self.original_attributes_listed:
                    unknown_arg.append(argument)
                    return "_"
                return argument
            return att

        att = inner(argument)
                
        return att, unknown_arg
    
    def _traduce_reason(self, reason : Optional[str], node_id : int) -> Optional[str]:
        if reason is not None:
            src_names = self.tmp_translation.pop(node_id)
            if isinstance(src_names, List):
                if len(src_names) > 1:
                    raise NotImplementedError("Randomization for multi-robot tool is not yet implemented")
                else:
                    src_names = src_names[0]
            if reason.startswith("Unknow attributes:"):
                return reason
            if reason.startswith("Missing arguments:") or reason.startswith("Argument"):
                return self._replace_missing_arg(reason, src_names)
            else:
                return self.traduce_attributes_to_llm(reason)
        return reason
    
    def _traduce_att_modif(self, modif : List[Tuple[str,Tuple[str,str]]]):
        tmp = []
        for action, tupl in modif:
            att, value = tupl
            tmp.append((action, (att, self.traduce_attributes_to_llm(value))))
        return tmp

    def traduce_end_gen(self, out : Dict[int, ToolStatus]) -> Dict[int, ToolStatus]:
        if out and self.attributes_equivalence:
            for node_id, status in out.items():
                status.mess = self._traduce_reason(status.mess, node_id)
                if status.next_situation is not None:
                    status.next_situation = self.get_randomized_situation(status.next_situation)
                status.attributes_modif = self._traduce_att_modif(status.attributes_modif)
        return out
    

    def traduce_end_eval(self, out : Dict[int, Dict]) -> Dict[int, Dict]:
        if out and self.attributes_equivalence:
            for node_id, status in out.items():
                status['reason'] = self._traduce_reason(status.get("reason", None), node_id)
                status["att_modif"] = self._traduce_att_modif(status["att_modif"])
        return out