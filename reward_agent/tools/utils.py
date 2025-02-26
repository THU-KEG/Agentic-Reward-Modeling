import ast
from typing import List

def extract_list_from_string(input_string):
    start_index = input_string.find('[')  
    end_index = input_string.rfind(']') 

    if start_index != -1 and end_index != -1 and start_index < end_index:
        return input_string[start_index:end_index + 1]
    else:
        return None
    
def extract_dict_from_string(input_string):
    start_index = input_string.find('{')
    end_index = input_string.rfind('}')

    if start_index != -1 and end_index != -1 and start_index < end_index:
        return input_string[start_index:end_index + 1]
    else:
        return None


def _type_check(output, expected_type=List):
    try:
        output_eval = ast.literal_eval(output)
        if not isinstance(output_eval, expected_type):
            return None
        return output_eval
    except:
        '''
        if(expected_type == List):
            valid_output = self.extract_list_from_string(output)
            output_eval = ast.literal_eval(valid_output)
            if not isinstance(output_eval, expected_type):
                return None
            return output_eval
        elif(expected_type == dict):
            valid_output = self.extract_dict_from_string(output)
            output_eval = ast.literal_eval(valid_output)
            if not isinstance(output_eval, expected_type):
                return None
            return output_eval
        '''
        return None


def process_judgment(judgment):
    if "[[A]]" in judgment:
        return "A"
    elif "[[B]]" in judgment:
        return "B"
    else:
        return "tie"


def process_judgment_multi(judgment):
    for i in range(100):
        if f"[[{i}]]" in judgment:
            return i - 1
    return "error"