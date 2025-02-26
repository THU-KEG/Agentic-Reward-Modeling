import re
import importlib
import multiprocessing
import importlib
import inspect
import logging
from .code_executer import execute_code

# Create a custom logger
logger = logging.getLogger("CustomLogger")
# Set the minimum log level for the logger (DEBUG is the lowest level)
logger.setLevel(logging.DEBUG)
# Create a console handler (to print logs to the terminal)
console_handler = logging.StreamHandler()
# Create a formatter to specify the log message format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Add the formatter to the handler
console_handler.setFormatter(formatter)
# Add the handler to the logger
logger.addHandler(console_handler)


def mean_func(numbers):
    mean_num = sum(numbers) / len(numbers)
    return mean_num


def _evaluate_reward(response, functions, reduction="mean"):
    results = []
    for function in functions:
        result, _ = execute_code(response, function)
        if result:
            results.append(1)
        else:
            results.append(0)
    if reduction == "mean":
        return mean_func(results)
    else:
        return results


def evaluate_if_reward(analyzer, instruction, answer_a, answer_b, return_detail=False):
    functions, checker_names = analyzer.analyze_checker(instruction)
    if len(functions) == 0:
        if return_detail:
            return 0.5, 0.5
        else:
            return "tie"
    if return_detail:
        score_a = _evaluate_reward(answer_a, functions, reduction=None)
        score_b = _evaluate_reward(answer_b, functions, reduction=None)

        score_a_detail, score_b_detail = [], []
        for _score_a, checker_name in zip(score_a, checker_names):
            score_a_detail.append({
                "Constraint": checker_name,
                "If Meet Constraint": True if _score_a == 1 else False
            })
        for _score_b, checker_name in zip(score_b, checker_names):
            score_b_detail.append({
                "Constraint": checker_name,
                "If Meet Constraint": True if _score_b == 1 else False
            })
        # return score_a_detail, score_b_detail
        return mean_func(score_a), mean_func(score_b)
    else:
        score_a = _evaluate_reward(answer_a, functions)
        score_b = _evaluate_reward(answer_b, functions)
        if score_a > score_b:
            return "A"
        elif score_a < score_b:
            return "B"
        else:
            return "tie"



# def evaluate_if_reward(analyzer, instruction, answer_a, answer_b, return_detail=False):
#     def parse_answer(text, pattern):
#         match = re.search(pattern, text)
#         if match is None or len(match.groups()) == 0:
#             return 5
#         score = int(match.group(1))
#         return score
#     prompt = f"""Given an instruction that may contain constraints and two responses, 
#     please determine whether each response adheres to the constraints outlined in the instruction.

#     [Instruction]
#     {instruction}

#     [Answer A]
#     {answer_a}

#     [Answer B]
#     {answer_b}

#     [Remeber]
#     For each answer, provide a score between 1 and 10, where 10 represents the highest instruction following score. Your output should only consist of the following:
#     Answer A: [[score]] (Wrap the score of A with [[ and ]])
#     Answer B: <<score>> (Wrap the score of B with << and >>)
#     Please also provide a compact explanation.
#     """
#     messages = [
#         {"role": "user", "content": prompt},
#     ]    
#     # Generate the response based on the comparison
#     response =analyzer.checker_extractor.generate_chat(messages, max_tokens=512)
#     score_a = parse_answer(response, r"\[\[(\d+)\]\]")
#     score_b = parse_answer(response, r"<<(\d+)>>")
#     return score_a / 10, score_b / 10


def evaluate_if_reward_multi(analyzer, instruction, answers, functions=None, return_functions=False):
    if functions is None:
        functions, checker_names = analyzer.analyze_checker(instruction)
    if len(functions) == 0:
        if return_functions:
            return ([0.5] * len(answers), functions)
        else:
            return [0.5] * len(answers)
    scores = []
    for answer in answers:
        scores.append(_evaluate_reward(answer, functions))
    if return_functions:
        return (scores, functions)
    else:
        return scores



class ConstraintAnalyzer:
    def __init__(self, coder):
        self.checker_extractor = coder
        self.checker_classifier = coder
        self.code_generator = coder

    def analyze_checker(self, instruction):
        """
        Analyze the given instruction and return corresponding checker functions that can be called directly.
        
        Parameters:
        - instruction (str): A string containing the constraints to check for.
        
        Returns:
        - List[function]: A list of checker functions that can be called directly (e.g., ResponseLanguageChecker)
        """
        
        MAX_REFLECT_TIMES = 2
        # 解析 instruction，提取出需要检查的约束条件
        checkers_to_apply = self._extract_checkers_from_instruction(instruction)
        
        # 对每个选择的检查器，导入并返回函数接口
        checker_functions = []
        checker_names = []
        for checker_name in checkers_to_apply:
            if not self._check_checker_verifiable(instruction, checker_name):
                continue
            
            # 使用 self.model 来生成最终的检查逻辑代码
            generated_code = self._generate_code_from_model(checker_name, instruction)

            # check
            flag = False
            check_times = 0
            while check_times < MAX_REFLECT_TIMES:
                error_info, flag = self._check_function(generated_code, "This is for a test.")
                if flag:
                    break
                generated_code = self.reflect_function(generated_code, error_info)
                check_times += 1

            # 将生成的代码转化为可以调用的函数
            # checker_function = self._create_function_from_code(generated_code)
            if flag:
                checker_functions.append(generated_code)
                checker_names.append(checker_name)
            else:
                print(f"{generated_code}\n{checker_name} has been filtered")
        
        return checker_functions, checker_names


    def _extract_checkers_from_instruction(self, instruction):
        """
        从 instruction 中提取出需要检查的约束条件（此处假设 instruction 是一个字符串）
        根据实际情况，这个方法可以做更多的解析和条件判断。
        
        Parameters:
        - instruction (str): 输入的指令，其中可能包含多个约束条件的标识符
        
        Returns:
        - List[str]: 需要检查的检查器名称列表
        """
        prompt = f"""
        You are an expert in natural language processing and constraint checking. Your task is to analyze a given instruction and identify which constraints need to be checked.

        The `instruction` contains a specific task query along with several explicitly stated constraints. Based on the instruction, you need to return a list of checker names that should be applied to the constaints.

        Task Example:  
        Instruction: Write a 300+ word summary of the Wikipedia page "https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli". Do not use any commas and highlight at least 3 sections that have titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*.\n\n  
        Response: NumberOfWordsChecker: 300+ word\nHighlightSectionChecker: highlight at least 3 sections that have titles in markdown format\nForbiddenWordsChecker: Do not use any commas

        Task Instruction:
        {instruction}

        ### Your task:
        - Generate the appropriate checker names with corresponding descriptions from original instruction description.
        - Return the checker names with their descriptions sperated by `\n`
        - Focus only on the constraints explicitly mentioned in the instruction (e.g., length, format, specific exclusions).  
        - Do **not** generate checkers for the task query itself or its quality.
        - Do **not** infer or output constraints that are implicitly included in the instruction (e.g., general style or unstated rules).
        - Each checker should be responsible for checking only one constraint.
        """

        messages = [
            {"role": "system", "content": prompt}
        ]

        response = self.checker_extractor.generate_chat(messages, max_tokens=256)
        MAX_CHECKERS = 8
        try:
            # checkers = eval(response.split("\n")[-1].replace("\n", ""))
            checkers = list(set([checker for checker in response.split("\n") if "Checker" in checker]))
            if len(checkers) > MAX_CHECKERS:
                print(checkers, "-"*20)
            checkers = checkers[:MAX_CHECKERS]
        except:
            print(response, "is not formatted correctly")
            checkers = []
        return checkers


    def _get_checker_code(self, checker_name):
        """
        获取 `checker_name` 对应的检查器函数的代码
        
        Parameters:
        - checker_name (str): 检查器的名称，例如 'ResponseLanguageChecker'
        
        Returns:
        - str: 对应的检查器函数的代码
        """
        try:
            # 假设 check.py 和此文件在同一目录下
            module = importlib.import_module("checker")  # 导入 check.py
            
            # 获取检查器函数的源码
            checker_function = getattr(module, checker_name, None)
            if checker_function is None:
                raise ImportError(f"Checker function '{checker_name}' not found in checker.py")
            
            # 获取函数源码
            code = inspect.getsource(checker_function)
            return code
        
        except ImportError as e:
            print(f"Error importing checker: {e}")
            return None
    
    def _get_checker_code_example(self):
        try:
            # 假设 check.py 和此文件在同一目录下
            module = importlib.import_module("code_exampler")  # 导入 check.py

            # 获取函数源码
            code = inspect.getsource(module)
            return code
        except ImportError as e:
            print(f"Error importing checker: {e}")
            return None
        

    def _check_checker_verifiable(self, instruction, checker_name):
        prompt = f"""
        Checker:
        {checker_name}
        
        Please determine:
        1. whether the checker is simple ** surface form checks**, such as lexical (word count, etc.), syntactic (start or end phrase, etc.), and grammatical (no commas, etc.) checks.
        1. whether the checker condition and definition is clear and has a deterministic answer. 
        If the checker meets all of the above conditions, respond with 'yes'; otherwise, respond with 'no'.
        PLEASE only respond with yes or no.
        """
        messages = [
            {"role": "user", "content": prompt},
        ]

        # Generate response from the classifier to determine if the checker is verifiable
        content = self.checker_classifier.generate_chat(messages, max_tokens=1)

        # Return True if 'yes' is returned, otherwise False
        if content.lower() == "yes":
            return True
        else:
            # import pdb; pdb.set_trace()
            logger.info("Checher: {}\tType: {}".format(checker_name, content))
            return False


    def _generate_code_from_model(self, checker_name, instruction):
        """
        使用 GPT-4 生成检查逻辑代码，输入为 checker_code 和 instruction。
        
        Parameters:
        - checker_code (str): 获取的检查器函数代码
        - instruction (str): 输入的指令
        
        Returns:
        - str: GPT-4 生成的检查逻辑代码

        Referenced Checker Class:
        {checker_code}
        """
        prompt = f"""
        You are tasked with implementing a Python function `check_following` that determines whether a given `response` satisfies a constraint defined by a checker. The function should return `True` if the constraint is satisfied, and `False` otherwise.

        [Instruction to check]:
        {instruction}

        [Specific Checker and Description]:
        {checker_name}

        Requirements:
        - The function accepts only one parameter: `response` which is a python string.
        - The function must return a boolean value (`True` or `False`) based on whether the `response` adheres to the constraint described by the checker.
        - The function must not include any I/O operations, such as `input()` or `ArgumentParser`.
        - The Python code for each checker should be designed to be generalizable, e.g., using regular expressions or other suitable techniques.
        - Only return the exact Python code, with no additional explanations.
        """

        messages = [
            {"role": "user", "content": prompt},
        ]    
        generated_code = self.code_generator.generate_chat(messages, max_tokens=256)
        # normalize
        generated_code = generated_code.replace("```python\n", "").replace("```", "")
        return generated_code

    def _create_function_from_code(self, generated_code):
        """
        从 GPT-4 生成的代码中创建一个可以执行的函数。
        
        Parameters:
        - generated_code (str): 由 GPT-4 生成的代码
        
        Returns:
        - function: 动态创建的函数
        """
        # 在当前作用域中执行生成的代码并返回函数
        local_scope = {}
        exec(generated_code, {}, local_scope)
        
        # 返回创建的函数
        return local_scope.get('check_following')


    def _check_function(self, function, response, timeout=5):
        result, error_info = execute_code(response, function)
        return error_info, error_info is None


    def reflect_function(self, function, error_info):
        # Prepare the prompt to ask `self.model` to fix the function based on the error_info
        prompt = f"""
        Based on the following error information `{error_info}`, please analyze and modify the given code `function`. 
        You need to:
        1. Fix the issues in the provided code and ensure it runs correctly.
        2. The `check_following` function should accept **only** a `response` parameter and return `True` or `False` based on the conditions described in the provided `instruction`.
        
        Code to be modified: 
        {function}
        
        Error Information: 
        {error_info}

        Requirements:
        1. Fix any errors and output the corrected code.

        Please only output exactly the Python code without ```python``` in the begin or in the end and without any explanations to the modifications.
        """
        messages = [
            {"role": "user", "content": prompt},
        ]
        modified_code = self.code_generator.generate_chat(messages)  # Assuming `generate_code` is the method for calling the model
        modified_code = modified_code.replace("```python\n", "").replace("```", "")
        
        return modified_code