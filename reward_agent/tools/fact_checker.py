import re
from .utils import _type_check, extract_list_from_string, process_judgment, process_judgment_multi


class FactChecker:
    def __init__(self, checker):
        self.checker = checker
    

    def get_difference(self, answers):
        formatted_answers = "\n".join([f"Answer {chr(65 + i)}: {answer}" for i, answer in enumerate(answers)])
        """
        This function compares two answers, answer_a and answer_b, and identifies any contradictions or differences in the facts presented in each.
        A fact is any piece of world knowledge that can be verified or disproven. 
        Return the differing or contradictory facts, separated by two newlines ("\n\n").
        
        Example output: 
        Fact 1 from answer_a contradicts fact 2 from answer_b.
        
        Fact 3 from answer_b differs from fact 4 from answer_a.
        """
        # Prompt to model
        prompt = f"""
        [Answers]
        {formatted_answers}

        [Your Task]
        Given the above responses, please identify and summarize one key points of contradiction or inconsistency between the claims.

        [Requirements]
        1. Return a Python list containing only the most significant differences between the two answers.
        2. Do not include any additional explanations, only output the list.
        3. If there are no inconsistencies, return an empty list.
        """

        messages = [
            {"role": "user", "content": prompt},
        ]
        # Generate the response based on the comparison
        response = self.checker.generate_chat(messages, max_tokens=256)
        # import pdb; pdb.set_trace()
        
        return response


    def query_generation(self, instruction, inconsistencies):
        prompt = f"""
        [Original question that caused the inconsistency]
        {instruction}

        [Inconsistencies]
        {inconsistencies}
        
        [Your Task]
        To resolve the inconsistencies, We need to query search engine. For each contradiction, please generate a corresponding query that can be used to retrieve knowledge to resolve the contradiction. 
        
        [Requirements]
        1. Each query should be specific and targeted, aiming to verify or disprove the conflicting points. 
        2. Provide the queries in a clear and concise manner, returning a Python list of queries corrresponding to the inconsistencies.
        3. Do not provide any additional explanations, only output the list.
        """

        messages = [
            {"role": "user", "content": prompt},
        ]    
        # Generate the response based on the comparison
        response = self.checker.generate_chat(messages, max_tokens=256)
        if response is None:
            print(prompt)
            return ["NA"]
        # import pdb; pdb.set_trace()
        response_list = _type_check(extract_list_from_string(response))
        if response_list is None or len(response_list) == 0:
            return [response]
        else:
            return response_list


    async def get_support(self, search_engine, queries):
        supports = []
        results = await search_engine.run(queries)
        for rs in results:
            rs = rs[:5]
            support = "\n\n".join([r["content"] for r in rs])
            supports.append(support)
        return supports


    def get_support_local(self, queries):
        supports = []
        for query in queries:
            support = self.checker.search(query, max_tokens=512)
            supports.append(support)
        return supports
    

    def parse_answer(self, text, pattern):
        match = re.search(pattern, text)
        if match is None or len(match.groups()) == 0:
            return 5
        score = int(match.group(1))
        return score


    def check(self, instruction, answers, inconsistencies, supports):
        formatted_answers = "\n".join([f"Answer {chr(65 + i)}: {answer}" for i, answer in enumerate(answers)])
        prompt = f"""Evaluate which of the two answers is more factual based on the supporting information.
        [Support knowledge sources]:
        {supports}

        [Instruction]
        {instruction}

        [Original Answers]:
        {formatted_answers}

        [Remeber]
        For each answer, provide a score between 1 and 10, where 10 represents the highest factual accuracy. Your output should only consist of the following:
        Answer A: [[score]] (Wrap the score of A with [[ and ]])
        Answer B: <<score>> (Wrap the score of B with << and >>)
        Please also provide a compact explanation. 
        If the two answers is semantically same (but with different expressions), please score both 10.
        If the two answers have similar factuality, please score higher for the shorter answer.
        """
        messages = [
            {"role": "user", "content": prompt},
        ]
        # Generate the response based on the comparison
        response = self.checker.generate_chat(messages, max_tokens=512)
        if response is None:
            print(prompt)
            return {
                "Answer A": 5 / 10,
                "Answer B": 5 / 10
            }
        score_a = self.parse_answer(response, r"\[\[(\d+)\]\]")
        score_b = self.parse_answer(response, r"<<(\d+)>>")
        return {
            "Answer A": score_a / 10,
            "Answer B": score_b / 10
        }

    # def check(self, instruction, answers, inconsistencies, supports):
    #     formatted_answers = "\n".join([f"Answer {chr(65 + i)}: {answer}" for i, answer in enumerate(answers)])
    #     prompt = f"""Evaluate which of the two answers is more factual based on the supporting information. If it is still unclear which answer 
    #     is more factual, return 'tie'.

    #     [Support knowledge sources]:
    #     {supports}

    #     [Original Answers]:
    #     {formatted_answers}

    #     [Remeber]
    #     Evaluate which of the two answers is more factual based on the supporting information. If it is still unclear which answer 
    #     is more factual, return 'tie'. 
    #     Your output should only consist of '[[A]]' if answer A is better, or '[[B]]' if answer B is better.
    #     """
    #     messages = [
    #         {"role": "user", "content": prompt},
    #     ]
    #     # Generate the response based on the comparison
    #     response = self.checker.generate_chat(messages, max_tokens=512)
    #     if "[[A]]" in response:
    #         return {
    #             "Answer A": 1.0,
    #             "Answer B": 0
    #         }
    #     elif "[[B]]" in response:
    #         return {
    #             "Answer A": 0,
    #             "Answer B": 1.0
    #         }
    #     else:
    #         return {
    #             "Answer A": 0.5,
    #             "Answer B": 0.5
    #         }