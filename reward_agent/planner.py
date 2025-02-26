
tools_desps = [
    {
        "name": "constraint check",
        "desp": "A 'constraint check' is required if the instruction contains any additional constraints or requirements on the output, such as length, keywords, format, number of sections, frequency, order, etc.",
        "identifier": "[[A]]"
    },
    {
        "name": "factuality check",
        "desp": "A 'factuality check' is required if the generated response to the instruction potentially contains claims about factual information or world knowledge.",
        "identifier": "[[B]]"
    }
]


class Planner:
    def __init__(self, model):
        """
        Initialize the Planner with a given model.

        Args:
            model: An instance of a text generation model that supports planning.
        """
        self.model = model

    
    def plan_01(self, instruction, response):
        """
        Determine whether the given instruction and response require a constraint check or a factuality check.

        Args:
            instruction (str): The instruction provided.
            response (str): The generated response.

        Returns:
            dict: A dictionary with keys 'constraint_check' and 'factuality_check', each mapping to a boolean.
        """
        # Define the prompt to be sent to the model
        # prompt = (
        #     "Given the following instruction and response, determine whether a constraint check or factuality check is required. "
        #     "Provide your answer in the format: 'constraint_check: <True/False>, factuality_check: <True/False>'.\n\n"
        #     f"Instruction: {instruction}\n"
        #     f"Response: {response}\n\n"
        #     "Answer:"
        # )
        prompt = f"""
        Instruction: {instruction}

        Given the above instruction, determine whether a constraint check or factuality check is required.

        A 'constraint check' is required if the instruction contains any additional constraints on the output, such as length, keywords, format, number of sections, frequency, etc.
        A 'factuality check' is required if the generated response contains claims about information or world knowledge.

        If a 'constraint check' is needed, your output should only consist of [[C]].
        If a 'factuality check' is needed, your output  should only consist of [[F]]. 
        """
        messages = [
            {"role": "user", "content": prompt}
        ]

        # Use the model to generate a decision
        output = self.model.generate_chat(messages)

        # Parse the output to extract decisions
        decisions = {
            "constraint_check": False,
            "factuality_check": False
        }

        if "[[C]]" in output:
            decisions["constraint_check"] = True
        if "[[F]]" in output:
            decisions["factuality_check"] = True

        # import pdb; pdb.set_trace()
        return decisions

    
    def plan(self, instruction):
        """
        Determine whether the given instruction and response require a constraint check or a factuality check.

        Args:
            instruction (str): The instruction provided.
            response (str): The generated response.

        Returns:
            dict: A dictionary with keys 'constraint_check' and 'factuality_check', each mapping to a boolean.
        """
        prompt = f"""
        Given the following instruction, determine whether the following check in needed.

        [Instruction]
        {instruction}

        [Checks]
        {tools_desps}

        If the instruction requires some checks, please output the corresponding identifiers (such as [[A]], [[B]]).
        Please do not output other identifiers if the corresponding checkers not needed.
        """
        messages = [
            {"role": "user", "content": prompt}
        ]

        # Use the model to generate a decision
        output = self.model.generate_chat(messages)

        # Parse the output to extract decisions
        decisions = {
            "constraint_check": False,
            "factuality_check": False
        }

        if "[[A]]" in output:
            decisions["constraint_check"] = True
        if "[[B]]" in output:
            decisions["factuality_check"] = True

        return decisions