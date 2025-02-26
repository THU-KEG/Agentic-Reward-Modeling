
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