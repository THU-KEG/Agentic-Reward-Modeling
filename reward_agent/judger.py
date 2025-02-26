from collections import Counter

class Judger:
    def __init__(self, model):
        """
        Initialize the Planner with a given model.

        Args:
            model: An instance of a text generation model that supports planning.
        """
        self.model = model

    def _argmax(self, answer_with_scores):
        items = list(answer_with_scores.items())
        items = sorted(items, key=lambda item: item[1], reverse=True)
        return items[0][0]

    def assert_all_equal(self, answer_with_scores):
        values = list(answer_with_scores.values())
        if len(set(values)) == 1:
            return True
        else:
            return False

    def judge(self, instruction, judgments, type="llm", weights=None):
        if type == "llm":
            prompt = f"""Please make a final judgment based on the given instruction and the judgments for Answer A and Answer B. Choose one answer (A or B) according to the following rules:


            [Instruction]
            {instruction}
            
            [Factuality Judgments]
            {judgments.get("factuality", "None")}

            [Constraint Judgments Scores, larger is more preferred]
            {judgments.get("constraint", "None")}

            [Human Preference Judgments Scores, larger is more preferred]
            {judgments.get("human_pref", "None")}

            [Remember]
            Choose one answer (A or B) according to the following rules:
            Factuality: Prioritize factuality above all. If one answer contains more factuality errors, do not choose that answer. Factuality errors refer to objective errors related to world knowledge; subjective inconsistencies are not considered factuality errors.
            Constraints: Choose the answer that satisfies more constraints.
            Human Preference: Choose the answer that aligns better with human preferences.

            Your output should only consist of '[[A]]' if answer A is better, or '[[B]]' if answer B is better.

            """
            messages = [
                {"role": "user", "content": prompt}
            ]
            # Use the model to generate a decision
            output = self.model.generate_chat(messages)
            return output
        elif type == "major_vote":
            weights = {
                "factuality": 1.2,
                "constraint": 1.1,
                "human_pref": 1.0
            }
            win_times = Counter()
            for key in judgments:
                best = self._argmax(judgments[key])
                if self.assert_all_equal(judgments[key]):
                    for answer in judgments[key]:
                        win_times[answer] += weights[key] / len(judgments[key])
                else:
                    win_times[best] += weights[key]
            winner = self._argmax(win_times)
            return f"[[{winner.split(' ')[-1]}]]"
        elif type == "weighted_sum":
            if weights is None:
                weights = {
                    "factuality": 1.0,
                    "constraint": 1.0,
                    "human_pref": 1.0
                }
            weighted_score = Counter()
            for key in judgments:
                for answer in judgments[key]:
                    weighted_score[answer] += weights[key] * judgments[key][answer]
            winner = self._argmax(weighted_score)
            return f"[[{winner.split(' ')[-1]}]]"
        else:
            raise ValueError()

    
    def judge_multi_legacy(self, instruction, judgments):
        prompt = f"""Please make a final judgment based on the given instruction and the judgments for each answer. Choose the best answer according to the following rules:

        [Instruction]
        {instruction}

        [Factuality Judgments]
        {judgments.get("factuality", "None")}

        [Constraint Judgments Scores, larger is more preferred]
        {judgments.get("constraint", "None")}

        [Human Preference Judgments Scores, larger is more preferred]
        {judgments.get("human_pref", "None")}

        [Remember!!]
        Choose the best answer according to the following rules:
        Factuality: Prioritize factuality above all. If one answer contains more factuality errors, do not choose that answer. Factuality errors refer to objective errors related to world knowledge; subjective inconsistencies are not considered factuality errors.
        Constraints: Choose the answer that satisfies more constraints.
        Human Preference: Choose the answer that aligns better with human preferences.

        Output Format: Your output should consist solely of '[[1]]', '[[2]]', etc., indicating which answer is the best (e.g., '[[1]]' if answer 1 is the best).
        """

        messages = [
            {"role": "user", "content": prompt}
        ]

        # Use the model to generate a decision
        output = self.model.generate_chat(messages)

        return output