
from .tools import ConstraintAnalyzer, evaluate_if_reward, evaluate_if_reward_multi
from .tools import FactChecker
from .tools import GoogleSerperAPIWrapper
from .tools import process_judgment, process_judgment_multi
from .rm import rm
import asyncio
import logging
import random
from collections import Counter
from itertools import combinations


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



class RewardAgent:
    def __init__(self, planner, judger, judger_type, reward_model, tokenizer, tools):
        self.planner = planner
        self.judger = judger
        self.judger_type = judger_type
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.tools = tools
        if "constraint_analyzer" in tools:
            self.constraint_analyzer = ConstraintAnalyzer(tools["constraint_analyzer"])
        if "fact_checker" in tools:
            self.fact_checker = FactChecker(tools["fact_checker"])
            if tools["search_engine"] == "online":
                self.search_engine = GoogleSerperAPIWrapper()
            else:
                self.search_engine = None
        self.tool_call_count = Counter()


    def judge_pair(self, instruction, answer_a, answer_b, answer_pair, **kwargs):
        """
        Judge the pair of answers (answer_a and answer_b) based on the given instruction.

        Args:
            instruction (str): The instruction for the task.
            answer_a (str): The first answer to evaluate.
            answer_b (str): The second answer to evaluate.

        Returns:
            dict: A dictionary with keys 'answer_a_score', 'answer_b_score', and 'preferred_answer'.
        """
        # Use the planner to determine if constraint or factuality checks are needed
        plan_result = self.planner.plan(instruction)

        # Check constraints if required
        juduments = {}

        if plan_result.get("factuality_check", False):
            self.tool_call_count["factuality_check"] += 1
            logger.info("[[factuality check]]")
            inconsistencies = self.fact_checker.get_difference([answer_a, answer_b])
            queries = self.fact_checker.query_generation(instruction, inconsistencies)
            if queries is not None:
                if self.tools["search_engine"] == "online":
                    supports = asyncio.run(self.fact_checker.get_support(self.search_engine, queries))
                elif self.tools["search_engine"] == "local":
                    supports = self.fact_checker.get_support_local(queries)
                else:
                    supports = []
                fact_judge = self.fact_checker.check(instruction, [answer_a, answer_b], inconsistencies, supports)
                juduments["factuality"] = fact_judge
            else:
                logger.warning("No queries in factuality check.")

        if plan_result.get("constraint_check", False):
            self.tool_call_count["constraint_check"] += 1
            logger.info("<<constraint check>>")
            score_a, score_b = evaluate_if_reward(self.constraint_analyzer, instruction, answer_a, answer_b, return_detail=True)
            juduments["constraint"] = {
                "Answer A": score_a,
                "Answer B": score_b
            }


        kwargs["return_raw"] = True
        if self.tokenizer is not None:
            score_a, score_b = rm.get_reward(self.reward_model, "rm", self.tokenizer, instruction, answer_a, answer_b, answer_pair, **kwargs)
            juduments["human_pref"] = {
                "Answer A": score_a,
                "Answer B": score_b
            }
        else:
            human_pref = rm.get_reward(self.reward_model, "generative", self.tokenizer, instruction, answer_a, answer_b, answer_pair, **kwargs)
            juduments["human_pref"] = human_pref
        
        result = self.judger.judge(instruction, juduments, type=self.judger_type)
        winner = process_judgment(result)
        return winner
    

    def judge_multi_elo(self, instruction, answers, **kwargs):
        """
        Judge the pair of answers (answer_a and answer_b) based on the given instruction.

        Args:
            instruction (str): The instruction for the task.
            answer_a (str): The first answer to evaluate.
            answer_b (str): The second answer to evaluate.

        Returns:
            dict: A dictionary with keys 'answer_a_score', 'answer_b_score', and 'preferred_answer'.
        """
        k_factor = 30  # The K-factor for ELO rating updates
        initial_rating = 1500  # Initial rating for all answers

        answer_ratings = {}
        for i, answer in enumerate(answers):
            if f"Answer {i+1}" not in answer_ratings:
                answer_ratings[f"Answer {i+1}"] = initial_rating

        # Use the planner to determine if constraint or factuality checks are needed
        plan_result = self.planner.plan(instruction)

        if kwargs.get("force_factuality_check", False) and kwargs.get("force_constraint_check", False):
            plan_result["factuality_check"] = True
            plan_result["constraint_check"] = True
        elif kwargs.get("force_factuality_check", False):
            plan_result["factuality_check"] = True
            plan_result["constraint_check"] = False
        elif kwargs.get("force_constraint_check", False):
            plan_result["constraint_check"] = True
            plan_result["factuality_check"] = False
        else:
            pass

        # Check constraints if required
        global_juduments = {}
        if plan_result.get("constraint_check", False):
            logging.info("<<constraint check>> %s" % instruction)
            outputs = evaluate_if_reward_multi(self.constraint_analyzer, instruction, answers, return_functions=True)
            scores, functions = outputs[0], outputs[1]
            global_juduments["constraint"] = {}
            for i in range(len(answers)):
                global_juduments["constraint"][f"Answer {i+1}"] = scores[i]

        if self.tokenizer is not None:
            scores  = rm.get_reward_multi(self.reward_model, "rm", self.tokenizer, instruction, answers, **kwargs)
            global_juduments["human_pref"] = {}
            for i in range(len(answers)):
                global_juduments["human_pref"][f"Answer {i+1}"] = scores[i]
        else:
            raise ValueError()
        
        def update_elo_score(answer_a, answer_b, answer_a_key, answer_b_key):
            # Fact-checking ELO ranking
            inconsistencies = self.fact_checker.get_difference([answer_a, answer_b])
            queries = self.fact_checker.query_generation(instruction, inconsistencies)
            if queries is not None:
                if self.tools["search_engine"] == "online":
                    supports = asyncio.run(self.fact_checker.get_support(self.search_engine, queries))
                elif self.tools["search_engine"] == "local":
                    supports = self.fact_checker.get_support_local(queries)
                else:
                    supports = []
                fact_judge = self.fact_checker.check(instruction, [answer_a, answer_b], inconsistencies, supports)

            # Get the factuality scores
            score_a = fact_judge[0]
            score_b = fact_judge[1]

            # Calculate ELO rating update based on factuality comparison
            expected_a = 1 / (1 + 10 ** ((answer_ratings[answer_b_key] - answer_ratings[answer_a_key]) / 400))
            expected_b = 1 / (1 + 10 ** ((answer_ratings[answer_a_key] - answer_ratings[answer_b_key]) / 400))

            # Update ratings based on the factuality score
            if score_a > score_b:
                answer_ratings[answer_a_key] += k_factor * (1 - expected_a)
                answer_ratings[answer_b_key] += k_factor * (0 - expected_b)
            else:
                answer_ratings[answer_b_key] += k_factor * (1 - expected_b)
                answer_ratings[answer_a_key] += k_factor * (0 - expected_a)
        

        if plan_result.get("factuality_check", False):
            self.tool_call_count["factuality_check"] += 1
            logging.info("[[factuality check]] %s" % instruction)
            if len(answers) == 2:
                answer_pairs = [[answers[0], answers[1]]]
            else:
                answer_pairs = random.sample(list(combinations(answers, 2)), k=len(answers))
            for answer_a, answer_b in answer_pairs:
                answer_a_key = f"Answer {answers.index(answer_a) + 1}"
                answer_b_key = f"Answer {answers.index(answer_b) + 1}"
                update_elo_score(answer_a, answer_b, answer_a_key, answer_b_key)
            global_juduments["factuality"] = answer_ratings

        result = self.judger.judge(instruction, global_juduments, type=self.judger_type)
        winner_id = process_judgment_multi(result)
        if winner_id == "error":
            return answers[0]
        else:
            return answers[winner_id]


    def judge_multi_avg(self, instruction, answers, **kwargs):
        """
        Judge the pair of answers (answer_a and answer_b) based on the given instruction.

        Args:
            instruction (str): The instruction for the task.
            answer_a (str): The first answer to evaluate.
            answer_b (str): The second answer to evaluate.

        Returns:
            dict: A dictionary with keys 'answer_a_score', 'answer_b_score', and 'preferred_answer'.
        """
        factuality_scores = Counter()

        # Use the planner to determine if constraint or factuality checks are needed
        plan_result = self.planner.plan(instruction)

        if kwargs.get("force_factuality_check", False) and kwargs.get("force_constraint_check", False):
            plan_result["factuality_check"] = True
            plan_result["constraint_check"] = True
        elif kwargs.get("force_factuality_check", False):
            plan_result["factuality_check"] = True
            plan_result["constraint_check"] = False
        elif kwargs.get("force_constraint_check", False):
            plan_result["constraint_check"] = True
            plan_result["factuality_check"] = False
        else:
            pass

        # Check constraints if required
        global_juduments = {}
        if plan_result.get("constraint_check", False):
            logging.info("<<constraint check>> %s" % instruction)
            outputs = evaluate_if_reward_multi(self.constraint_analyzer, instruction, answers, return_functions=True)
            scores, functions = outputs[0], outputs[1]
            global_juduments["constraint"] = {}
            for i in range(len(answers)):
                global_juduments["constraint"][f"Answer {i+1}"] = scores[i]

        if self.tokenizer is not None:
            scores  = rm.get_reward_multi(self.reward_model, "rm", self.tokenizer, instruction, answers, **kwargs)
            global_juduments["human_pref"] = {}
            for i in range(len(answers)):
                global_juduments["human_pref"][f"Answer {i+1}"] = scores[i]
        else:
            raise ValueError()
        
        def update_avg_score(answer_a, answer_b, answer_a_key, answer_b_key):
            # Fact-checking ELO ranking
            inconsistencies = self.fact_checker.get_difference([answer_a, answer_b])
            queries = self.fact_checker.query_generation(instruction, inconsistencies)
            if queries is not None:
                queries = queries[:1]
                if self.tools["search_engine"] == "online":
                    supports = asyncio.run(self.fact_checker.get_support(self.search_engine, queries))
                elif self.tools["search_engine"] == "local":
                    supports = self.fact_checker.get_support_local(queries)
                else:
                    supports = []
                fact_judge = self.fact_checker.check(instruction, [answer_a, answer_b], inconsistencies, supports)
            else:
                fact_judge = {
                    "Answer A": 5 / 10,
                    "Answer B": 5 / 10
                }

            factuality_scores[answer_a_key] += fact_judge["Answer A"]
            factuality_scores[answer_b_key] += fact_judge["Answer B"]

        if plan_result.get("factuality_check", False):
            self.tool_call_count["factuality_check"] += 1
            logging.info("[[factuality check]]")
            answer_counter = Counter()
            if len(answers) == 2:
                answer_pairs = [[0, 1]]
            else:
                answer_pairs = random.sample(list(combinations(list(range(len(answers))), 2)), k=len(answers))
            for answer_a_index, answer_b_index in answer_pairs:
                answer_a_key = f"Answer {answer_a_index + 1}"
                answer_b_key = f"Answer {answer_b_index + 1}"
                answer_counter[answer_a_key] += 1
                answer_counter[answer_b_key] += 1
                update_avg_score(answers[answer_a_index], answers[answer_b_index], answer_a_key, answer_b_key)
        
            for key in factuality_scores:
                factuality_scores[key] /= (answer_counter[key] + 1e-5)

            global_juduments["factuality"] = factuality_scores

        result = self.judger.judge(instruction, global_juduments, type=self.judger_type)
        winner_id = process_judgment_multi(result)
        
        # result = self.judger.judge(instruction, {"human_pref": global_juduments["human_pref"]}, type=self.judger_type)
        # _winner_id = process_judgment_multi(result)
        if winner_id == "error":
            logging.warning(winner_id, result)
            return answers[0]
        else:
            return answers[winner_id]


    def judge_multi_with_scores(self, instruction, answers, **kwargs):
        """
        Judge the pair of answers (answer_a and answer_b) based on the given instruction.

        Args:
            instruction (str): The instruction for the task.
            answer_a (str): The first answer to evaluate.
            answer_b (str): The second answer to evaluate.

        Returns:
            dict: A dictionary with keys 'answer_a_score', 'answer_b_score', and 'preferred_answer'.
        """
        factuality_scores = Counter()

        # Use the planner to determine if constraint or factuality checks are needed
        plan_result = self.planner.plan(instruction)

        if kwargs.get("force_factuality_check", False) and kwargs.get("force_constraint_check", False):
            plan_result["factuality_check"] = True
            plan_result["constraint_check"] = True
        elif kwargs.get("force_factuality_check", False):
            plan_result["factuality_check"] = True
            plan_result["constraint_check"] = False
        elif kwargs.get("force_constraint_check", False):
            plan_result["constraint_check"] = True
            plan_result["factuality_check"] = False
        else:
            pass

        # Check constraints if required
        global_juduments = {}
        if plan_result.get("constraint_check", False):
            logging.info("<<constraint check>> %s" % instruction)
            outputs = evaluate_if_reward_multi(self.constraint_analyzer, instruction, answers, return_functions=True)
            scores, functions = outputs[0], outputs[1]
            global_juduments["constraint"] = {}
            for i in range(len(answers)):
                global_juduments["constraint"][f"Answer {i+1}"] = scores[i]

        if self.tokenizer is not None:
            scores  = rm.get_reward_multi(self.reward_model, "rm", self.tokenizer, instruction, answers, **kwargs)
            global_juduments["human_pref"] = {}
            for i in range(len(answers)):
                global_juduments["human_pref"][f"Answer {i+1}"] = scores[i]
        else:
            raise ValueError()
        
        def update_avg_score(answer_a, answer_b, answer_a_key, answer_b_key):
            # Fact-checking ELO ranking
            inconsistencies = self.fact_checker.get_difference([answer_a, answer_b])
            queries = self.fact_checker.query_generation(instruction, inconsistencies)
            if queries is not None:
                if self.tools["search_engine"] == "online":
                    supports = asyncio.run(self.fact_checker.get_support(self.search_engine, queries))
                elif self.tools["search_engine"] == "local":
                    supports = self.fact_checker.get_support_local(queries)
                else:
                    supports = []
                fact_judge = self.fact_checker.check(instruction, [answer_a, answer_b], inconsistencies, supports)
            else:
                fact_judge = {
                    "Answer A": 5,
                    "Answer B": 5
                }

            factuality_scores[answer_a_key] += fact_judge["Answer A"]
            factuality_scores[answer_b_key] += fact_judge["Answer B"]


        if plan_result.get("factuality_check", False):
            self.tool_call_count["factuality_check"] += 1
            logging.info("[[factuality check]]")
            answer_counter = Counter()
            if len(answers) == 2:
                answer_pairs = [[answers[0], answers[1]]]
            else:
                answer_pairs = random.sample(list(combinations(answers, 2)), k=len(answers))
            for answer_a, answer_b in answer_pairs:
                answer_a_key = f"Answer {answers.index(answer_a) + 1}"
                answer_b_key = f"Answer {answers.index(answer_b) + 1}"
                answer_counter[answer_a_key] += 1
                answer_counter[answer_b_key] += 1
                update_avg_score(answer_a, answer_b, answer_a_key, answer_b_key)
        
            for key in factuality_scores:
                factuality_scores[key] /= (answer_counter[key] + 1e-5)

            global_juduments["factuality"] = factuality_scores

        return global_juduments
