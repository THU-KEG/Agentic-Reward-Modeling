from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union, List
from tools import ConstraintAnalyzer, evaluate_if_reward_multi
from build_model import APIModel
import itertools


# 定义输入的请求体
class RewardRequest(BaseModel):
    instruction: str
    answers: List[str]


# 初始化 FastAPI 实例
app = FastAPI()

api_urls = [
]

# 使用 itertools.cycle 来实现轮询负载均衡
api_url_cycle = itertools.cycle(api_urls)

# 选择一个 LocalAPIModel
def get_next_apimodel():
    next_url = next(api_url_cycle)
    return APIModel(next_url[0], next_url[1])

# 创建一个分析器实例
def get_analyzer():
    apimodel = get_next_apimodel()
    return ConstraintAnalyzer(apimodel)

@app.post("/evaluate/")
async def evaluate(request: RewardRequest):
    try:
        print(request)
        analyzer = get_analyzer()
        result = evaluate_if_reward_multi(analyzer, request.instruction, request.answers)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

