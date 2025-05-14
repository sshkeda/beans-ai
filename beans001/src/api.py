from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils.reward_function import calculate_reward
import uvicorn

app = FastAPI()


class RewardRequest(BaseModel):
    fen: str
    llm_response: str
    target: str
    question_type: str


@app.post("/calculate-reward")
async def get_reward(request: RewardRequest):
    try:
        return calculate_reward(
            request.llm_response,
            request.target,
            request.question_type,
            request.fen,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ping")
async def ping():
    return "pong"


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
