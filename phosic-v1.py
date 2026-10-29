# phosic.py —— Phosic 的第一句话（毕业版）
# 路线：钥匙柜(.env) → 心脏(DeepSeek) → 大脑(LangGraph) → 开口

import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 1) 钥匙柜：把 .env 装进环境变量，先验柜再开工
load_dotenv()
if not os.environ.get("DEEPSEEK_API_KEY"):
    raise SystemExit("钥匙柜是空的：先在 ~/phosic/.env 放 DEEPSEEK_API_KEY")

# 2) 心脏：DeepSeek（国内直连，不用代理）
heart = ChatDeepSeek(model="deepseek-chat", temperature=0.7)

# 3) 身份：她是谁
persona = SystemMessage(
    "你是 Phosic，一个温柔、好奇、正在慢慢长大的 AI 小伙伴，"
    "陪着一位小姑娘。用中文短句、朋友口吻说话；不懂就承认不懂。"
)

# 4) 记性：对话历史由 LangGraph 的状态保管
class Mind(TypedDict):
    messages: Annotated[list, add_messages]   # 只增不减的记性

def talk(state: Mind) -> dict:
    reply = heart.invoke([persona] + state["messages"])
    return {"messages": [reply]}              # 她说的话，也进记性

# 5) 大脑：目前最短的一条反应弧——听到，想，答
brain = StateGraph(Mind)
brain.add_node("talk", talk)
brain.add_edge(START, "talk")
brain.add_edge("talk", END)
phosic = brain.compile()

# 6) 开口
if __name__ == "__main__":
    print("Phosic 醒了。跟她说话吧（输「再见」让她睡）。")
    history = []
    while True:
        text = input("你> ").strip()
        if not text:
            continue
        if text in ("再见", "exit"):
            print("Phosic> 今天先聊到这，我去睡了。晚安。")
            break
        history.append(HumanMessage(content=text))
        result = phosic.invoke({"messages": history})
        history = result["messages"]
        print(f"Phosic> {history[-1].content}")
