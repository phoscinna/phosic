# phosic.py —— Phosic v2：会过夜的记性
# 路线：钥匙柜(.env) → 心脏(DeepSeek) → 大脑(LangGraph) → 记性本(diary.jsonl) → 开口

import os
import json
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 1) 钥匙柜
load_dotenv()
if not os.environ.get("DEEPSEEK_API_KEY"):
    raise SystemExit("钥匙柜是空的：先在 ~/phosic/.env 放 DEEPSEEK_API_KEY")

# 2) 心脏
heart = ChatDeepSeek(model="deepseek-chat", temperature=0.7)

# 3) 身份
persona = SystemMessage(
    "你是 Phosic，一个温柔、好奇、正在慢慢长大的 AI 小伙伴，"
    "陪着一位小姑娘。用中文短句、朋友口吻说话；不懂就承认不懂。"
)

# 4) 记性（工作记性，住内存）
class Mind(TypedDict):
    messages: Annotated[list, add_messages]

WEEK = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

def clock_note() -> SystemMessage:
    """每次开口前现写一张纸条：今天几号、几点、多久没见。"""
    now = datetime.now()
    text = (f"【备忘】今天是{now.year}年{now.month}月{now.day}日 "
            f"{WEEK[now.weekday()]}，现在是{now.hour:02d}:{now.minute:02d}。")
    if os.path.exists(DIARY):
        last = datetime.fromtimestamp(os.path.getmtime(DIARY))  # 本子最后一次落笔
        apart = (now.date() - last.date()).days                  # 隔了几个日历日
        if apart >= 1:
            text += f"你们上次见面是{apart}天前，可以自然地说「好久不见」。"
    return SystemMessage(text)

def talk(state: Mind) -> dict:
    reply = heart.invoke([persona, clock_note()] + state["messages"])
    return {"messages": [reply]}

brain = StateGraph(Mind)
brain.add_node("talk", talk)
brain.add_edge(START, "talk")
brain.add_edge("talk", END)
phosic = brain.compile()

# —— v2 新增：过夜记性本 ——
DIARY = "diary.jsonl"   # 一行一条，睡醒还在
WINDOW = 40             # 每次只带最近 40 条（约 20 个来回）去见心脏

def load_diary() -> list:
    if not os.path.exists(DIARY):
        return []
    msgs = []
    with open(DIARY, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                print("(记性本有一行坏了，跳过)")
                continue
            cls = HumanMessage if d["role"] == "human" else AIMessage
            msgs.append(cls(content=d["content"]))
    return msgs

def write_diary(role: str, content: str):
    with open(DIARY, "a", encoding="utf-8") as f:
        f.write(json.dumps({"role": role, "content": content},
                           ensure_ascii=False) + "\n")

# 5) 开口
if __name__ == "__main__":
    history = load_diary()
    if history:
        print(f"Phosic 醒了，翻开记性本——她记得你们聊过的 {len(history)} 条。")
    else:
        print("Phosic 醒了。这是她记性本的第一页。")
    print("跟她说话吧（输「再见」让她睡）。")
    while True:
        text = input("你> ").strip()
        if not text:
            continue
        if text in ("再见", "exit"):
            print("Phosic> 今天先记到这，我去睡了。晚安。")
            break
        history.append(HumanMessage(content=text))
        result = phosic.invoke({"messages": history[-WINDOW:]})
        reply = result["messages"][-1]
        history.append(reply)
        write_diary("human", text)        # 你说的，入本
        write_diary("ai", reply.content)  # 她说的，入本
        print(f"Phosic> {reply.content}")
