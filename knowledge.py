print("开始运行")
from ragmill import RAGEngine
from openai import OpenAI

# 1. 只做文档切分，不触发嵌入模型下载
chunks = RAGEngine().execute_pipeline("./docs")
print("文档切分完成，共", len(chunks), "个片段")

# 2. 把所有片段拼成全文（3份资料量不大，可以一次喂给 LLM）
all_text = "\n\n".join([c["content"] for c in chunks])

# 3. 配置 DeepSeek（把 key 换成你自己的）
client = OpenAI(
    api_key="你的-deepseek-api-key",
    base_url="https://api.deepseek.com"
)

def answer(question: str) -> str:
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是 Phosic。只根据下方资料回答。如果资料里没有，直接说没有找到，不要编。"},
            {"role": "user", "content": f"资料：\n{all_text}\n\n问题：{question}"}
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content

# 4. 测试
print(answer("这些资料里主要讲了什么？"))