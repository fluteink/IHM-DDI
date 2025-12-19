import json

import requests
from zhipuai import ZhipuAI


def call_glm(prompt, temperature=0.5, max_tokens=2048):
    """
    调用智谱AI的GLM-4-Flash模型
    参数：
        prompt: 输入的提示语
        temperature: 控制生成随机性的参数（0-1），建议默认0.8[3](@ref)
        max_tokens: 生成的最大token数（最高支持2048）
    """
    client = ZhipuAI(
        api_key="2d6978606d436ad01ff3a3fd1c2abda3.4GteySa2wjR6tTxy"  # 替换为您的API Key
    )

    try:
        response = client.chat.completions.create(
            model="glm-4-flash",  # 指定Flash版本[3,5](@ref)
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False  # 同步调用模式[3](@ref)
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"API调用异常：{str(e)}"
def call_qwen(prompt, temperature=0.5, max_tokens=2048):
    """
    调用本地部署的Qwen2.5模型
    参数：
        prompt: 输入的提示语
        temperature: 控制生成随机性的参数（0-1）
        max_tokens: 生成的最大token数
    """
    url = "http://localhost:8000/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY"  # 如果不需要认证可以保留这个header
    }

    data = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(data),
            timeout=30
        )
        response.raise_for_status()  # 检查HTTP错误

        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            return "Error: Unexpected response format"

    except requests.exceptions.RequestException as e:
        return f"Request error: {str(e)}"
    except json.JSONDecodeError:
        return "Error: Failed to parse JSON response"
