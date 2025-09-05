# 这是一个【单元测试】，专门用于验证“AI自校正”模块的核心能力

import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
# 假设您的get_llm函数和一些基础设置在app.py中
from app import get_llm

# ----------------------------------------------------------------------
# 步骤1：将被测试的“单元”——纠正模块，完整地复制到这里
# ----------------------------------------------------------------------
CORRECTION_PROMPT_TEMPLATE = """
你是一个顶级的Python代码调试专家。你之前的同事在执行一个分析计划的某个步骤时失败了。
你的唯一任务是，分析**失败的代码**和**具体的错误信息**，然后生成一行**【修正后】**的、全新的Python代码来完成这个步骤。

# 分析计划的上下文:
- 原始用户请求: {user_question}
- 完整的分析计划: {plan}
- 当前失败的步骤: `{failed_step_code}`
- 执行时遇到的错误: `{error_message}`

# 你的输出规则:
1.  你的输出**【必须】**只有一行修正后的Python代码。
2.  **【严禁】**包含任何解释、道歉或额外的文字。
3.  修正后的代码必须与原步骤的目标保持一致。

# 修正后的单行Python代码:
"""


# ----------------------------------------------------------------------
# 步骤2：设计一个极其简单的、专门用于调用和展示的测试函数
# ----------------------------------------------------------------------
def run_correction_test(chain, faulty_code, error_message, test_name):
    """直接调用correction_chain，并打印结果"""
    print(f"\n===== [开始单元测试: {test_name}] =====")

    # 准备模拟的输入
    inputs = {
        "user_question": "这是一个模拟测试，旨在验证纠错能力",
        "plan": [faulty_code],  # 简化计划
        "failed_step_code": faulty_code,
        "error_message": error_message
    }

    print(f"--- [输入] 失败的代码: `{faulty_code}`")
    print(f"--- [输入] 错误信息: `{error_message}`")

    # 【核心调用！】
    corrected_code = chain.invoke(inputs)

    print(f"--- [输出] AI生成的修正代码: `{corrected_code}`")

    # 进行一个简单的、可视化的判断
    if corrected_code != faulty_code and len(corrected_code) > 5:
        print("✅ [结论] 测试通过！AI成功生成了与原始错误代码不同的、新的修正代码。")
    else:
        print("❌ [结论] 测试失败！AI未能生成有效的修正代码。")

    print(f"===== [测试结束: {test_name}] =====")


# ----------------------------------------------------------------------
# 步骤3：主执行脚本
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 准备环境
    llm = get_llm()
    correction_prompt = PromptTemplate.from_template(CORRECTION_PROMPT_TEMPLATE)
    correction_chain = correction_prompt | llm | StrOutputParser()

    # --- 测试用例1：语法错误 ---
    code1 = "df.groupby('region')['total_amount'].sum("
    error1 = "SyntaxError: unexpected EOF while parsing"
    run_correction_test(correction_chain, code1, error1, "语法错误纠正能力测试")

    # --- 测试用例2：逻辑错误 ---
    code2 = "df.groupby('cat')['total_amount'].sum()"
    error2 = "KeyError: 'cat'"
    run_correction_test(correction_chain, code2, error2, "逻辑错误纠正能力测试")