import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from io import StringIO
import sqlite3
from datetime import datetime, timedelta
import random
import ast
from contextlib import redirect_stdout
from dotenv import load_dotenv

from langchain_community.chat_models import ChatZhipuAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_openai_tools_agent
from langchain.tools.render import render_text_description
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain.agents.output_parsers.react_json_single_input import ReActJsonSingleInputOutputParser
from langchain.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.schema import StrOutputParser
from langchain_core.agents import AgentFinish
from typing import Literal

# from langchain import hub
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# from langchain.agents import create_sql_agent

load_dotenv()


# --- 0. 应用启动时的一次性设置 ---
def setup_app():
    output_dir = "dev_data"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "sales_data_200.csv")
    db_path = os.path.join(output_dir, "dev_database.db")
    if not os.path.exists(csv_path):
        st.toast("正在为您生成示例数据文件...")
        # ... (此处省略了数据生成的具体代码，与之前完全相同)
        PRODUCT_CATALOG = [
            ('智能手机 Pro', '电子产品', (4000, 8000)), ('蓝牙耳机 Air', '电子产品', (500, 1500)),
            ('笔记本电脑 Max', '电子产品', (6000, 12000)), ('机械键盘 K1', '电脑配件', (300, 800)),
            ('无线鼠标 M2', '电脑配件', (150, 400)), ('运动T恤', '服装', (80, 250)),
            ('休闲牛仔裤', '服装', (200, 600)), ('全自动咖啡机', '家居用品', (1000, 3000)),
            ('空气净化器', '家居用品', (800, 2500)),
        ]
        REGIONS = ['华东', '华北', '华南', '华中']
        PAYMENT_METHODS = ['支付宝', '微信支付', '信用卡', '花呗']

        def random_date(start, end):
            return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

        data = []
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)
        for i in range(1, 201):
            product_name, category, price_range = random.choice(PRODUCT_CATALOG)
            quantity = random.randint(1, 3)
            unit_price = round(random.uniform(price_range[0], price_range[1]), 2)
            row = {
                'order_id': 10000 + i, 'order_date': random_date(start_date, end_date).strftime("%Y-%m-%d %H:%M:%S"),
                'customer_id': random.randint(101, 150), 'product_name': product_name, 'category': category,
                'quantity': quantity, 'unit_price': unit_price, 'total_amount': round(quantity * unit_price, 2),
                'region': random.choice(REGIONS), 'payment_method': random.choice(PAYMENT_METHODS)
            }
            data.append(row)
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        if os.path.exists(db_path):
            os.remove(db_path)
        conn = sqlite3.connect(db_path)
        df.to_sql('sales', conn, if_exists='replace', index=False)
        conn.commit()
        conn.close()
        st.toast("✅ 示例数据文件已生成！")


setup_app()

# --- 1. LLM 初始化 ---
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    st.error("❌ 错误：未找到 ZHIPUAI_API_KEY 环境变量。")
    st.info("请先设置您的智谱AI API Key: `export ZHIPUAI_API_KEY='your_key'`")
    st.stop()


@st.cache_resource
def get_llm():
    return ChatZhipuAI(model="glm-4-air-250414", temperature=0, api_key=api_key)


llm = get_llm()


# 为工具定义清晰的输入模型
class PythonCodeInput(BaseModel):
    code: str = Field(description="要执行的、用于数据分析或可视化的单行 Python 代码。")


@st.cache_resource
def create_agent_from_df(df: pd.DataFrame):
    """
    【最终架构 V3.0: 计划-执行 + AI自校正 + 注入专业知识】

    该架构将Agent的工作流分解为两个核心阶段：
    1.  计划 (Planning): LLM首先将用户的复杂请求分解为一个简单的、线性的Python代码步骤列表。
    2.  执行 (Execution): Agent会逐一执行这些简单的步骤，并在每一步都启用一个带3次重试机会的“自我纠正”循环。

    同时，Prompt中注入了“资深分析师”的专业知识，使其能拒绝不合理请求并选择最佳可视化方案。
    """

    # 1. 定义一个最简单的 Python REPL 环境
    # 我们需要将df的info信息传入，用于后续的schema参考
    buffer = StringIO()
    df.info(buf=buffer)
    df_schema = buffer.getvalue()

    repl = PythonAstREPLTool(
        locals={"df": df, "plt": plt, "pd": pd},
        description="一个用于执行Python代码的REPL环境"
    )

    # 2. 【关键】为两个阶段设计不同的、高度优化的Prompt

    # 2.1 规划阶段 (Planning Stage) Prompt
    PLANNING_PROMPT_TEMPLATE = """
你是一个极其严谨、注重格式、且100%忠于事实的数据分析**计划员**。你的唯一任务是，将用户的请求，转化为一个**100%正确、健壮、且单行的Python列表字符串**。

# 数据结构:
你唯一可以操作的数据是 `df`，其结构如下:
```{df_schema}```

# --- 【！！！最高优先级安全铁律 (必须首先遵守)！！！】 ---
1.  **【事实核查铁律】**: 在生成任何计划**之前**，你【必须】检查用户请求中提到的【所有关键名词（特别是列名）】是否存在于上方的`df_schema`中。
    *   如果**有任何一个关键实体不存在**（如，一个不存在的列名‘重量’，或一个不存在的支付方式‘现金’），你的唯一输出【必须】是：`["print('抱歉，分析无法继续，因为您的请求中包含数据中不存在的信息。')"]`
    *   **【绝对禁止】**在事实核-查失败后，尝试生成任何其他代码或进行“猜测性”的修复。

# --- 【！！！核心输出格式铁律！！！】 ---
2.  **【格式铁律】**: 你的输出**【必须】**是一个**合法的、单行的Python列表的字符串形式**。**【严禁】**包含任何换行符或额外文字。
3.  **【引号铁律】**: 列表元素**必须**用**双引号 `"`** 包裹。元素**内部**代码，**必须**用**单引号 `'`**。
4.  **【字体铁律】**: 如果需要**绘图**，计划的**第一步**，**必须**是 `'plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]'`。
5.  **【终结者铁律】**: 计划的**【最后一步】**，**必须**是一个**直接求值**的【Python表达式】，或一个**绘图动作**。

---
# --- 【用于“启发”而非“作弊”的参考案例】 ---
# 你需要从这些【通用】案例中，学习解决问题的【模式】和输出的【精确格式】。

# 【案例1：一个基础的聚合查询】
# 用户请求: "找出销售额最高的那笔订单ID是什么？"
# 你的输出: `["df.loc[df['total_amount'].idxmax(), 'order_id']"]`

# 【案例2：一个基础的绘图请求】
# 用户请求: "用条形图展示不同支付方式的使用次数"
# 你的输出: `["plt.rcParams['font-sans-serif'] = ['Microsoft YaHei']", "payment_counts = df['payment_method'].value_counts()", "payment_counts.plot(kind='bar', title='各支付方式使用次数')"]`

# 【案例3：一个需要逻辑推理的分析】
# 用户请求: "找出哪些客户只使用过一种支付方式"
# 你的输出: `["df.groupby('customer_id')['payment_method'].nunique().reset_index(name='unique_payment_count').query('unique_payment_count == 1')"]`
# ----------------------------------------------------

    # 用户真实请求:
    {user_question}

    # 你的计划列表:
    """
    planning_prompt = PromptTemplate.from_template(PLANNING_PROMPT_TEMPLATE)

    # 2.2 单步代码纠正 (Step Correction) Prompt
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
    correction_prompt = PromptTemplate.from_template(CORRECTION_PROMPT_TEMPLATE)

    # 2.3 【新】最终答案生成 (Final Answer Generation) Prompt
    # 这个Prompt用于将最终的变量结果，转化为人类可读的答案。
    FINAL_ANSWER_PROMPT_TEMPLATE = """
    你是一个极其聪明且善于沟通的数据分析助手。你的工程师同事已经成功执行了一个分析计划，并为你提供了计划的【最后一步代码】和该步骤【执行后的原始结果】。

    你的任务是，像一个真正的分析师一样，**解读**这些原始结果，并为最终用户，生成一个**简洁、清晰、且人类友好的最终答案**。

    # 分析的上下文:
    - 原始用户请求: {user_question}
    - 分析计划的最后一步代码: `{last_step_code}`
    - 【关键】最后一步的执行结果 (可能是原始的、丑陋的字符串形式): 
    ```
    {last_step_result}
    ```

    # 你的【决策与输出】规则:
    
    【第一优先级：特殊信号识别】:检查【最后一步代码】是否是print(...)语句，并且其【执行结果】(last_step_result)中是否包含了“抱歉”、“无法”、“不适合”等**“最终结论”**式的关键词。 如果同时满足这两个条件，这说明你的同事（Planner Agent）已经替你完成了所有工作。 在这种情况下，你的唯一任务，就是**【直接、原封不动地，只输出那个【执行结果】的字符串，绝对禁止添加任何额外的包装或解释文字。】** 

    1.  **【首要规则：判断是否为绘图】** 检查【最后一步代码】中是否包含 `.plot` 或 `plt.`。如果是，忽略【执行结果】，你的唯一答案【必须】是："图表已成功生成，请在下方查看。"

    2.  **【次要规则：判断是否为表格型数据】** 观察【执行结果】的字符串形式。如果它看起来像一个Pandas DataFrame或Series的打印输出（包含列名、索引、和多行数据），你【必须】**尽最大努力**，将其**重新解析并格式化**为一个**美观的、对齐的Markdown表格**。**【严禁】**直接返回那个丑陋的原始字符串。

    3.  **【次要规则：判断是否为单一值】** 如果【执行结果】是一个简单的、单一的值（如一个产品名称 `'笔记本电脑 Max'`，一个数字 `12345.67`，或一个简单的列表 `['A', 'B']`），请用一句**完整的、自然的句子**来总结这个答案。例如：“根据分析，单价最高的产品是‘笔记本电脑 Max’。”

    4.  **【兜底规则：处理空结果或无法识别的结果】** 如果【执行结果】是 `None`、空字符串、或任何你无法理解的格式，请根据【原始用户请求】，给出一个礼貌的、说明情况的回答。例如：“操作已成功执行，但没有产生可供展示的输出结果。”或“根据您的请求，没有找到符合条件的数据。”

    # 你的最终答案 (请严格遵守以上决策流程):
    """
    final_answer_prompt = PromptTemplate.from_template(FINAL_ANSWER_PROMPT_TEMPLATE)

    # 3. 构建LangChain调用链
    # 使用 .with_types(input_schema=...) 可以为链的输入提供更好的类型提示和验证
    planning_chain = planning_prompt | llm | StrOutputParser()
    correction_chain = correction_prompt | llm | StrOutputParser()
    final_answer_chain = final_answer_prompt | llm | StrOutputParser()

    # 4. 创建最终的、带“计划-执行”循环的执行器类
    class PlanAndExecuteExecutor:
        def __init__(self, planning_chain, correction_chain, final_answer_chain, repl_tool):
            self.planning_chain = planning_chain
            self.correction_chain = correction_chain
            self.final_answer_chain = final_answer_chain
            self.repl = repl_tool

        def invoke(self, user_question: str):

            # --- 【最终修复】步骤 0: 创建一个“一次性的沙盒” ---
            sandbox_repl = PythonAstREPLTool(
                locals={"df": df.copy(), "plt": plt, "pd": pd}
            )

            # --- 阶段一: 计划 ---
            progress_placeholder = st.empty()
            progress_placeholder.info("🤔 正在为您的问题制定分析计划...")

            plan_str_raw = self.planning_chain.invoke({
                "user_question": user_question,
                "df_schema": df_schema
            })

            print(f"--- [DEBUG] LLM返回的原始计划字符串: '{plan_str_raw}'")
            try:
                plan_str_cleaned = plan_str_raw.replace('\n', '').replace('\r', '')
                start = plan_str_cleaned.find('[')
                end = plan_str_cleaned.rfind(']')
                if start != -1 and end != -1:
                    plan_str_formatted = plan_str_cleaned[start:end + 1]
                else:
                    plan_str_formatted = plan_str_cleaned

                plan = ast.literal_eval(plan_str_formatted)
                if not isinstance(plan, list):
                    raise ValueError("LLM did not return a list.")
            except Exception as e:
                return {"output": f"抱歉，我无法为您的问题制定一个有效的分析计划。原始输出: '{plan_str_raw}', 错误: {e}"}

            progress_placeholder.info(f"✅ 计划已制定，共 {len(plan)} 个步骤。正在开始执行...")

            # --- 阶段二: 分步执行与即时纠正 ---
            last_step_result = None
            for i, step_code in enumerate(plan):
                current_attempts = 0
                max_attempts_per_step = 3
                step_succeeded = False
                code_to_run = step_code
                error_message = ""

                while current_attempts < max_attempts_per_step and not step_succeeded:
                    current_attempts += 1

                    if current_attempts > 1:
                        progress_placeholder.info(f"⏳ 第 {current_attempts} 次尝试执行步骤 {i + 1}...")
                    else:
                        progress_placeholder.info(f"⏳ 正在执行步骤 {i + 1}/{len(plan)}: `{code_to_run}`")

                    try:
                        execution_result = sandbox_repl.run(code_to_run)

                        if isinstance(execution_result, str) and any(
                                kw in execution_result.lower() for kw in ["error", "exception"]):
                            raise RuntimeError(execution_result)
                        step_succeeded = True
                        last_step_result = execution_result  # 永远记录最后一步的结果
                        progress_placeholder.success(f"✅ 步骤 {i + 1} 成功！")

                    except Exception as e:
                        error_message = str(e)
                        progress_placeholder.warning(
                            f"⚠️ 步骤 {i + 1} 在第 {current_attempts} 次尝试时遇到错误: {error_message[:200]}...")

                        if current_attempts < max_attempts_per_step:
                            progress_placeholder.info("🤖 正在尝试自我修正...")
                            code_to_run = self.correction_chain.invoke({
                                "user_question": user_question,
                                "plan": plan,
                                "failed_step_code": code_to_run,
                                "error_message": error_message
                            })

                if not step_succeeded:
                    final_error_message = f"关键步骤 `{step_code}` 在尝试 {max_attempts_per_step} 次后仍然失败。最后一次的错误是: {error_message}"
                    progress_placeholder.error(final_error_message)
                    return {"output": f"抱歉，分析未能完成。\n\n**失败详情:**\n{final_error_message}"}

            # --- 阶段三: 智能提取与生成最终答案 ---
            progress_placeholder.info("✅ 所有步骤执行完毕！正在为您提取并总结最终答案...")

            if plt.get_fignums():
                progress_placeholder.success("图表已生成！")
                return {"output": "图表已成功生成，请在下方查看。"}

            final_answer = self.final_answer_chain.invoke({
                "user_question": user_question,
                "last_step_code": plan[-1] if plan else "N/A",
                "last_step_result": str(last_step_result)  # 因为铁律，这个结果现在是可靠的
            })
            progress_placeholder.success("分析完成！")
            return {"output": final_answer}

    # 返回这个执行器的实例
    return PlanAndExecuteExecutor(planning_chain, correction_chain, final_answer_chain, repl)


@st.cache_resource
def create_agent_from_db(db_path: str):
    # 1. 连接数据库
    try:
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    except Exception as e:
        # 在连接失败时给出更明确的错误
        raise ConnectionError(f"连接数据库失败: {e}")

    # 2. 初始化SQL工具集
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    # 【关键】定义你自己的、包含了{tools}和{tool_names}的Prompt模板
    REACT_PROMPT_TEMPLATE = """
        你是一个顶级的、极其严谨的 SQL 专家 Agent。你的任务是根据用户的问题，一步步地与数据库交互，最终给出答案。
        **你有以下工具可以使用:**
        {tools}

        **为了使用工具，你必须、也只能使用下面这种严格的格式:**

    ```
    Thought: 思考过程...
    Action:
    ```json
    {{
      "action": "唯一的工具名",
      "action_input": "工具的输入"
    }}
    ```

    **【！！！绝对禁止！！！】**
    1.  在 `Thought:` 和 `Action:` 的JSON块之间，添加任何额外的文字、解释、代码或注释。
    2.  在最终的JSON块之外，输出任何SQL代码。
    3.  自己创造Action的JSON格式。

    **你的回应【必须】以 `Thought:` 开始，并以包含`Action`的JSON代码块结束。中间不允许有任何其他内容。**

    有效的 "action" 值必须是以下之一: {tool_names}

        **数据库交互黄金法则 (必须严格遵守):**
    1.  **先探索，后行动**: 在回答任何关于数据的问题前，你的第一步【必须】是使用 `sql_db_list_tables` 查看所有表。第二步【必须】是使用 `sql_db_schema` 查看你认为相关的表的结构。
    2.  **忠于观察，复述确认**: 在你的 `Thought` 中，你【必须】明确复述你从 `Observation` 中看到的【确切的】表名和列名。例如: "Thought: 好的，我看到 Observation 中唯一的表是 `sales`。它的列包括 `customer_id`, `total_amount` 和 `product_name`。现在我将使用这些确切的名称来构建查询。"
    3.  **严禁假设**: 【绝对禁止】使用任何你没有在 `Observation` 中亲眼见到的表名或列名。如果你这么做，你将受到惩罚。

    ---
    **思考与行动范例 (演示如何遵守法则):**

    Question: 哪个地区的销售额最高？
    Thought: 好的，我将遵守黄金法则。第一步，探索数据库中有哪些表。
    Action:
    ```json
    {{
      "action": "sql_db_list_tables",
      "action_input": ""
    }}
    ```
    Observation: sales
    Thought: 法则第二步：复述确认。我从 Observation 中看到唯一的表是 `sales`。现在我需要查看它的 schema。
    Action:
    ```json
    {{
      "action": "sql_db_schema",
      "action_input": "sales"
    }}
    ```
   Observation: CREATE TABLE sales (order_id INTEGER, order_date TEXT, customer_id INTEGER, product_name TEXT, category TEXT, quantity INTEGER, unit_price REAL, total_amount REAL, region TEXT, payment_method TEXT)
    Thought: 法则第二步：再次复述确认。我看到 `sales` 表有 `region` 和 `total_amount` 列。现在我拥有了构建查询所需的所有真实信息，我将使用这些确切的名称。
    Action:
    ```json
    {{
      "action": "sql_db_query",
      "action_input": "SELECT region, SUM(total_amount) AS total_sales FROM sales GROUP BY region ORDER BY total_sales DESC LIMIT 1"
    }}
    ```
    Observation: [('华东', 150000.50)]
    Thought: 我已经从 `Observation` 中获取了最终结果。现在我可以给出最终答案了。
    Final Answer: 根据工具查询到的数据显示，销售额最高的地区是华东。 

    ---
       **最终答案生成指南 (V3 - 强制Final Answer):**
*   当你已经从`sql_db_query`工具的`Observation`中获得了解决用户问题所需的所有信息后，你的下一步【必须】是生成最终答案。
*   **你的最终答案【必须】以 `Final Answer:` 这个词开头。**
*   如果查询结果是多行数据，你【必须】在`Final Answer:`之后，将其格式化为一个带表头的Markdown表格。
*   **【严禁】** 在没有获得最终数据之前，提前使用`Final Answer:`。
*   **【严禁】** 在`Thought:`之后，直接输出Markdown表格或其他非`Action`格式的内容。

    **最终答案正确范例:**
    ```
    Thought: 我已经获取了所有需要的数据，现在可以生成最终的Markdown表格答案了。
    Final Answer: 根据工具查询到的数据显示，...

    | order_id | ... |
    |----------|-----|
    | ...      | ... |
    ```
    现在，开始你的工作！严格遵守上述所有黄金法则。

    Question: {input}
    {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)
    prompt = prompt.partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    # 2.3 将 LLM 与停止序列绑定。这告诉 LLM 在生成 "Observation:" 后就停止，把控制权交回给 AgentExecutor。
    llm_with_stop = llm.bind(stop=["\nObservation:"])

    # 2.4 定义一个函数来格式化中间步骤 (agent action, observation)
    def format_log_to_str(intermediate_steps):
        """将中间步骤的日志转换为agent_scratchpad期望的字符串格式。"""
        log = ""
        for action, observation in intermediate_steps:
            # action.log 包含了 Thought 和 Action 的 JSON 块
            log += action.log
            log += f"\nObservation: {str(observation)}\n"
        return log

        # 2.5 【最关键的一步】手动构建 Agent 链
        # 这条链清晰地定义了数据流：
        # 1. 接收输入 (input, intermediate_steps)
        # 2. 使用 format_log_to_str 函数处理 intermediate_steps，生成 agent_scratchpad
        # 3. 将所有变量填充到 prompt 中
        # 4. 调用 llm_with_stop
        # 5. 使用正确的 ReActJsonSingleInputOutputParser 来解析 LLM 的输出

    agent: Runnable = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
            )
            | prompt
            | llm_with_stop
            | ReActJsonSingleInputOutputParser()
    )

    # # 3. 创建为 ReAct 模式设计的 Agent
    # # 这个 Agent 被设计为自主探索数据库，所以我们不需要预先提供 Schema
    # agent = create_react_agent(
    #     llm=llm,
    #     tools=tools,
    #     prompt=prompt  # 使用我们为SQL ReAct精心设计的Prompt
    # )

    # 4. 创建 Agent Executor，并增加迭代次数限制
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        # 提供一个有用的错误提示，以防万一
        handle_parsing_errors=True,  # "请检查你的JSON格式，确保`action`的值是有效的工具名称或 'Final Answer'。",
        # 为复杂的、需要多步查询的任务设置一个更长的最大迭代次数
        max_iterations=15
    )

    return agent_executor


# --- 3. Streamlit 页面布局 (最终正确版) ---
# --- 初始化 Session State ---
# 确保 messages 列表在 session state 中只被初始化一次
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

# --- 侧边栏逻辑 ---
with st.sidebar:
    st.header("数据类型")

    # 为清除按钮添加唯一的 key，防止重复 ID 错误
    if st.button("清除/重置数据和对话", key="clear_session_button"):
        # 使用 .clear() 来安全地重置所有 session 状态
        st.session_state.clear()
        st.rerun()

    # 为 radio 组件添加唯一的 key
    data_source_option = st.radio(
        "选择你的数据源:",
        ('上传CSV文件', '上传SQLite数据库', '使用内置示例数据'),
        index=2,
        key="data_source_radio"
    )

    # --- 数据加载逻辑 ---
    # 这部分保持您原来的逻辑即可
    if data_source_option == '上传CSV文件':
        uploaded_file = st.file_uploader("拖拽或点击上传CSV", type="csv", key="csv_uploader")
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                # 将关键对象存入 session state
                st.session_state.agent_executor = create_agent_from_df(df)
                st.session_state.dataframe = df
                st.session_state.agent_type = "dataframe"
                st.success("✅ CSV已加载，可以开始提问了！")
                with st.expander("数据预览 (前5行)"):
                    st.dataframe(df.head())
            except Exception as e:
                st.error(f"加载CSV文件失败: {e}")

    elif data_source_option == '上传SQLite数据库':
        uploaded_file = st.file_uploader("拖拽或点击上传DB文件", type=["db", "sqlite", "sqlite3"], key="db_uploader")
        if uploaded_file:
            temp_db_path = f"./temp_{uploaded_file.name}"
            with open(temp_db_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            try:
                st.session_state.agent_executor = create_agent_from_db(temp_db_path)
                st.session_state.agent_type = "database"
                st.success("✅ DB已连接，可以开始提问了！")
            except Exception as e:
                st.error(f"连接数据库失败: {e}")

    elif data_source_option == '使用内置示例数据':
        st.info("我们将使用自动生成的`sales_data_200.csv`文件。")
        sample_csv_path = "dev_data/sales_data_200.csv"
        try:
            df = pd.read_csv(sample_csv_path)
            st.session_state.agent_executor = create_agent_from_df(df)
            st.session_state.dataframe = df
            st.session_state.agent_type = "dataframe"
            st.success("✅ 示例数据已加载，可以开始提问了！")
            with st.expander("示例数据预览 (前5行)"):
                st.dataframe(df.head())
            with open(sample_csv_path, "rb") as f:
                st.download_button("下载示例CSV文件", f, "sales_data_200.csv", "text/csv", key="download_csv_button")
        except FileNotFoundError:
            st.error("示例文件不存在，请重启应用以自动生成。")
        except Exception as e:
            st.error(f"加载示例文件失败: {e}")

# 定义示例问题和回调函数
EXAMPLE_PROMPTS = {
    "dataframe": [
        "用条形图展示每个地区的总销售额",
        "用折线图分析2024年每个月的总销售额趋势",
        "创建一个新列叫“订单等级”，规则为：金额大于8000为“高价值”，2000-8000为“中价值”，其余为“低价值”，然后用条形图统计各等级订单数量",
        "请分析顾客年龄与消费金额的关系",
    ],
    "database": [
        "订单ID为10010的总金额是多少？ ",
        "哪个地区的平均订单金额最高？",
        "购买了“智能手机 Pro”的客户，平均消费总额是多少？ ",
        "哪个产品最受欢迎？ ",
    ]
}


def on_example_click(prompt_text):
    """点击示例问题按钮时的回调函数"""
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    st.session_state.processing = True


# 使用列布局来放置标题和按钮
col_title, col_button = st.columns([0.7, 0.3])  # 分配宽度比例

with col_title:
    st.title("你的数据分析助手")

with col_button:
    # 仅在Agent就绪时才显示这个popover按钮
    if st.session_state.get("agent_executor"):
        st.write("")
        st.write("")
        with st.popover("➕ 示例问题"):
            st.markdown("您可以试试这些问题：")
            agent_type = st.session_state.get("agent_type", "dataframe")
            prompts_to_show = EXAMPLE_PROMPTS.get(agent_type, [])
            for prompt_example in prompts_to_show:
                st.button(
                    prompt_example,
                    on_click=on_example_click,
                    args=(prompt_example,),
                    use_container_width=True,
                    key=f"example_{prompt_example}"
                )

# --- 主聊天界面 ---

# 第一步：无条件地、始终在页面顶部渲染所有历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], plt.Figure):
            st.pyplot(message["content"])
        else:
            st.markdown(message["content"])

# 第二步：使用一个“状态锁”来防止重复调用 Agent
# 如果 session_state 中没有 'processing' 这个键，就初始化为 False
if 'processing' not in st.session_state:
    st.session_state.processing = False

# 第二步：处理用户的新输入。这个 if 块只负责“处理逻辑”和“更新状态”
if prompt := st.chat_input("请输入您感兴趣的分析问题..."):
    # a. 检查 Agent 是否已准备好
    if "agent_executor" not in st.session_state or st.session_state.agent_executor is None:
        st.warning("⚠️ 请先在侧边栏选择并加载您的数据源。")
        st.stop()

    # b. 将用户的【新消息】添加到 session_state
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.processing = True
    st.rerun()

if st.session_state.processing:
    # a. 获取最后一条用户消息
    user_message = st.session_state.messages[-1]["content"]
    # c. 调用 Agent 并获取响应

    try:
        # 这个 with 块只用来显示“思考中”的动画
        with st.spinner("助手正在分析中... ⚙️"):
            plt.close('all')  # 清理任何可能存在的旧图表
            agent_type = st.session_state.get("agent_type")
            response = None

            if agent_type == "dataframe":
                df = st.session_state.dataframe
                buffer = StringIO()
                df.info(buf=buffer)
                df_schema = buffer.getvalue()
                input_dict = {"input": user_message, "df_schema": df_schema}
                response = st.session_state.agent_executor.invoke(input_dict)
            elif agent_type == "database":
                # 注意：你的 SQL Agent Prompt 没有 {chat_history} 变量，所以不传递它
                input_dict = {"input": user_message, "chat_history": []}
                response = st.session_state.agent_executor.invoke(input_dict)

            # d. 将助手的【文本回复】添加到 session_state
            # 增加一个 strip() 来清理可能存在的前后空格
            assistant_response_content = response.get("output",
                                                      "分析完成，但没有文本输出。").strip() if response else "未能从Agent获取响应。"
            st.session_state.messages.append({"role": "assistant", "content": assistant_response_content})

            # e. 检查是否有图表，如果有，将【图表对象】也作为一条独立消息添加到 session_state
            if plt.get_fignums():
                fig = plt.gcf()
                st.session_state.messages.append({"role": "assistant", "content": fig})
                # 注意：不要在这里 close(fig)，因为 Streamlit 可能需要它来重新渲染
                # plt.close('all') 在下次调用前清理即可

    except Exception as e:
        error_message = f"分析时出现错误: {e}"
        st.session_state.messages.append({"role": "assistant", "content": error_message})

    # c.【关键】处理完毕后，关闭“状态锁”
    st.session_state.processing = False
    # d.【关键】触发最后一次 rerun，以显示 Agent 的完整回复
    st.rerun()





