import sys
print(sys.executable)
import streamlit as st
import pandas as pd
import subprocess
import os
import tempfile
from openai import OpenAI

# ==========================
# 页面配置
# ==========================
st.set_page_config(page_title="AI驱动R语言全流程实证系统", layout="wide")

# ==========================
# 全局状态
# ==========================
if "data_files" not in st.session_state:
    st.session_state.data_files = {}

if "research_plan" not in st.session_state:
    st.session_state.research_plan = ""

if "r_preprocess_code" not in st.session_state:
    st.session_state.r_preprocess_code = ""

if "r_reg_code" not in st.session_state:
    st.session_state.r_reg_code = ""

if "df" not in st.session_state:
    st.session_state.df = None

if "page" not in st.session_state:
    st.session_state.page = 0

if "dag_elements" not in st.session_state:
    st.session_state.dag_elements = []
if "confounding_check" not in st.session_state:
    st.session_state.confounding_check = ""
if "backdoor_check" not in st.session_state:
    st.session_state.backdoor_check = ""
if "node_attr_input" not in st.session_state:
    st.session_state.node_attr_input = ""

# ==========================
# AI API
# ==========================
try:
    api_key = st.secrets["SILICON_API_KEY"]
except:
    api_key = None

base_url = "https://api.siliconflow.cn/v1"
MODEL = "deepseek-ai/DeepSeek-V3.2"
# deepseek-ai/DeepSeek-V3.2
# deepseek-ai/DeepSeek-R1-0528-Qwen3-8B

# ==========================
# 页面
# ==========================
pages = [
    "🤖 AI 研究方案",
    "📊 AI 数据预处理（R代码）",
    "🚀 运行 R 全流程",
    "📈 结果汇总",
    "📐 DAG 因果图分析"   # 新增第5页
]

# ==========================
# 翻页
# ==========================
def go(page):
    st.session_state.page = page
    st.rerun()

# ==========================
# 标题
# ==========================
st.title("📊 因果推断与政策评估系统")
st.markdown(f"## {pages[st.session_state.page]}")
st.divider()

col1, _, col3 = st.columns([1, 6, 1])
with col1:
    if st.session_state.page > 0:
        st.button("← 上一步", on_click=go, args=(st.session_state.page-1,))
with col3:
    if st.session_state.page < len(pages)-1:
        st.button("下一步 →", on_click=go, args=(st.session_state.page+1,))
st.divider()

# ==============================================================================
# 第1页：AI 生成研究方案
# ==============================================================================
if st.session_state.page == 0:
    st.subheader("输入研究问题 → AI 生成完整实证方案")
    q = st.text_area("例如：数字化转型对企业生产率的影响，控制企业特征，固定效应+行业聚类")

    if st.button("🧠 生成研究方案"):
        if not api_key:
            st.warning("请配置 API Key")
        else:
            with st.spinner("AI 思考中..."):
                c = OpenAI(api_key=api_key, base_url=base_url)
                prompt = f"""
                你是计量经济学顶级专家，生成严谨学术实证方案：
                研究问题：{q}
                输出：
                1.被解释变量
                2.核心解释变量
                3.控制变量
                4.模型形式（OLS/固定效应/DID等）
                5.标准误处理（聚类/稳健）
                6.需要的数据预处理（缺失值、缩尾、虚拟变量、描述性统计）
                """
                res = c.chat.completions.create(model=MODEL, messages=[{"role":"user","content":prompt}])
                st.session_state.research_plan = res.choices[0].message.content
                st.markdown(res.choices[0].message.content)

    if st.session_state.research_plan:
        with st.expander("✅ 已保存研究方案"):
            st.markdown(st.session_state.research_plan)

# ==============================================================================
# 第2页：多文件上传 + AI生成合并方案
# ==============================================================================
elif st.session_state.page == 1:
    st.subheader("上传多个数据集 → AI 自动生成合并逻辑")

    uploaded = st.file_uploader("上传多个 CSV 文件", type=["csv"], accept_multiple_files=True)

    if uploaded:
        st.session_state.data_files = {}
        for f in uploaded:
            df = pd.read_csv(f)
            st.session_state.data_files[f.name] = df
            st.success(f"✅ 已加载：{f.name} | 变量：{list(df.columns)}")

        with st.expander("🔍 查看所有上传数据"):
            for name, df in st.session_state.data_files.items():
                st.markdown(f"**{name}**")
                st.dataframe(df.head(3), use_container_width=True)

        if st.button("📝 AI 分析多数据 → 生成合并方案"):
            if not st.session_state.research_plan:
                st.warning("先生成研究方案！")
            else:
                info = ""
                for name, df in st.session_state.data_files.items():
                    # 生成描述性统计
                    desc_stats = df.describe().to_string()

                    # 拼进给 AI 的信息
                    info += f"文件名：{name}\n"
                    info += f"变量：{list(df.columns)}\n"
                    info += f"前5行数据：\n{df.head().to_csv()}\n"
                    info += f"描述性统计：\n{desc_stats}\n\n"
                    info += f"缺失值统计：\n{df.isnull().sum().to_string()}\n"

                with st.spinner("AI 分析多表结构，生成合并方案..."):
                    c = OpenAI(api_key=api_key, base_url=base_url)
                    prompt = f"""
                    你是R语言数据合并专家，根据多个数据表结构+实证方案，生成最优合并方案。
                    多个数据如下：
                    {info}
                    实证方案：{st.session_state.research_plan}

                    输出必须分成两部分，清晰分开：

                    【一、多表合并方案】
                    1.匹配关键字
                    2.合并方式
                    3.合并顺序

                    【二、合并后数据预处理方案】
                    1.缺失值处理
                    2.异常值/缩尾处理
                    3.虚拟变量生成
                    4.描述性统计要求
                    """
                    res = c.chat.completions.create(model=MODEL, messages=[{"role":"user","content":prompt}])
                    st.session_state.merge_plan = res.choices[0].message.content
                    st.markdown("### 📌 AI 多表合并方案")
                    st.markdown(st.session_state.merge_plan)

# ==============================================================================
# 第3页：AI生成回归代码 + 一键运行R
# ==============================================================================
elif st.session_state.page == 2:
    st.subheader("AI生成R回归代码 → 一键运行全流程")

    if st.session_state.df is None or not st.session_state.r_preprocess_code or not st.session_state.research_plan:
        st.error("请完成前两步！")
        st.stop()

    # 生成回归代码
    if st.button("📝 AI 生成 R 回归代码"):
        with st.spinner("生成回归代码..."):
            c = OpenAI(api_key=api_key, base_url=base_url)
            prompt = f"""
            你是R计量专家，根据实证方案生成可直接运行的回归代码。
            数据：data.csv
            已处理数据：data_clean
            实证方案：{st.session_state.research_plan}

            要求：
            1. 使用fixest::feols或lm
            2. 固定效应/聚类标准误按要求实现
            3. 输出完整回归结果
            4. 代码可直接运行
            5. 对于使用的包，需要进行检查，不存在则下载，存在则直接使用
            只输出代码。
            """
            res = c.chat.completions.create(model=MODEL, messages=[{"role":"user","content":prompt}])
            reg_code = res.choices[0].message.content.replace("```r", "").replace("```", "").strip()
            st.session_state.r_reg_code = reg_code
            st.code(reg_code, language="r")

    # 运行 R
    if st.button("🚀 运行 R 全流程（预处理+回归）"):
        with tempfile.TemporaryDirectory() as tmpdir:

            for name, df in st.session_state.data_files.items():
                path = os.path.join(tmpdir, name)
                df.to_csv(path, index=False, encoding="utf-8")

            # 合并代码
            full_code = st.session_state.r_preprocess_code + "\n\n" + st.session_state.r_reg_code
            r_file = os.path.join(tmpdir, "run.R")
            with open(r_file, "w", encoding="utf-8") as f:
                f.write(full_code)

            # 运行
            result = subprocess.run(
                ["R", "--vanilla", "-f", r_file],
                cwd=tmpdir,
                capture_output=True,
                text=True
            )

            st.subheader("📊 R 运行结果")
            st.code(result.stdout)

            if result.stderr:
                with st.expander("⚠️ 警告/错误"):
                    st.error(result.stderr)

# ==============================================================================
# 第4页：汇总
# ==============================================================================
elif st.session_state.page == 3:
    st.subheader("📈 全流程实证完成")
    st.success("AI研究方案 → AI数据处理(R) → AI回归(R) → 结果输出")
    with st.expander("研究方案"):
        st.markdown(st.session_state.research_plan)
    with st.expander("R 预处理代码"):
        st.code(st.session_state.r_preprocess_code)
    with st.expander("R 回归代码"):
        st.code(st.session_state.r_reg_code)

# ==============================================================================
# 第5页：DAG 因果图分析（交互式）
# ==============================================================================
elif st.session_state.page == 4:
    from streamlit_cytoscape import cytoscape

    st.subheader("📐 构建并分析因果图 (DAG)")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("#### 1️⃣ 定义因果边 (格式：`X -> Y, Z -> X, Z -> Y`)")
        edges_input = st.text_area(
            "输入因果关系边，每行一条或用逗号分隔",
            value="X -> Y\nZ -> X\nZ -> Y",
            height=120,
            help="例如：\nEducation -> Income\nFather_Edu -> Education\nFather_Edu -> Income"
        )

        st.markdown("#### 2️⃣ 节点属性 (用于悬停显示取值范围)")
        node_attr_input = st.text_area(
            "输入每个变量的取值范围或描述 (格式：`变量名: 描述`)",
            value="X: 0/1 二值变量\nY: 连续变量 (0-100)\nZ: 分类变量 (1,2,3)",
            height=100,
            help="例如：\nAge: 20-60岁\nTreatment: 处理组=1, 对照组=0"
        )
        st.session_state.node_attr_input = node_attr_input

        if st.button("🧬 生成 DAG 图", type="primary"):
            # 解析边输入
            raw_edges = edges_input.replace(",", "\n").split("\n")
            edges = []
            nodes_set = set()
            for line in raw_edges:
                line = line.strip()
                if "->" in line:
                    parts = line.split("->")
                    if len(parts) == 2:
                        src = parts[0].strip()
                        tgt = parts[1].strip()
                        if src and tgt:
                            edges.append((src, tgt))
                            nodes_set.add(src)
                            nodes_set.add(tgt)

            # 解析节点属性描述
            attr_dict = {}
            for line in node_attr_input.split("\n"):
                if ":" in line:
                    name, desc = line.split(":", 1)
                    attr_dict[name.strip()] = desc.strip()

            # 构造 cytoscape 元素
            elements = []
            for node in nodes_set:
                node_data = {
                    "data": {"id": node, "label": node},
                    "classes": "node"
                }
                # 将属性描述存入 data 中，用于 tooltip
                if node in attr_dict:
                    node_data["data"]["description"] = attr_dict[node]
                elements.append(node_data)

            for src, tgt in edges:
                elements.append({
                    "data": {"source": src, "target": tgt},
                    "classes": "edge"
                })

            st.session_state.dag_elements = elements

            # 调用 AI 检验混杂和后门路径（如果有研究方案）
            if api_key and st.session_state.research_plan:
                with st.spinner("AI 分析混杂与后门路径中..."):
                    c = OpenAI(api_key=api_key, base_url=base_url)
                    dag_desc = "\n".join([f"{src} -> {tgt}" for src, tgt in edges])
                    prompt = f"""
                    你是一个因果推断专家。给定以下 DAG 边列表和研究方案，请分析：
                    1. 是否存在混杂变量？如果是，列出具体变量名。
                    2. 是否存在后门路径？如果是，用变量名+箭头完整写出后门路径（如 Z <- X -> Y）。

                    DAG 边：
                    {dag_desc}

                    研究方案（可作为背景参考）：
                    {st.session_state.research_plan}

                    请按以下格式严格输出（只输出这两部分，不要多余解释）：
                    【混杂变量】：是/否
                    变量列表：（如果“是”，列出变量名，逗号分隔）
                    【后门路径】：是/否
                    路径：（如果“是”，写出具体路径，用 <- 或 -> 表示方向）
                    """
                    res = c.chat.completions.create(model=MODEL, messages=[{"role":"user","content":prompt}])
                    output = res.choices[0].message.content

                    # 解析结果
                    lines = output.split("\n")
                    conf_exists = "否"
                    conf_vars = ""
                    back_exists = "否"
                    back_path = ""
                    for line in lines:
                        if "【混杂变量】" in line:
                            conf_exists = "是" if "是" in line else "否"
                        if "变量列表" in line and conf_exists == "是":
                            conf_vars = line.split("：")[-1].strip()
                        if "【后门路径】" in line:
                            back_exists = "是" if "是" in line else "否"
                        if "路径" in line and back_exists == "是":
                            back_path = line.split("：")[-1].strip()

                    st.session_state.confounding_check = f"混杂变量：{conf_exists}"
                    if conf_exists == "是" and conf_vars:
                        st.session_state.confounding_check += f" → {conf_vars}"
                    st.session_state.backdoor_check = f"后门路径：{back_exists}"
                    if back_exists == "是" and back_path:
                        st.session_state.backdoor_check += f" → {back_path}"
            else:
                st.session_state.confounding_check = "（需要研究方案和 API 才能自动分析）"
                st.session_state.backdoor_check = "（需要研究方案和 API 才能自动分析）"

    with col_right:
        st.markdown("#### 🖱️ 交互提示")
        st.info(
            """
            - **悬停** 节点显示变量描述
            - **点击** 节点可放大并加金色环（可多选）
            - 选中的节点会高亮其**连入/连出**的边
            """
        )

    if st.session_state.dag_elements:
        st.markdown("---")
        st.markdown("#### 因果有向无环图 (DAG)")
        # 定义 cytoscape 样式，实现点击放大、高亮边
        stylesheet = [
            {
                "selector": "node",
                "style": {
                    "background-color": "#4A90E2",
                    "label": "data(label)",
                    "font-size": "14px",
                    "color": "#000000",
                    "text-valign": "center",
                    "text-halign": "center",
                    "width": "60px",
                    "height": "60px",
                    "border-width": 2,
                    "border-color": "#FFFFFF",
                    "tooltip": "data(description)" if "description" in "data" else "data(label)"
                }
            },
            {
                "selector": "edge",
                "style": {
                    "width": 3,
                    "line-color": "#888888",
                    "target-arrow-color": "#888888",
                    "target-arrow-shape": "triangle",
                    "curve-style": "bezier"
                }
            },
            # 选中节点样式（变大、加金色环）
            {
                "selector": "node:selected",
                "style": {
                    "width": "80px",
                    "height": "80px",
                    "border-width": 5,
                    "border-color": "#FFD700",
                    "background-color": "#1E5CB3",
                    "z-index": 999
                }
            },
            # 高亮与选中节点相关的边
            {
                "selector": "edge:selected",
                "style": {
                    "width": 5,
                    "line-color": "#FF8C00",
                    "target-arrow-color": "#FF8C00"
                }
            }
        ]

        # 使用 cytoscape 组件
        selected = cytoscape(
            elements=st.session_state.dag_elements,
            stylesheet=stylesheet,
            layout={"name": "dagre", "rankDir": "TB"},   # 从上到下布局
            height="500px",
            key="dag_canvas",
            selection_type="additive",   # 允许多选
            user_zooming_enabled=True,
            user_panning_enabled=True,
        )

        # 下方显示混杂与后门检验结果
        st.markdown("---")
        st.markdown("### 🔍 因果检验结果")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**混杂变量检测**  \n{st.session_state.confounding_check}")
        with col2:
            st.markdown(f"**后门路径检测**  \n{st.session_state.backdoor_check}")

        # 可选：显示当前选中的节点 ID
        if selected and "nodes" in selected:
            selected_nodes = [node["id"] for node in selected["nodes"]]
            st.caption(f"当前选中节点：{', '.join(selected_nodes)}")