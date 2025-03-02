import os
import uuid
import io
import random
from typing import Annotated, Literal
# from pydantic import BaseModel, Field
##########エージェントによるデータ取得検証###########
from pydantic.v1 import BaseModel, Field
###################################################
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode


load_dotenv()


llm_4o = ChatOpenAI(model="gpt-4o")
llm_o1 = ChatOpenAI(model="o1")


problem_examples = [
    "300円で最も満足できるおやつの組み合わせは？",
    # "ナップサックに詰め込む品物の総価値を最大にするには？",
    # "5人に8つの仕事を割り当てます。熟練度やコストを考慮した割り当て方は？"
]


# セッション初期化
session_init_dict = {
    "session_id": str(uuid.uuid1()),
    "problem_example": random.choice(problem_examples),
    "executed": False,
    "optim_model": None,
}
for session_name, init_value in session_init_dict.items():
    if session_name not in st.session_state:
        st.session_state[session_name] = init_value

# 保存先フォルダ作成
workdir = os.path.join("outputs", st.session_state["session_id"])
filepath_usermsg = os.path.join(workdir, "user_msg.md")
filepath_text = os.path.join(workdir, "result.md")
filepath_constants = os.path.join(workdir, "constants.xlsx")


def indent(depth: int = 0) -> str:
    return f"{' ' * (4 * depth)}"


class Constant(BaseModel):
    """定数"""
    name: str = Field(..., description="定数名。定数の内容がわかる日本語（例1: 品物の集合。例2: 品物iの価値。）")
    expression: str = Field(..., description="定数の定義式。LaTeX形式（例1: I=\{A,B,\\ldots,E\}。例2: v_i \\quad (i \\in I)")
    value: str = Field(..., description="定数の値。CSV形式。ラベルがある場合は列名や行名にする（例: A,B,C,D,E,\n60,100,120,90,30）")
    # value: str = Field(..., description="定数の値。Markdown形式（例: |A|B|C|D|E|\n|-:-|-:-|-:-|-:-|-:-|\n|60|100|120|90|30|）")

    def getText(self, depth: int = 0) -> str:
        text = f"{indent(depth)}1. {self.name}\n$$\n{self.expression}\n$$"
        return text
    
    @staticmethod
    def list2Text(c_list: list["Constant"], depth: int = 0) -> str:
        text = "\n".join([c.getText(depth) for c in c_list])
        return text
    
    @staticmethod
    def toExcel(filepath: str, c_list: list["Constant"]) -> None:
        """シート名=定数名としてExcelに保存"""
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            for constant in c_list:
                # データがCSV
                df = pd.read_csv(io.StringIO(constant.value))
                # データがMarkdown
                # markdown_text = constant.value
                # markdown_text = markdown_text.replace('|', ',').replace('\n', '\n').strip()
                # markdown_text = "\n".join([line.strip() for line in markdown_text.split("\n")])
                # df = pd.read_csv(io.StringIO(markdown_text))
                # 保存
                df.to_excel(writer, sheet_name=constant.name, index=False)


class Variable(BaseModel):
    """決定変数"""
    description: str = Field(..., description="決定変数の内容")
    expression: str = Field(..., description="決定変数の定義式。定義域も含める。LaTeX形式（例: x_i \\in \\{0, 1\\} \\quad (i=1,2,\\ldots,N)）")

    def getText(self, depth: int = 0) -> str:
        text = f"{indent(depth)}1. {self.description}\n$$\n{self.expression}\n$$"
        return text
    
    @staticmethod
    def list2Text(v_list: list["Variable"], depth: int = 0) -> str:
        text = "\n".join([v.getText(depth) for v in v_list])
        return text


class Constraint(BaseModel):
    """制約条件"""
    description: str = Field(..., description="制約条件の内容")
    additional_variables: list[Variable] = Field(
        default_factory=list,
        description="制約条件の定式化に追加で必要な決定変数"
    )
    formulae: list[str] = Field(
        default_factory=list,
        description="制約条件の式。LaTeX形式。複数ある場合はインデックスに入る値を明記する"
    )

    def getText(self, depth: int = 0, desc_only: bool = False) -> str:
        text = f"{indent(depth)}1. {self.description}"
        if not desc_only:
            if len(self.additional_variables) != 0:
                _text_additional = (
                    f"{indent(depth + 1)}1. 追加変数\n"
                    f"{Variable.list2Text(self.additional_variables, depth + 2)}"
                )
                text += f"\n{_text_additional}"
            if len(self.formulae) != 0:
                _text_formulae = (
                    f"{indent(depth + 1)}1. 定式化\n"
                    f"{'\n'.join([f'$$\n{formula}\n$$' for formula in self.formulae])}"
                )
                text += f"\n{_text_formulae}"
        return text
    
    @staticmethod
    def list2Text(c_list: list["Constraint"], depth: int = 0, desc_only: bool = False) -> str:
        text = "\n".join([c.getText(depth, desc_only) for c in c_list])
        return text


class Objective(BaseModel):
    """目的関数"""
    description: str = Field(..., description="目的関数の内容")
    additional_variables: list[Variable] = Field(
        default_factory=list,
        description="目的関数の定式化に追加で必要な決定変数"
    )
    formulae: list[str] = Field(
        default_factory=list,
        description="目的関数の式。目的関数の定式化に必要な追加の制約条件も含めて良い。LaTeX形式。複数ある場合はインデックスに入る値を明記する"
    )

    def getText(self, depth: int = 0, desc_only: bool = False) -> str:
        text = f"{indent(depth)}1. {self.description}"
        if not desc_only:
            if len(self.additional_variables) != 0:
                _text_additional = (
                    f"{indent(depth + 1)}1. 追加変数\n"
                    f"{Variable.list2Text(self.additional_variables, depth + 2)}"
                )
                text += f"\n{_text_additional}"
            if len(self.formulae) != 0:
                _text_formulae = (
                    f"{indent(depth + 1)}1. 定式化\n"
                    f"{'\n'.join([f'$$\n{formula}\n$$' for formula in self.formulae])}"
                )
                text += f"\n{_text_formulae}"
        return text
    
    @staticmethod
    def list2Text(o_list: list["Objective"], depth: int = 0, desc_only: bool = False) -> str:
        text = "\n".join([o.getText(depth, desc_only) for o in o_list])
        return text


class OptimModel(BaseModel):
    """最適化モデル"""
    description: str = Field(..., description="最適化問題の概要")
    given_constants: list[Constant] = Field(..., description="ユーザから与えられた定数")
    additional_constants: list[Constant] = Field(..., description="最適化モデル化に追加で必要な定数")
    variables: list[Variable] = Field(..., description="最適化モデルの決定変数")
    constraints: list[Constraint] = Field(..., description="最適化モデルの制約条件")
    objectives: list[Objective] = Field(..., description="最適化モデルの目的関数")

    def getConstantsText(self, depth: int = 0) -> str:
        text = Constant.list2Text(self.given_constants + self.additional_constants, depth)
        return text
    
    def getVariablesText(self, depth: int = 0) -> str:
        text = Variable.list2Text(self.variables, depth)
        return text
    
    def getConstraintsText(self, depth: int = 0, desc_only: bool = False) -> str:
        text = Constraint.list2Text(self.constraints, depth, desc_only)
        return text
    
    def getObjectivesText(self, depth: int = 0, desc_only: bool = False) -> str:
        text = Objective.list2Text(self.objectives, depth, desc_only)
        return text

##########エージェントによるデータ取得検証###########

@tool
def search_data(query: Annotated[str, "検索文字列"]) -> str:
    """
    マスタデータからデータを検索し、取得したデータに関する情報を文字列として返す。
    取得に失敗したら空の文字列を返す。
    引数として許されるのは、"おやつの値段", "ナップザックの容量", "仕事の名称"
    """
    if query == "おやつの値段":
        data_info = """
定数名: おやつの値段
定義式: v_i \\quad (i \\in I)
値: ,値段\nチョコ,60\nポテチ,100\nポッキー,120\nせんべい,90\nガム,30
"""
    elif query == "ナップザックの容量":
        data_info = """
定数名: ナップザックの容量
定義式: c_i \\quad (i \\in I)
値: ,容量\nA,30\nB,50\nC,40
"""
    elif query == "仕事の名称":
        data_info = """
定数名: 仕事の名称
定義式: i \\in I
値: 仕事の名称\n清掃\n加工\n研磨\n検査
"""
    else:
        data_info = ""
    return data_info


class AgentState(MessagesState):
    final_response: OptimModel


# LLM定義
tools = [search_data, OptimModel]
model_with_response_tool = llm_4o.bind_tools(tools, tool_choice="any")


# ノード定義
def call_model(state: AgentState):
    response = model_with_response_tool.invoke(state["messages"])
    return {"messages": [response]}


def respond(state: AgentState):
    tool_call = state["messages"][-1].tool_calls[0]
    response = OptimModel(**tool_call["args"])
    tool_message = {
        "type": "tool",
        "content": "Here is your structured response",
        "tool_call_id": tool_call["id"],
    }
    return {"final_response": response, "messages": [tool_message]}


# 条件付きエッジ
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if (
        len(last_message.tool_calls) == 1
        and last_message.tool_calls[0]["name"] == "OptimModel"
    ):
        return "respond"
    else:
        return "continue"


# グラフ構築
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("respond", respond)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "respond": "respond",
    },
)
workflow.add_edge("tools", "agent")
workflow.add_edge("respond", END)
graph = workflow.compile()

###################################################

st.title("マスタ作成支援")

if st.button("サンプル問題をセット"):
    st.session_state["request_value"] = st.session_state["problem_example"]

# ユーザの入力
user_request = st.text_area(
    label="最適化問題の内容",
    key="request_value",
    placeholder=st.session_state["problem_example"],
)

# Excelファイルのアップロード
uploaded_files = st.file_uploader("Excelファイルをアップロード", type=["xlsx", "xls"], accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            # Excelファイルを開いてシート名を取得（データをロードしない）
            excel_file = pd.ExcelFile(uploaded_file)
        except Exception as e:
            st.error(f"❌ 読み込みエラー: {e}")


if st.button("実行"):
    assert len(user_request) != 0, "最適化問題の内容が入力されていません。"

    # 出力先フォルダがなければ作成
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    # Excelファイルの各シートをテキスト（markdown）に変換
    user_data = {}
    for uploaded_file in uploaded_files:
        excel_file = pd.ExcelFile(uploaded_file)
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
            user_data[sheet_name] = df.to_markdown()

    # ユーザ入力とExcelファイルの統合
    user_msg = f"# 最適化問題の内容\n{user_request}"
    if len(user_data) != 0:
        user_msg += f"\n\n# データ"
        for sheet_name, table_txt in user_data.items():
            user_msg += f"\n\n## {sheet_name}\n{table_txt}"
    
    # ユーザメッセージ保存
    with open(filepath_usermsg, "w") as f:
        f.write(user_msg)

    # LLM実行
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "あなたは混合整数線形計画問題のスペシャリストです。"
                "ユーザから問題概要（と場合によりデータ）が与えられます。"
                "問題を解くために必要な定数、決定変数の定義、制約条件、目的関数の定義を行ってください。"
                "なお、ユーザから与えられるデータは十分でない場合があるため、必要なら追加の定数を定義してください。"
            ),
            (
                "human",
                "{user_msg}"
            ),
        ]
    )
    # chain = prompt | llm_4o.with_structured_output(OptimModel)
    # optim_model: OptimModel = chain.invoke(user_msg)

    ##########エージェントによるデータ取得検証###########
    response = graph.invoke(input={"messages": [("human", user_msg)]}, debug=True)
    optim_model: OptimModel = response["final_response"]
    ###################################################

    st.session_state["executed"] = True
    st.session_state["optim_model"] = optim_model
    st.rerun()


if st.session_state["executed"]:
    optim_model: OptimModel = st.session_state["optim_model"]

    st.divider()

    # 結果まとめ
    result_overview = (
        "## 最適化問題の内容\n"
        f"{optim_model.description}\n\n"
        "## 目的関数\n"
        f"{optim_model.getObjectivesText(depth=0, desc_only=True)}\n\n"
        "## 制約条件\n"
        f"{optim_model.getConstraintsText(depth=0, desc_only=True)}"
    )
    result_formulation = (
        "## 定数\n"
        f"{optim_model.getConstantsText(depth=0)}\n\n"
        "## 決定変数\n"
        f"{optim_model.getVariablesText(depth=0)}\n\n"
        "## 目的関数\n"
        f"{optim_model.getObjectivesText(depth=0, desc_only=False)}\n\n"
        "## 制約条件\n"
        f"{optim_model.getConstraintsText(depth=0, desc_only=False)}"
    )
    result_text = f"{result_overview}\n\n{result_formulation}"

    # 出力テキストの保存
    with open(filepath_text, "w") as f:
        f.write(result_text)
    # 定数情報をExcelに保存
    Constant.toExcel(
        filepath=filepath_constants,
        c_list=optim_model.given_constants + optim_model.additional_constants,
    )

    # 出力テキストのダウンロードボタン
    st.download_button(
        label="出力テキストのダウンロード",
        data=open(filepath_text, "rb").read(),
        file_name="出力テキスト.md",
        mime="text/plain",
    )
    # 定数情報のダウンロードボタン
    st.download_button(
        label="マスタデータのダウンロード",
        data=open(filepath_constants, "rb").read(),
        file_name="マスタデータ.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # 結果の表示
    container = st.container(border=True)
    container.write("問題概要")
    container.write(result_overview)
    container = st.container(border=True)
    container.write("定式化")
    container.write(result_formulation)

    with st.expander("LLM出力"):
        st.write(optim_model)
