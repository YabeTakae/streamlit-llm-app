import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# ローカル用：.env を読み込む（Cloudでは .env は無いが害はない）
load_dotenv()


def get_api_key() -> str:
    """ローカル(.env) → Cloud(Secrets) の順にAPIキーを取得"""
    key = os.getenv("OPENAI_API_KEY")

    if not key:
        try:
            key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            key = None

    return key


OPENAI_API_KEY = get_api_key()

st.set_page_config(page_title="Streamlit LLM App", page_icon="🤖")

st.title("🤖 Streamlit × LangChain LLMアプリ")
st.write(
    """
### このアプリでできること
- 入力したテキストをLLMに渡して回答を表示します
- ラジオボタンで「専門家タイプ」を選ぶと、LLMの役割（システムメッセージ）が切り替わります

### 使い方
1. 専門家タイプを選ぶ
2. 質問を入力する
3. 「送信」を押す
"""
)

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY が未設定です。ローカルは .env、Cloudは Secrets に設定してください。")
    st.stop()

# 専門家タイプ（A/B）
expert_type = st.radio(
    "専門家タイプを選択",
    options=["A: キャリアコーチ", "B: 旅行プランナー"],
    horizontal=True,
)

# 入力フォーム（1つ）
user_text = st.text_input(
    "入力テキスト",
    placeholder="例：転職の自己PRを添削して / 2泊3日の旅行プラン作って",
)


# 必須：関数（入力テキスト＋選択値 → LLM回答）
def get_llm_answer(input_text: str, selected_expert: str) -> str:
    if selected_expert.startswith("A"):
        system_message = (
            "あなたは経験豊富なキャリアコーチです。"
            "ユーザーの状況を整理し、現実的で実行可能なアドバイスを日本語で提供してください。"
            "箇条書きを多めに、必要なら追加質問を1つだけ添えてください。"
        )
    else:
        system_message = (
            "あなたはプロの旅行プランナーです。"
            "ユーザーの希望に沿った旅行プラン（行程・移動・予算感・注意点）を日本語で提案してください。"
            "見出し＋箇条書き中心で、必要なら追加質問を1つだけ添えてください。"
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("user", "{question}"),
        ]
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=OPENAI_API_KEY,
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(question=input_text)


# ボタンで実行
if st.button("送信", type="primary"):
    if not user_text.strip():
        st.warning("入力テキストを入力してください。")
    else:
        with st.spinner("回答生成中..."):
            answer = get_llm_answer(user_text, expert_type)
        st.subheader("回答")
        st.write(answer)

st.caption("※注意：.env（APIキー）はGitHubにアップロードしないでください。")
