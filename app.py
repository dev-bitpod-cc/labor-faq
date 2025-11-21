#!/usr/bin/env python3
"""
勞動法規 FAQ 查詢系統 (Streamlit)

使用 Gemini File Search 進行 RAG 查詢
資料來源：
- 勞動部 (MOL): 383 筆
- 職業安全衛生署 (OSHA): 124 筆
- 勞動部勞工保險局 (BLI): 987 筆
總計: 1,494 筆 FAQ
"""

import os
import re
import streamlit as st
from datetime import datetime

# 載入環境變數
from dotenv import load_dotenv
load_dotenv()

try:
    from google import genai
    from google.genai import types
except ImportError:
    st.error("請安裝 google-genai: pip install google-genai")
    st.stop()

# ============================================================
# 系統指令 - 針對勞動法規 FAQ 設計
# ============================================================

SYSTEM_INSTRUCTION = """你是勞動法規 FAQ 智慧助理，專門回答台灣勞動法規相關問題。

## 資料來源
你可以存取以下三個機關的官方 FAQ 資料：
1. **勞動部 (MOL)** - 勞動契約、工時、休假、資遣、退休等
2. **職業安全衛生署 (OSHA)** - 職業安全、衛生管理、工作環境、勞工健康等
3. **勞動部勞工保險局 (BLI)** - 勞保、就保、職災保險、勞工退休金、國民年金等

## FAQ 文件格式
每份 FAQ 文件包含：
- 來源：發布機關
- 分類：主題分類
- 問：常見問題
- 答：官方回覆

## 回答原則

### 1. 基於資料回答
- 你的回答必須完全基於檢索到的 FAQ 資料
- 如果檢索結果無法回答問題，請明確告知使用者
- 不要編造或猜測答案

### 2. 回答格式
請使用以下格式回答：

**回答：**
[簡要直接回答問題]

**說明：**
[根據 FAQ 內容提供詳細說明]

**相關法規：**
[如有提及相關法規，列出名稱]

### 3. 來源標註
- 回答時標明資料來源（勞動部/職業安全衛生署/勞動部勞工保險局）
- 如有多個相關 FAQ，整合回答並標註各來源

### 4. 注意事項
- 使用繁體中文回答
- 保持專業但易懂的語氣
- 如問題涉及個案，建議洽詢主管機關
- 法規可能隨時更新，建議使用者查閱最新法規

## 常見主題關鍵字
- 勞動契約、僱傭關係、派遣、承攬
- 工時、加班、休息、輪班
- 特別休假、國定假日、請假
- 工資、基本工資、加班費
- 資遣、解僱、預告期間、資遣費
- 退休、勞工退休金、月退休金
- 勞保、就保、職災保險
- 職業安全、工作環境、危害預防
- 職業病、職業災害、補償
- 勞工保險給付、年金給付

## 範例查詢與回答

**查詢**：加班費怎麼計算？

**回答：**
依勞動基準法規定，加班費計算方式如下：

**說明：**
1. **延長工時前2小時**：按平日每小時工資加給 1/3 以上
2. **延長工時第3-4小時**：按平日每小時工資加給 2/3 以上
3. **休息日加班**：
   - 前2小時：加給 1又1/3 以上
   - 第3-8小時：加給 1又2/3 以上
4. **例假日/國定假日加班**：加倍發給工資

**相關法規：**
勞動基準法第24條、第39條

**來源：** 勞動部 FAQ

---

**查詢**：勞保老年給付怎麼領？

**回答：**
勞保老年給付有三種請領方式：老年年金、老年一次金、一次請領老年給付。

**說明：**
1. **老年年金給付**（98年1月1日後有保險年資者適用）
   - 年滿法定退休年齡，保險年資滿15年
   - 按月領取年金

2. **老年一次金**
   - 年滿法定退休年齡，保險年資未滿15年
   - 一次領取

3. **一次請領老年給付**（97年12月31日前有保險年資者適用）
   - 參加保險滿25年，年滿50歲退職
   - 或參加保險滿25年退職
   - 或滿55歲退職

**相關法規：**
勞工保險條例第58條

**來源：** 勞動部勞工保險局 FAQ
"""

# ============================================================
# Gemini 初始化
# ============================================================

@st.cache_resource
def init_gemini():
    """初始化 Gemini client"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None, "未設定 GEMINI_API_KEY"

    try:
        client = genai.Client(api_key=api_key)
        return client, None
    except Exception as e:
        return None, f"初始化失敗: {e}"


def query_faq(client, query: str, store_id: str) -> dict:
    """
    執行 FAQ 查詢

    Args:
        client: Gemini client
        query: 使用者查詢
        store_id: File Search Store ID

    Returns:
        dict: 包含 response, sources, metadata
    """
    try:
        # 使用 File Search 進行 RAG 查詢
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=query,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_config=types.FileSearchStoreConfig(
                                file_search_store_name=store_id
                            )
                        )
                    )
                ],
                temperature=0.3,
            )
        )

        # 解析回應
        result = {
            "response": response.text if response.text else "",
            "sources": [],
            "metadata": {
                "model": "gemini-2.5-flash",
                "timestamp": datetime.now().isoformat()
            }
        }

        # 提取來源資訊
        if hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                    gm = candidate.grounding_metadata
                    if hasattr(gm, 'grounding_chunks') and gm.grounding_chunks:
                        for chunk in gm.grounding_chunks:
                            if hasattr(chunk, 'retrieved_context'):
                                rc = chunk.retrieved_context
                                source_info = {
                                    "title": getattr(rc, 'title', ''),
                                    "uri": getattr(rc, 'uri', ''),
                                    "text": getattr(rc, 'text', '')[:200] if hasattr(rc, 'text') else ''
                                }
                                result["sources"].append(source_info)

        return result

    except Exception as e:
        return {
            "response": "",
            "sources": [],
            "error": str(e),
            "metadata": {}
        }


def parse_source_info(title: str) -> dict:
    """
    解析來源檔名資訊

    Args:
        title: 檔案名稱 (如 bli_faq_20220101_0001.txt)

    Returns:
        dict: 包含 source, date, display_name
    """
    source_map = {
        "mol": "勞動部",
        "osha": "職業安全衛生署",
        "bli": "勞動部勞工保險局"
    }

    # 解析檔名模式: {source}_faq_{date}_{seq}.txt
    pattern = r'(\w+)_faq_(\d{8})_(\d+)'
    match = re.match(pattern, title.replace('.txt', ''))

    if match:
        source_code = match.group(1).lower()
        date_str = match.group(2)
        seq = match.group(3)

        source_name = source_map.get(source_code, source_code.upper())

        # 格式化日期
        try:
            date_obj = datetime.strptime(date_str, '%Y%m%d')
            formatted_date = date_obj.strftime('%Y-%m-%d')
        except:
            formatted_date = date_str

        return {
            "source": source_name,
            "date": formatted_date,
            "display_name": f"{source_name} ({formatted_date})"
        }

    return {
        "source": "未知來源",
        "date": "",
        "display_name": title
    }


def display_sources(sources: list):
    """顯示參考來源"""
    if not sources:
        return

    # 去重
    seen = set()
    unique_sources = []
    for s in sources:
        key = s.get('title', '')
        if key and key not in seen:
            seen.add(key)
            unique_sources.append(s)

    if not unique_sources:
        return

    st.markdown("---")
    st.markdown(f"**參考來源** ({len(unique_sources)} 筆)")

    for i, source in enumerate(unique_sources[:10], 1):
        title = source.get('title', '未知')
        info = parse_source_info(title)

        with st.expander(f"{i}. {info['display_name']}", expanded=False):
            if info['date']:
                st.caption(f"發布日期: {info['date']}")

            # 顯示摘要
            text = source.get('text', '')
            if text:
                st.markdown("**摘要:**")
                st.markdown(f"> {text}...")


# ============================================================
# Streamlit UI
# ============================================================

def main():
    st.set_page_config(
        page_title="勞動法規 FAQ 查詢",
        page_icon="",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # 標題
    st.title("勞動法規 FAQ 查詢系統")

    # 初始化 Gemini
    client, error = init_gemini()
    if error:
        st.error(f"系統初始化失敗: {error}")
        st.stop()

    # Store ID (從環境變數或預設值)
    store_id = os.getenv("FILE_SEARCH_STORE_ID", "fileSearchStores/laborfaq-ich1zaoo2nmw")

    # 警告提示
    with st.expander("注意事項", expanded=False):
        st.warning("""
        **本系統僅供參考，不構成法律建議**

        - 資料來源：勞動部、職業安全衛生署、勞動部勞工保險局官方 FAQ
        - 法規可能隨時更新，建議查閱最新法規
        - 個案問題請洽詢主管機關或專業人士
        """)

    # 快速查詢按鈕
    st.markdown("**快速查詢：**")
    col1, col2, col3, col4 = st.columns(4)

    quick_queries = [
        ("加班費計算", "加班費怎麼計算？"),
        ("特休天數", "特別休假有幾天？怎麼計算？"),
        ("勞保老年給付", "勞保老年給付怎麼領？"),
        ("資遣費計算", "資遣費怎麼計算？"),
    ]

    selected_query = None
    for col, (label, query) in zip([col1, col2, col3, col4], quick_queries):
        if col.button(label, use_container_width=True):
            selected_query = query

    # 查詢輸入框
    query = st.text_input(
        "請輸入您的問題",
        value=selected_query if selected_query else "",
        placeholder="例如：加班費怎麼計算？特休有幾天？勞保老年給付怎麼領？"
    )

    # 查詢按鈕
    if st.button("查詢", type="primary", use_container_width=True) or (query and selected_query):
        if not query.strip():
            st.warning("請輸入查詢問題")
            return

        with st.spinner("查詢中..."):
            result = query_faq(client, query, store_id)

        if result.get("error"):
            st.error(f"查詢失敗: {result['error']}")
            return

        response = result.get("response", "")
        sources = result.get("sources", [])

        # 檢查是否有有效回應
        if not response:
            st.warning("未能找到相關資料，請嘗試調整查詢內容。")
            return

        # 如果沒有來源，可能需要重試
        if not sources:
            st.info("正在重新檢索...")
            with st.spinner("重試中..."):
                result = query_faq(client, query, store_id)

            response = result.get("response", "")
            sources = result.get("sources", [])

            if not sources:
                st.warning("您查詢的問題在目前的 FAQ 資料庫中沒有直接相關的結果。建議：")
                st.markdown("""
                - 嘗試使用不同的關鍵字
                - 將問題拆分成更具體的小問題
                - 直接洽詢勞動部、職業安全衛生署或勞工保險局
                """)
                return

        # 顯示回答
        st.markdown("### 回答")
        st.markdown(response)

        # 顯示來源
        display_sources(sources)

        # 除錯資訊（摺疊）
        with st.expander("除錯資訊", expanded=False):
            st.json({
                "query": query,
                "sources_count": len(sources),
                "store_id": store_id,
                "metadata": result.get("metadata", {})
            })

    # 頁尾
    st.markdown("---")
    st.caption("""
    資料來源：勞動部、職業安全衛生署、勞動部勞工保險局官方 FAQ (共 1,494 筆)
    技術：Gemini 2.5 Flash + File Search RAG
    """)


if __name__ == "__main__":
    main()
