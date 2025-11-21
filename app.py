#!/usr/bin/env python3
"""
勞動法規 FAQ 查詢系統 (Streamlit)

使用 Gemini File Search 進行 RAG 查詢
資料來源：
- 勞動部 (MOL): 勞動契約、工時、休假、資遣等
- 職業安全衛生署 (OSHA): 職業安全、衛生管理等
- 勞動部勞工保險局 (BLI): 勞保、就保、職災、退休金等
總計: 1,110 筆 FAQ

Version: 1.0.0 (2025-11-21)
"""

import os
import re
import json
import streamlit as st
from datetime import datetime
from pathlib import Path

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


@st.cache_data
def load_file_mapping():
    """載入 FAQ 檔案映射（包含原始連結）"""
    mapping_path = Path(__file__).parent / "data" / "faq_file_mapping.json"
    if mapping_path.exists():
        with open(mapping_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('files', {})
    return {}


@st.cache_data
def load_gemini_id_mapping():
    """載入 Gemini ID 映射（gemini_file_id → document_id）"""
    mapping_path = Path(__file__).parent / "data" / "faq_gemini_id_mapping.json"
    if mapping_path.exists():
        with open(mapping_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            files = data.get('files', {})
            # 建立反向映射: gemini_file_id → document_id
            reverse_map = {}
            for doc_id, info in files.items():
                gemini_id = info.get('gemini_file_id', '')
                if gemini_id:
                    reverse_map[gemini_id] = doc_id
            return reverse_map
    return {}


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
                            file_search_store_names=[store_id]
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


def parse_source_info(title: str, text: str = "", file_mapping: dict = None, gemini_id_mapping: dict = None) -> dict:
    """
    解析來源資訊

    Args:
        title: 檔案名稱 (可能是 Gemini file ID 或檔名)
        text: 內容片段（可從中提取來源和問題）
        file_mapping: 檔案映射（包含原始連結）
        gemini_id_mapping: Gemini ID 反向映射（gemini_file_id → document_id）

    Returns:
        dict: 包含 source, question, display_name, detail_url
    """
    source_map = {
        "mol": "勞動部",
        "osha": "職業安全衛生署",
        "bli": "勞動部勞工保險局"
    }

    source_name = ""
    question = ""
    category = ""
    detail_url = ""
    doc_id = ""

    # 嘗試從檔名提取 document ID
    pattern = r'(\w+_faq_\d{8}_\d+)'
    match = re.search(pattern, title.replace('.txt', ''))
    if match:
        doc_id = match.group(1)

    # 如果 title 是 Gemini file ID，從反向映射查詢 document_id
    if not doc_id and gemini_id_mapping and title in gemini_id_mapping:
        doc_id = gemini_id_mapping[title]

    # 從 file_mapping 查詢原始連結
    if file_mapping and doc_id and doc_id in file_mapping:
        mapping_info = file_mapping[doc_id]
        detail_url = mapping_info.get('detail_url', '')
        if not question:
            question = mapping_info.get('question', '')
        if not source_name:
            source_name = mapping_info.get('source', '')

    # 優先從內容中提取來源和問題
    if text:
        # 提取來源
        source_match = re.search(r'來源:\s*(.+?)(?:\n|$)', text)
        if source_match:
            source_name = source_match.group(1).strip()

        # 提取分類
        category_match = re.search(r'分類:\s*(.+?)(?:\n|$)', text)
        if category_match:
            category = category_match.group(1).strip()

        # 提取問題
        question_match = re.search(r'問:\s*(.+?)(?:\n|答:|$)', text, re.DOTALL)
        if question_match:
            question = question_match.group(1).strip()
            # 截斷過長的問題
            if len(question) > 50:
                question = question[:50] + "..."

    # 如果從內容提取失敗，嘗試從檔名解析來源
    if not source_name and doc_id:
        source_code = doc_id.split('_')[0].lower()
        source_name = source_map.get(source_code, source_code.upper())

    # 建立顯示名稱
    if question:
        display_name = question
    elif category:
        display_name = f"{source_name} - {category}" if source_name else category
    elif source_name:
        display_name = source_name
    else:
        display_name = "FAQ 資料"

    return {
        "source": source_name or "未知來源",
        "question": question,
        "category": category,
        "display_name": display_name,
        "detail_url": detail_url
    }


def display_sources(sources: list):
    """顯示參考來源"""
    if not sources:
        return

    # 載入檔案映射
    file_mapping = load_file_mapping()
    gemini_id_mapping = load_gemini_id_mapping()

    # 去重（使用內容片段去重）
    seen = set()
    unique_sources = []
    for s in sources:
        text = s.get('text', '')
        key = text[:100] if text else s.get('title', '')
        if key and key not in seen:
            seen.add(key)
            unique_sources.append(s)

    if not unique_sources:
        return

    st.markdown("---")
    st.markdown(f"**📚 參考來源** ({len(unique_sources)} 筆)")

    for i, source in enumerate(unique_sources[:10], 1):
        title = source.get('title', '未知')
        text = source.get('text', '')
        info = parse_source_info(title, text, file_mapping, gemini_id_mapping)

        # 來源圖示
        source_icon = {
            "勞動部": "🏛️",
            "職業安全衛生署": "⚠️",
            "勞動部勞工保險局": "🛡️"
        }.get(info['source'], "📄")

        with st.expander(f"{i}. {source_icon} {info['display_name']}", expanded=False):
            # 顯示來源機關
            if info['source'] and info['source'] != "未知來源":
                st.caption(f"來源：{info['source']}")

            # 顯示分類
            if info.get('category'):
                st.caption(f"分類：{info['category']}")

            # 顯示原始連結
            if info.get('detail_url'):
                st.markdown(f"🔗 [查看原始頁面]({info['detail_url']})")

            # 顯示內容摘要
            if text:
                # 清理內容，移除 metadata 部分
                clean_text = re.sub(r'^(來源|分類|路徑|問|答):.+?\n', '', text, flags=re.MULTILINE)
                clean_text = clean_text.strip()
                if clean_text:
                    st.markdown(f"> {clean_text[:300]}{'...' if len(clean_text) > 300 else ''}")


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
    st.markdown("**🚀 快速查詢：**")

    quick_queries = [
        ("加班費計算", "加班費怎麼計算？"),
        ("特休天數", "特別休假有幾天？怎麼計算？"),
        ("勞保老年給付", "勞保老年給付怎麼領？"),
        ("資遣費計算", "資遣費怎麼計算？"),
        ("職災補償", "發生職業災害可以申請哪些補償？"),
        ("育嬰留停", "育嬰留職停薪怎麼申請？津貼怎麼領？"),
    ]

    cols = st.columns(3)
    selected_query = None
    for idx, (label, q) in enumerate(quick_queries):
        col_idx = idx % 3
        if cols[col_idx].button(f"📌 {label}", key=f"quick_{idx}", use_container_width=True):
            selected_query = q

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

    # 頁尾
    st.markdown("---")
    st.caption("""
    資料來源：勞動部、職業安全衛生署、勞動部勞工保險局官方 FAQ (共 1,110 筆)
    技術：Gemini 2.5 Flash + File Search RAG | v1.0.0
    """)


if __name__ == "__main__":
    main()
