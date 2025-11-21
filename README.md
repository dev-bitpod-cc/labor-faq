# 勞動法規 FAQ 查詢系統

使用 Gemini File Search (RAG) 技術的勞動法規 FAQ 智慧查詢系統。

## 資料來源

| 機關 | 簡稱 | FAQ 數量 |
|------|------|----------|
| 勞動部 | MOL | 383 筆 |
| 職業安全衛生署 | OSHA | 124 筆 |
| 勞動部勞工保險局 | BLI | 987 筆 |
| **總計** | | **1,494 筆** |

## 主題範圍

- 勞動契約、僱傭關係
- 工時、加班、休息時間
- 特別休假、國定假日
- 工資、基本工資、加班費
- 資遣、解僱、資遣費
- 退休、勞工退休金
- 勞保、就保、職災保險
- 職業安全、工作環境
- 職業病、職業災害補償

## 本地開發

```bash
# 安裝依賴
pip install -r requirements.txt

# 設定環境變數
cp .env.example .env
# 編輯 .env，填入 GEMINI_API_KEY

# 執行
streamlit run app.py
```

## Streamlit Cloud 部署

1. Fork 此專案到你的 GitHub
2. 在 [Streamlit Cloud](https://share.streamlit.io/) 建立新應用
3. 連結你的 GitHub 倉庫
4. 在 Secrets 中設定 `GEMINI_API_KEY`

## 技術架構

- **前端**: Streamlit
- **AI 模型**: Gemini 2.5 Flash
- **RAG**: Gemini File Search
- **資料格式**: Plain Text (語義優化)

## 注意事項

- 本系統僅供參考，不構成法律建議
- 法規可能隨時更新，建議查閱最新法規
- 個案問題請洽詢主管機關或專業人士

## 授權

MIT License
