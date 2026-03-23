"""인식 로그 조회 페이지."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from datetime import date, datetime, timedelta

import requests
import streamlit as st

from ui.components.sidebar import render_sidebar

st.set_page_config(page_title="로그 — DeepFace Live", layout="wide")
render_sidebar()

st.title("📊 인식 로그")

API_BASE = st.session_state.get("api_base_url", "http://localhost:8000/api")

# ── 필터 ──
st.subheader("🔍 필터")
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    start_date = st.date_input("시작 날짜", value=date.today() - timedelta(days=7))
with col_f2:
    end_date = st.date_input("종료 날짜", value=date.today())
with col_f3:
    # 인물 필터
    filter_person_id = None
    try:
        resp = requests.get(f"{API_BASE}/persons", timeout=5)
        if resp.status_code == 200:
            persons = resp.json()
            person_opts = {"전체": None}
            for p in persons:
                label = p.get("display_name") or p["name"]
                person_opts[f"{label} (ID:{p['id']})"] = p["id"]
            selected = st.selectbox("인물 필터", list(person_opts.keys()))
            filter_person_id = person_opts[selected]
    except Exception:
        st.selectbox("인물 필터", ["전체"])

col_page1, col_page2 = st.columns(2)
with col_page1:
    page = st.number_input("페이지", min_value=1, value=1, step=1)
with col_page2:
    page_size = st.selectbox("페이지 크기", [20, 50, 100], index=1)

# ── 로그 조회 ──
st.divider()
st.subheader("📋 로그 목록")

params = {
    "start_date": datetime.combine(start_date, datetime.min.time()).isoformat(),
    "end_date": datetime.combine(end_date, datetime.max.time()).isoformat(),
    "page": page,
    "page_size": page_size,
}
if filter_person_id is not None:
    params["person_id"] = filter_person_id

try:
    resp = requests.get(f"{API_BASE}/logs", params=params, timeout=10)
    if resp.status_code == 200:
        logs = resp.json()
        if logs:
            table_data = []
            for log in logs:
                table_data.append(
                    {
                        "ID": log.get("id"),
                        "시간": log.get("recognized_at", "")[:19],
                        "인물": log.get("person_name") or "Unknown",
                        "인물ID": log.get("person_id") or "-",
                        "신뢰도": f"{log.get('confidence', 0) * 100:.1f}%",
                        "스냅샷": log.get("snapshot_path") or "-",
                    }
                )
            st.dataframe(table_data, use_container_width=True)
            st.caption(f"총 {len(logs)}건 표시 (페이지 {page})")
        else:
            st.info("해당 기간에 인식 기록이 없습니다.")
    else:
        st.warning(f"로그를 불러올 수 없습니다 (HTTP {resp.status_code})")
except requests.ConnectionError:
    st.error("API 서버에 연결할 수 없습니다.")
except Exception as e:
    st.error(f"오류: {e}")

# ── 통계 차트 ──
st.divider()
st.subheader("📈 인식 통계")

try:
    resp = requests.get(f"{API_BASE}/logs/stats", timeout=10)
    if resp.status_code == 200:
        stats = resp.json()
        total = stats.get("total_logs", 0)
        st.metric("총 인식 기록", f"{total}건")

        person_stats = stats.get("person_stats", [])
        if person_stats:
            try:
                import pandas as pd
                import plotly.express as px

                df = pd.DataFrame(person_stats)
                # person_stats 구조에 따라 컬럼명 조정
                name_col = (
                    "person_name"
                    if "person_name" in df.columns
                    else ("name" if "name" in df.columns else df.columns[0])
                )
                count_col = (
                    "count"
                    if "count" in df.columns
                    else ("total" if "total" in df.columns else df.columns[-1])
                )

                fig = px.bar(
                    df,
                    x=name_col,
                    y=count_col,
                    title="인물별 인식 빈도",
                    labels={name_col: "인물", count_col: "인식 횟수"},
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.warning("plotly가 설치되지 않았습니다. 차트를 표시할 수 없습니다.")
                st.json(person_stats)
        else:
            st.info("인물별 통계 데이터가 없습니다.")
    else:
        st.warning("통계를 불러올 수 없습니다.")
except requests.ConnectionError:
    st.error("API 서버에 연결할 수 없습니다.")
except Exception as e:
    st.error(f"오류: {e}")

# ── 로그 정리 ──
st.divider()
st.subheader("🧹 로그 정리")
col_clean1, col_clean2 = st.columns([2, 1])

with col_clean1:
    retention_days = st.number_input(
        "보관 기간 (일)", min_value=1, max_value=365, value=30, step=1
    )
with col_clean2:
    st.write("")  # 간격 조절
    st.write("")
    if st.button("🗑️ 오래된 로그 삭제", type="primary"):
        try:
            resp = requests.delete(
                f"{API_BASE}/logs/cleanup",
                params={"retention_days": retention_days},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                deleted = data.get("deleted_count", 0)
                st.success(f"✅ {deleted}건의 로그가 삭제되었습니다.")
            else:
                st.error(f"삭제 실패 (HTTP {resp.status_code})")
        except requests.ConnectionError:
            st.error("API 서버에 연결할 수 없습니다.")
        except Exception as e:
            st.error(f"오류: {e}")
