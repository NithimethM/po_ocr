from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from po_extract import (
    DEFAULT_MODEL,
    DEFAULT_URL,
    DEFAULT_VISION_MODEL,
    HEADER_KEYS,
    ITEM_KEYS,
    OllamaError,
    extract_po,
    read_existing_po_numbers,
    list_models,
    to_qty,
    validate,
    write_pos_to_workbook,
)

st.set_page_config(page_title="PO Extract", page_icon="🧾", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """<style>
[data-testid="stMainBlockContainer"], .block-container {max-width: 1080px; padding-top: 2.2rem;}
h1 {font-size: 1.9rem !important; margin-bottom: 0 !important;}
.step {display:flex; align-items:center; gap:.65rem; margin: 1.8rem 0 .6rem; font-weight:600; font-size:1.12rem;}
.step span {background:#2E75B6; color:#fff; border-radius:50%; width:1.75rem; height:1.75rem;
            display:inline-flex; align-items:center; justify-content:center; font-size:.9rem; flex:none;}
footer {visibility: hidden;}
</style>""",
    unsafe_allow_html=True,
)


def step(n: int, text: str) -> None:
    st.markdown(f'<div class="step"><span>{n}</span>{text}</div>', unsafe_allow_html=True)


@st.cache_data(ttl=15, show_spinner=False)
def cached_models(url: str):
    return list_models(url)


def pick(models: list[str], hints: tuple[str, ...], default: str) -> int:
    for i, m in enumerate(models):
        if m == default:
            return i
    for i, m in enumerate(models):
        if any(h in m.lower() for h in hints):
            return i
    return 0


def df_to_items(df: pd.DataFrame) -> list[dict]:
    items = []
    for _, r in df.iterrows():
        txt = lambda k: "" if pd.isna(r.get(k)) else str(r.get(k)).strip()
        row = {"product_code": txt("product_code"), "material_code": txt("material_code"),
               "description": txt("description"), "qty": to_qty(r.get("qty", 0)) if not pd.isna(r.get("qty")) else 0,
               "unit": txt("unit")}
        if row["material_code"] or row["description"] or row["qty"]:
            items.append(row)
    return items


# --------------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    url = st.text_input("Ollama URL", value=DEFAULT_URL, key="po_url")
    models = cached_models(url)
    if models:
        model = st.selectbox("โมเดลข้อความ (PDF ที่มีตัวหนังสือ)", models,
                             index=pick(models, ("qwen",), DEFAULT_MODEL), key="po_model")
        vision_model = st.selectbox("โมเดลอ่านภาพ (สแกน/ภาพถ่าย)", models,
                                    index=pick(models, ("vl", "vision", "gemma3", "minicpm"), DEFAULT_VISION_MODEL),
                                    key="po_vmodel")
        st.caption("🟢 เชื่อมต่อ Ollama แล้ว")
    else:
        model = st.text_input("โมเดลข้อความ", value=DEFAULT_MODEL, key="po_model_t")
        vision_model = st.text_input("โมเดลอ่านภาพ", value=DEFAULT_VISION_MODEL, key="po_vmodel_t")
        st.caption("🔴 เชื่อมต่อ Ollama ไม่ได้" if models is None else "🟡 ยังไม่มีโมเดลใน Ollama")
    force_vision = st.checkbox("บังคับอ่านจากภาพ (แม้ PDF มีตัวหนังสือ)", value=False)
    ctx = st.number_input("Context size", 4096, 65536, 12288, step=2048)
    st.caption("ข้อมูลถูกส่งไปยัง URL ข้างบนเท่านั้น ใช้ Ollama ในเครื่องที่เชื่อถือได้กับเอกสารบริษัท")

try:
    st.page_link("app.py", label="← กลับหน้าวิเคราะห์ Excel")
except Exception:
    pass
st.title("🧾 อ่านใบ PO ด้วย AI")
st.caption("อัปโหลด PO (PDF หรือภาพ) → AI อ่านข้อมูล → ตรวจ/แก้ในตาราง → เขียนลงชีต PO ของไฟล์ Excel (macro SAP)")

# --------------------------------------------------------------------------------------- step 1: files
step(1, "อัปโหลดใบ PO")
files = st.file_uploader("PDF หรือภาพ (เลือกได้หลายไฟล์)", type=["pdf", "png", "jpg", "jpeg"],
                         accept_multiple_files=True, label_visibility="collapsed")
if not files:
    st.info("อัปโหลดใบ PO ได้ทั้ง PDF ที่มีตัวหนังสือ และไฟล์สแกน/ภาพถ่าย (ระบบเลือกวิธีอ่านให้เอง)")
    st.stop()

store: dict = st.session_state.setdefault("po_results", {})
hashes = {hashlib.sha256(f.getvalue()).hexdigest()[:16]: f for f in files}
todo = [h for h in hashes if h not in store]

c_run, c_note = st.columns([2, 3], vertical_alignment="center")
run = c_run.button(f"🔍 อ่านข้อมูลด้วย AI ({len(todo)} ไฟล์ใหม่)" if todo else "🔄 อ่านใหม่ทั้งหมด",
                   type="primary", width="stretch")
c_note.caption("🟢 Ollama พร้อมใช้งาน" if models else "🔴 ยังเชื่อมต่อ Ollama ไม่ได้ — อ่านข้อมูลไม่ได้จนกว่าจะเปิด Ollama")

if run:
    targets = list(hashes) if not todo else todo
    with st.status("กำลังอ่านใบ PO...", expanded=True) as status:
        for h in targets:
            f = hashes[h]
            st.write(f"📄 {f.name}")
            try:
                res = extract_po(f.getvalue(), f.name, url, model, vision_model, int(ctx), force_vision,
                                 log=lambda m: st.write(f"　{m}"))
                res["name"] = f.name
                res["original_issues"] = validate(res["data"], res["source_text"])
                store[h] = res
            except OllamaError as exc:
                store[h] = {"name": f.name, "error": str(exc)}
                st.write(f"　❌ {exc}")
            except Exception as exc:
                store[h] = {"name": f.name, "error": f"อ่านไฟล์ไม่สำเร็จ: {exc}"}
                st.write(f"　❌ {exc}")
        status.update(label="อ่านเสร็จแล้ว — ตรวจข้อมูลด้านล่างก่อนเขียนลง Excel", state="complete", expanded=False)

# --------------------------------------------------------------------------------------- step 2: review
step(2, "ตรวจและแก้ข้อมูล")
final: list[dict] = []
total_issues = 0
ITEM_CFG = {
    "product_code": st.column_config.TextColumn("Product code"),
    "material_code": st.column_config.TextColumn("Material code (16 ตัว)"),
    "description": st.column_config.TextColumn("รายละเอียด", width="large"),
    "qty": st.column_config.NumberColumn("จำนวน", min_value=0),
    "unit": st.column_config.TextColumn("หน่วย"),
}
for h, f in hashes.items():
    res = store.get(h)
    if res is None:
        st.caption(f"⏳ {f.name} — ยังไม่ได้อ่าน (กดปุ่มด้านบน)")
        continue
    if "error" in res:
        st.error(f"{res['name']}: {res['error']}")
        continue
    mode = "ภาพ (OCR)" if res["mode"] == "vision" else "ข้อความ"
    with st.expander(f"{res['name']} — อ่านแบบ{mode} · {len(res['data']['items'])} รายการ"
                     + (f" · ⚠️ {len(res['original_issues'])} ข้อควรตรวจ" if res["original_issues"] else " · ✅"),
                     expanded=bool(res["original_issues"])):
        d = res["data"]
        a, b, c = st.columns(3)
        po_no = a.text_input("PO No.", d["po_no"], key=f"{h}_po")
        po_date = b.text_input("วันที่ PO", d["po_date"], key=f"{h}_pd")
        ship_date = c.text_input("วันส่งของ", d["ship_date"], key=f"{h}_sd")
        sold_to = st.text_input("Sold-to (ผู้ซื้อ)", d["sold_to"], key=f"{h}_st")
        ship_to = st.text_area("Ship-to (สถานที่ส่ง)", d["ship_to"], height=68, key=f"{h}_sh")
        credit = st.text_input("Credit term (ไม่ถูกเขียนลง Excel)", d["credit_term"], key=f"{h}_ct")
        edited = st.data_editor(pd.DataFrame(d["items"], columns=list(ITEM_KEYS)), key=f"{h}_items",
                                num_rows="dynamic", width="stretch", hide_index=True, column_config=ITEM_CFG)
        data = {"po_no": po_no.strip(), "po_date": po_date.strip(), "sold_to": sold_to.strip(),
                "ship_to": ship_to.strip(), "ship_date": ship_date.strip(), "credit_term": credit.strip(),
                "items": df_to_items(edited)}
        issues = validate(data, res["source_text"]) + res.get("notes", [])
        if issues:
            st.warning("**ข้อที่ควรตรวจ**\n\n" + "\n".join(f"- {i}" for i in issues))
        else:
            st.success("ตรวจเทียบกับไฟล์แล้วไม่พบสิ่งผิดปกติ")
        use = st.checkbox("เขียนใบ PO นี้ลง Excel", value=True, key=f"{h}_use")
        if use:
            final.append(data)
            total_issues += len(issues)

# --------------------------------------------------------------------------------------- step 3: write
step(3, "เขียนลงไฟล์ Excel")
tpl = st.file_uploader("ไฟล์ Excel ปลายทาง (ต้องมีชีตชื่อ PO) เช่น PO_Template.xlsm", type=["xlsm", "xlsx"], key="po_tpl")
append = st.checkbox("เขียนต่อท้ายข้อมูลเดิมในชีต PO", value=True)
dup_msgs: list[str] = []
seen_nos: dict[str, int] = {}
for p_ in final:
    seen_nos[p_["po_no"]] = seen_nos.get(p_["po_no"], 0) + 1
dup_msgs += [f"PO No. {n} ถูกเลือกซ้ำ {c} ครั้งในรอบนี้" for n, c in seen_nos.items() if c > 1 and n]
if tpl:
    try:
        already = read_existing_po_numbers(tpl.getvalue())
        dup_msgs += [f"PO No. {n} มีอยู่ในชีต PO ของไฟล์ปลายทางแล้ว (ระวังบันทึกซ้ำ)" for n in seen_nos if n and n in already]
    except Exception:
        pass
if dup_msgs:
    st.warning("**ตรวจพบความเสี่ยงบันทึกซ้ำ**\n\n" + "\n".join(f"- {m}" for m in dup_msgs))
total_issues += len(dup_msgs)
ack = True
if total_issues:
    ack = st.checkbox(f"ฉันตรวจข้อมูลแล้ว (ยังมี {total_issues} ข้อควรตรวจ) — ข้อมูลนี้จะถูกใช้สร้าง SO ใน SAP")
rows_n = sum(len(p["items"]) for p in final)
go = st.button(f"✨ สร้างไฟล์ Excel ({len(final)} ใบ PO · {rows_n} แถว)", type="primary", width="stretch",
               disabled=not (tpl and final and ack and rows_n))
if not tpl:
    st.caption("เลือกไฟล์ปลายทางก่อน — โปรแกรมไม่แก้ไฟล์ต้นฉบับ แต่ให้ดาวน์โหลดไฟล์ใหม่ที่เขียนต่อท้ายแล้ว")
if go:
    try:
        out, total = write_pos_to_workbook(tpl.getvalue(), final, append)
        stem, _, ext = tpl.name.rpartition(".")
        st.session_state["po_out"] = {"bytes": out, "name": f"{stem}_updated.{ext}", "rows": rows_n, "total": total}
    except Exception as exc:
        st.error(f"เขียนไฟล์ไม่สำเร็จ: {exc}")
out = st.session_state.get("po_out")
if out:
    st.success(f"เขียนแล้ว {out['rows']} แถว (จำนวนสะสมในชีต PO = {out['total']})")
    mime = ("application/vnd.ms-excel.sheet.macroEnabled.12" if out["name"].lower().endswith("xlsm")
            else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("⬇️ ดาวน์โหลดไฟล์ Excel", data=out["bytes"], file_name=out["name"], mime=mime,
                       type="primary", width="stretch")
    st.caption("เปิดไฟล์ใหม่ตรวจชีต PO ก่อนรัน macro — ถ้าจะใช้ต่อ ให้ใช้ไฟล์ที่ดาวน์โหลดนี้เป็นไฟล์ปลายทางของรอบถัดไป")
