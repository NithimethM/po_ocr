"""
psd2Excel_ollama.py  (command-line version; needs po_extract.py in the same folder)
--------------------------------------------------
Reads a Purchase Order PDF/image with a local Ollama model and writes the lines into the
Excel "PO" sheet used by the SAP auto-create-SO macro (same cell layout as psd2Excel.py).

    python psd2Excel_ollama.py "PO.pdf" "PO_Template.xlsm"
    python psd2Excel_ollama.py "PO.pdf" "PO_Template.xlsm" "Output.xlsm" --yes
    python psd2Excel_ollama.py "PO.pdf" --dry-run            # extract + check only
    python psd2Excel_ollama.py "scan.pdf" "PO_Template.xlsm" --vision
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import openpyxl

from po_extract import (
    DEFAULT_MODEL,
    DEFAULT_URL,
    DEFAULT_VISION_MODEL,
    OllamaError,
    extract_po,
    validate,
    write_po_to_sheet,
)


def print_preview(data):
    print("\n" + "=" * 70)
    for k, label in (("po_no", "PO No"), ("po_date", "PO Date"), ("ship_date", "Ship date"),
                     ("credit_term", "Credit term"), ("sold_to", "Sold-to"), ("ship_to", "Ship-to")):
        print(f"{label:12}: {data[k]}")
    print(f"Items found : {len(data['items'])}")
    for n, it in enumerate(data["items"], start=1):
        print(f"  {n:>2}. {it['material_code']:<18} {it['qty']:>8} {it['unit']:<4} {it['description'][:45]}")
    print("=" * 70)


def write_to_excel(data, excel_path, output_path=None, append=True):
    output_path = output_path or excel_path
    source_path = output_path if os.path.exists(output_path) else excel_path
    wb = openpyxl.load_workbook(source_path, keep_vba=True)
    total = write_po_to_sheet(wb["PO"], data, append)
    wb.save(output_path)
    return total


def main():
    p = argparse.ArgumentParser(description="Extract a PO with Ollama and write it to the macro's PO sheet.")
    p.add_argument("pdf")
    p.add_argument("excel", nargs="?", help="Excel template/workbook with a 'PO' sheet (not needed with --dry-run)")
    p.add_argument("output", nargs="?", help="Output path (default: append to the excel file itself)")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"text model (default {DEFAULT_MODEL})")
    p.add_argument("--vision-model", default=DEFAULT_VISION_MODEL, help=f"for scans (default {DEFAULT_VISION_MODEL})")
    p.add_argument("--vision", action="store_true", help="force the image path even if the PDF has text")
    p.add_argument("--ctx", type=int, default=12288, help="Ollama context size (default 12288)")
    p.add_argument("--dry-run", action="store_true", help="extract and check only; do not touch Excel")
    p.add_argument("--yes", "-y", action="store_true", help="write without asking for confirmation")
    p.add_argument("--json", dest="json_out", help="also save the extracted data to this JSON file (audit trail)")
    args = p.parse_args()
    if not args.dry_run and not args.excel:
        p.error("ต้องระบุไฟล์ Excel (หรือใช้ --dry-run)")

    print(f"Reading: {args.pdf}")
    with open(args.pdf, "rb") as f:
        raw = f.read()
    try:
        res = extract_po(raw, args.pdf, args.url, args.model, args.vision_model, args.ctx, args.vision, log=print)
    except OllamaError as exc:
        sys.exit(str(exc))
    data = res["data"]
    issues = validate(data, res["source_text"]) + res["notes"]
    print_preview(data)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"data": data, "issues": issues}, f, ensure_ascii=False, indent=2)
        print(f"บันทึก JSON: {args.json_out}")

    if issues:
        print(f"\nพบ {len(issues)} ข้อที่ควรตรวจ:")
        for i in issues:
            print(f"  ! {i}")
    else:
        print("\nตรวจเทียบกับไฟล์แล้วไม่พบสิ่งผิดปกติ")
    if args.dry_run:
        return
    if not data["items"]:
        sys.exit("ไม่มีรายการสินค้าให้เขียน")
    if not args.yes and input("\nเขียนลง Excel? [y/N] ").strip().lower() != "y":
        sys.exit("ยกเลิก ไม่ได้เขียนอะไรลงไฟล์")
    total = write_to_excel(data, args.excel, args.output)
    print(f"เขียนแล้ว {len(data['items'])} แถว (จำนวนสะสมในชีต PO = {total}) → {args.output or args.excel}")


if __name__ == "__main__":
    main()
