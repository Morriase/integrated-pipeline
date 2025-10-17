import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def docx_to_text(docx_path: Path) -> str:
    with zipfile.ZipFile(docx_path) as zf:
        data = zf.read("word/document.xml")
    root = ET.fromstring(data)
    parts = []
    for para in root.findall(".//w:p", NS):
        texts = [node.text for node in para.findall(".//w:t", NS) if node.text]
        if texts:
            parts.append("".join(texts))
    return "\n".join(parts)


def main(args):
    if not args:
        print("Usage: python extract_docx.py <file1.docx> [file2.docx ...]")
        return 1
    for arg in args:
        path = Path(arg).resolve()
        if not path.exists():
            print(f"[skip] {path} does not exist")
            continue
        if path.suffix.lower() != ".docx":
            print(f"[skip] {path} is not a .docx file")
            continue
        text = docx_to_text(path)
        out_path = path.with_suffix(".txt")
        out_path.write_text(text, encoding="utf-8")
        print(f"[ok] Extracted to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
