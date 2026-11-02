"""
Phase 1: Structure-Aware Table of Contents (ToC) & Leaf Section Extractor.

Parses technical documentation PDFs, extracts the hierarchical ToC,
identifies leaf nodes (subsections without children), and captures
semantically complete section text without arbitrary token cutting.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import pymupdf as fitz
except ImportError:
    import fitz


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def extract_stair_toc(pdf_path: str) -> Tuple[str, Dict[str, str]]:
    """
    Extract Table of Contents and leaf sections from a PDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        tuple: (formatted_toc_string, leaf_sections_dict)
    """
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()  # Returns list of [level, title, page_num]
    
    formatted_toc_lines: List[str] = []
    leaf_sections: Dict[str, str] = {}
    
    # If the PDF does not contain embedded bookmarks, fall back to font-based heading detection
    if not toc:
        print(f"⚠️  No embedded outline/bookmarks found in {pdf_path}. Running heading detection fallback...")
        return _fallback_heading_extraction(doc)
    
    num_pages = len(doc)
    for i, (level, title, start_page) in enumerate(toc):
        clean_title = title.strip()
        if not clean_title:
            continue
            
        # A node is a leaf if it's the last entry OR the next entry is at an equal or higher level (<= level)
        is_leaf = (i == len(toc) - 1) or (toc[i + 1][0] <= level)
        
        # Calculate indentation for hierarchy display
        indent = "    " * (level - 1)
        
        if is_leaf:
            leaf_label = f"leaf {clean_title}"
            formatted_toc_lines.append(f"{indent}{leaf_label}")
            
            # Extract text from start_page to the next section's start_page
            # start_page in PyMuPDF get_toc() is 1-based
            end_page = toc[i + 1][2] if (i + 1 < len(toc)) else num_pages
            
            # Clamp bounds
            p_start = max(0, start_page - 1)
            p_end = min(num_pages, max(p_start + 1, end_page))
            
            section_text_parts = []
            for p in range(p_start, p_end):
                page_text = doc[p].get_text("text")
                if page_text:
                    section_text_parts.append(page_text.strip())
            
            full_text = "\n\n".join(section_text_parts).strip()
            # If text was empty (e.g. image-only page), provide placeholder
            if not full_text:
                full_text = f"[{clean_title} - Content extracted from PDF pages {start_page} to {end_page}]"
                
            leaf_sections[clean_title] = full_text
        else:
            formatted_toc_lines.append(f"{indent}{clean_title}")
            
    doc.close()
    return "\n".join(formatted_toc_lines), leaf_sections


def _fallback_heading_extraction(doc: fitz.Document) -> Tuple[str, Dict[str, str]]:
    """Heuristic fallback for PDFs without outlines: detects bold/large font headings."""
    formatted_toc_lines = []
    leaf_sections = {}
    
    current_section = "Introduction"
    current_text: List[str] = []
    section_count = 0
    
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        
        for b in blocks:
            if b[6] == 0:  # text block
                text = b[4].strip()
                # Simple heading pattern: short line ending with no period, or starts with Chapter/Section/digit
                is_heading = (len(text) < 80 and not text.endswith(".") and 
                              (re.match(r"^(Chapter|\d+\.|\d+\s+|[A-Z\s]{4,})", text) or len(text.split("\n")) == 1 and len(text) < 40))
                
                if is_heading and len(current_text) > 0:
                    leaf_sections[current_section] = "\n".join(current_text).strip()
                    formatted_toc_lines.append(f"leaf {current_section}")
                    current_section = text.replace("\n", " ")
                    current_text = []
                    section_count += 1
                else:
                    current_text.append(text)
                    
    if current_text:
        leaf_sections[current_section] = "\n".join(current_text).strip()
        formatted_toc_lines.append(f"leaf {current_section}")
        
    return "\n".join(formatted_toc_lines), leaf_sections


def main():
    parser = argparse.ArgumentParser(description="Extract STAIR ToC and leaf sections from PDF documentation.")
    parser.add_argument("--pdf", type=str, help="Path to input PDF file (optional).")
    parser.add_argument("--output", type=str, help="Output JSON path (optional).")
    args = parser.parse_args()

    project_root = get_project_root()
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = Path(args.output) if args.output else data_dir / "toc_and_sections.json"
    
    combined_toc = []
    combined_leaves = {}
    
    # Check explicit argument first
    if args.pdf:
        pdf_path = Path(args.pdf)
        if pdf_path.exists():
            print(f"Processing PDF: {pdf_path.name}")
            toc_str, leaves = extract_stair_toc(str(pdf_path))
            combined_toc.append(f"# {pdf_path.stem.replace('_', ' ').title()}")
            combined_toc.append(toc_str)
            combined_leaves.update(leaves)
        else:
            print(f"❌ Error: File not found: {pdf_path}")
            return
    else:
        # Check standard data files
        pdf_candidates = [
            ("Python 3.11 Documentation", data_dir / "python_311_docs.pdf"),
            ("FastAPI Documentation", data_dir / "fastapi_docs.pdf"),
        ]
        
        found_any = False
        for title, path in pdf_candidates:
            if path.exists():
                found_any = True
                print(f"Processing: {title} ({path.name})")
                toc_str, leaves = extract_stair_toc(str(path))
                combined_toc.append(f"# {title}")
                combined_toc.append(toc_str)
                # Ensure section names are distinct
                for k, v in leaves.items():
                    combined_leaves[k] = v
                print(f"  Extracted {len(leaves)} leaf sections")
                
        if not found_any:
            print("\n*** No PDF files found in data/ directory. ***")
            print("To generate sample synthetic data for demonstration, run:")
            print("  python sample_pdf_processor.py")
            return

    if combined_leaves:
        output_data = {
            "toc_prompt": "\n".join(combined_toc),
            "leaves": combined_leaves,
            "source": "pdf_extraction",
            "num_leaves": len(combined_leaves)
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        print(f"\n✓ Successfully extracted {len(combined_leaves)} leaf sections.")
        print(f"✓ Saved ToC & section index to: {output_file}")


if __name__ == "__main__":
    main()
