"""
extract_mathtype.py  (v2 — DSMT/MTEF5 support)
------------------------------------------------
Extracts and decodes MathType formulas from a .docx file.
Supports MTEF v3 (legacy) and MTEF v5 / DSMT format (MathType 5+).

Requirements:
    pip install olefile

Usage:
    python extract_mathtype.py your_document.docx
"""

import zipfile
import olefile
import struct
import sys
import os
import tempfile


# ---------------------------------------------------------------------------
# Font style index → name
# ---------------------------------------------------------------------------
FONT_STYLES = {
    0: 'MathItalic',
    1: 'Math',
    2: 'Text',
    3: 'Function',
    4: 'Variable',
    5: 'LCGreek',
    6: 'UCGreek/Symbol',
    7: 'Symbol',
}

# Symbol font (style index 6) — Symbol font code → Unicode character
SYMBOL_FONT = {
    # Lowercase Greek
    0x61:'α', 0x62:'β', 0x63:'χ', 0x64:'δ', 0x65:'ε', 0x66:'φ',
    0x67:'γ', 0x68:'η', 0x69:'ι', 0x6a:'ϕ', 0x6b:'κ', 0x6c:'λ',
    0x6d:'μ', 0x6e:'ν', 0x6f:'ο', 0x70:'π', 0x71:'θ', 0x72:'ρ',
    0x73:'σ', 0x74:'τ', 0x75:'υ', 0x76:'ϖ', 0x77:'ω', 0x78:'ξ',
    0x79:'ψ', 0x7a:'ζ',
    # Uppercase Greek
    0x41:'Α', 0x42:'Β', 0x43:'Χ', 0x44:'Δ', 0x45:'Ε', 0x46:'Φ',
    0x47:'Γ', 0x48:'Η', 0x49:'Ι', 0x4a:'ϑ', 0x4b:'Κ', 0x4c:'Λ',
    0x4d:'Μ', 0x4e:'Ν', 0x4f:'Ο', 0x50:'Π', 0x51:'Θ', 0x52:'Ρ',
    0x53:'Σ', 0x54:'Τ', 0x55:'Υ', 0x56:'ς', 0x57:'Ω', 0x58:'Ξ',
    0x59:'Ψ', 0x5a:'Ζ',
    # Math operators and punctuation
    0x2b:'+', 0x3d:'=', 0x3c:'<', 0x3e:'>',
    0x28:'(', 0x29:')', 0x5b:'[', 0x5d:']', 0x7b:'{', 0x7d:'}',
    0x2f:'/', 0x2e:'.', 0x2c:',', 0x3a:':', 0x21:'!',
    0xb1:'±', 0xd7:'×', 0xf7:'÷', 0xb7:'·',
    0xd1:'∑', 0xd2:'∫', 0xd6:'√', 0xb9:'∞',
    0xab:'←', 0xae:'→', 0xad:'↑', 0xaf:'↓', 0xdb:'⟺',
    0xa3:'≤', 0xb3:'≥', 0xb9:'∞', 0xba:'°',
    0xc6:'∈', 0xcc:'⊂', 0xcd:'⊃', 0xc7:'∩', 0xc8:'∪',
    0x22:'"', 0x27:"'",
}

TMPL_NAMES = {
    0x00: 'Fence',       0x01: 'Fraction',   0x02: 'Radical',
    0x03: 'Superscript', 0x04: 'Subscript',  0x05: 'Sub+Super',
    0x06: 'Summation',   0x07: 'Integral',   0x08: 'Overbar',
    0x09: 'Underbar',    0x0a: 'Arrow',      0x0b: 'Limit',
    0x0c: 'HSpacing',    0x0d: 'VSpacing',   0x0e: 'TBox',
    0x0f: 'Sqrt',
}


# ---------------------------------------------------------------------------
# DSMT / MTEF v5 decoder
# ---------------------------------------------------------------------------

def decode_dsmt(mtef: bytes) -> str:
    """
    Decode DSMT (MathType 5+) MTEF binary into a readable string.

    Structure:
      Byte 0:   MTEF version (5)
      Byte 1:   Platform (0=Mac, 1=Win)
      Byte 2:   Reserved
      Then: preamble records (RULER, FONT_DEF, SIZE_DEF)
      Then: equation data split into two sections by a 0x0a marker:
            - Template structure section (compact binary tree)
            - Slot data section (readable character nodes)
    """
    if len(mtef) < 3:
        return '<empty>'

    pos = 3  # skip version(1) + platform(1) + reserved(1)

    # --- Parse preamble: RULER + FONT_DEF records ---
    fonts = {}  # style_index → font_name
    eq_start = len(mtef)  # will be updated when we hit SIZE_DEF (0x12)

    while pos < len(mtef):
        tag = mtef[pos]

        if tag == 0x07:   # RULER — skip fixed block
            pos += 1
            if pos < len(mtef):
                block_len = mtef[pos]; pos += 1
                pos += block_len  # skip DSMT product string

        elif tag == 0x01: # LINE — preamble ended, codec string follows
            pos += 1
            # null-terminated codec name (e.g. "WinAllBasicCodePages")
            end = mtef.index(0x00, pos) if 0x00 in mtef[pos:] else len(mtef)
            pos = end + 1

        elif tag == 0x11: # FONT_DEF: style_index(1) + null-terminated name
            pos += 1
            if pos < len(mtef):
                style_idx = mtef[pos]; pos += 1
                end = mtef.index(0x00, pos) if 0x00 in mtef[pos:] else len(mtef)
                font_name = mtef[pos:end].decode('latin-1', errors='replace')
                fonts[style_idx] = font_name
                pos = end + 1

        elif tag == 0x12: # SIZE_DEF — equation data follows immediately after
            pos += 3      # skip SIZE_DEF (3 bytes)
            eq_start = pos
            break

        else:
            pos += 1  # unknown preamble byte, skip

    if eq_start >= len(mtef):
        return '<no equation data>'

    eq = mtef[eq_start:]

    # --- Find the 0x0a separator between template section and slot section ---
    # The slot section begins right after the first 0x0a byte that appears
    # after the template structure (which ends with 0x0c).
    slot_start = None
    for i, b in enumerate(eq):
        if b == 0x0a and i > 0:
            slot_start = i + 1
            break

    if slot_start is None:
        return '<no slot data>'

    slot_data = eq[slot_start:]

    # --- Decode slot data ---
    # Each slot is a sequence of nodes terminated by END (0x00).
    # Node types:
    #   0x00 = END (closes current slot)
    #   0x01 = LINE + flags(1)                          — 2 bytes
    #   0x02 = CHAR + flags(1) + font_raw(1) + char(1) + null(1)  — 5 bytes
    #          if flags == 0x04: one extra embellishment byte after null
    #   0x03 = TMPL + flags(1) + selector(1) + null(1) — 4 bytes
    #   0x0b = SUB size marker                          — 1 byte
    #   0x0d = SYM size marker                          — 1 byte
    #   0x0f = FULL size + param(1)                     — 2 bytes

    slots = []
    current = []
    i = 0

    while i < len(slot_data):
        b = slot_data[i]

        if b == 0x00:   # END of slot
            slots.append(''.join(current))
            current = []
            i += 1

        elif b == 0x01: # LINE node
            i += 2      # skip tag + flags

        elif b == 0x02: # CHAR node
            if i + 4 < len(slot_data):
                flags    = slot_data[i + 1]
                font_raw = slot_data[i + 2]
                char_code= slot_data[i + 3]
                i += 5  # tag + flags + font + char + null

                font_idx = font_raw & 0x07

                if font_idx == 6:   # UC Greek / Symbol font
                    c = SYMBOL_FONT.get(char_code,
                        chr(char_code) if 32 <= char_code < 127 else f'[0x{char_code:02X}]')
                else:
                    c = chr(char_code) if 32 <= char_code < 127 else f'[0x{char_code:02X}]'

                current.append(c)

                # flags=0x04 means one extra embellishment byte
                if flags == 0x04 and i < len(slot_data):
                    i += 1

            else:
                i += 1

        elif b == 0x03: # TMPL node (4 bytes)
            i += 4

        elif b in (0x0b, 0x0d):  # size markers (1 byte)
            i += 1

        elif b == 0x0f:          # FULL size (2 bytes)
            i += 2

        else:
            i += 1  # unknown, skip

    # --- Reconstruct readable formula from slots ---
    non_empty = [s for s in slots if s.strip()]
    return '  |  '.join(non_empty) if non_empty else '<decoded: empty slots>'


# ---------------------------------------------------------------------------
# MTEF v3 decoder (legacy)
# ---------------------------------------------------------------------------

MATH_CHARS_V3 = {
    (2, 0x61): 'α', (2, 0x62): 'β', (2, 0x67): 'γ', (2, 0x64): 'δ',
    (2, 0x65): 'ε', (2, 0x7a): 'ζ', (2, 0x68): 'η', (2, 0x71): 'θ',
    (2, 0x69): 'ι', (2, 0x6b): 'κ', (2, 0x6c): 'λ', (2, 0x6d): 'μ',
    (2, 0x6e): 'ν', (2, 0x78): 'ξ', (2, 0x70): 'π', (2, 0x72): 'ρ',
    (2, 0x73): 'σ', (2, 0x74): 'τ', (2, 0x75): 'υ', (2, 0x66): 'φ',
    (2, 0x63): 'χ', (2, 0x79): 'ψ', (2, 0x77): 'ω',
    (2, 0x47): 'Γ', (2, 0x44): 'Δ', (2, 0x51): 'Θ', (2, 0x4c): 'Λ',
    (2, 0x4e): 'Ξ', (2, 0x50): 'Π', (2, 0x53): 'Σ', (2, 0x55): 'Υ',
    (2, 0x46): 'Φ', (2, 0x59): 'Ψ', (2, 0x57): 'Ω',
    (2, 0xb1): '±', (2, 0xb4): '×', (2, 0xf7): '÷',
    (2, 0xd2): '∫', (2, 0xd1): '∑', (2, 0xd6): '√', (2, 0xb9): '∞',
}


class MTEFV3Decoder:
    def __init__(self, data):
        self.data = data
        self.pos  = 0
        self.out  = []

    def decode(self):
        if len(self.data) < 2:
            return '<empty>'
        self.pos = 2  # skip version + platform
        self._parse()
        return ''.join(self.out).strip() or '<decoded: empty>'

    def _peek(self):
        return self.data[self.pos] if self.pos < len(self.data) else None

    def _read(self):
        b = self.data[self.pos]; self.pos += 1; return b

    def _parse(self):
        while self.pos < len(self.data):
            tag = self._peek()
            if tag is None: break
            rec = tag & 0x0F
            opt = tag >> 4
            self._read()

            if rec == 0:   break
            elif rec == 1: self._parse()
            elif rec == 2:
                if self.pos + 1 < len(self.data):
                    tf = self._read(); ch = self._read()
                    if (tf, ch) in MATH_CHARS_V3:
                        self.out.append(MATH_CHARS_V3[(tf, ch)])
                    elif 32 <= ch < 127:
                        self.out.append(chr(ch))
                    else:
                        self.out.append(f'[0x{ch:02X}@tf{tf}]')
            elif rec == 3:
                if self.pos < len(self.data):
                    sel = self._read()
                    self._emit_template(sel, opt)
            elif rec == 4: self._parse()
            elif rec == 5:
                if self.pos + 1 < len(self.data):
                    rows = self._read(); cols = self._read()
                    self.out.append(f'[Matrix {rows}×{cols}]')
            elif rec == 9:
                if self.pos < len(self.data): self._read()
            elif rec in (10, 11, 12, 13, 14): pass
            elif rec == 255:
                if self.pos + 1 < len(self.data):
                    self._read(); self._read()
            else:
                break

    def _emit_template(self, sel, opt):
        TMPL = {1:'(%s / %s)', 2:'√(%s)', 3:'(%s)^(%s)',
                4:'(%s)_(%s)', 5:'(%s)_(%s)^(%s)',
                6:'Σ_(%s)^(%s)[%s]', 7:'∫_(%s)^(%s)[%s]', 0:'(%s)'}
        saved = self.out
        self.out = []
        self._parse()
        a = ''.join(self.out); self.out = []
        if sel in (1, 3, 4, 6, 7):
            self._parse()
            b = ''.join(self.out); self.out = []
        if sel in (6, 7):
            self._parse()
            c = ''.join(self.out); self.out = []
        self.out = saved
        if sel == 1:   self.out.append(f'({a} / {b})')
        elif sel == 2: self.out.append(f'√({a})')
        elif sel == 3: self.out.append(f'({a})^({b})')
        elif sel == 4: self.out.append(f'({a})_({b})')
        elif sel == 5: self.out.append(f'({a})_({b})^({c if "c" in dir() else ""})')
        elif sel == 6: self.out.append(f'Σ_({a})^({b})[{c}]')
        elif sel == 7: self.out.append(f'∫_({a})^({b})[{c}]')
        elif sel == 0: self.out.append(f'({a})')
        else:          self.out.append(f'[tmpl{sel}:{a}]')


# ---------------------------------------------------------------------------
# OLE extractor
# ---------------------------------------------------------------------------

def extract_mathtype_formulas(docx_path: str):
    if not os.path.exists(docx_path):
        print(f'ERROR: File not found: {docx_path}')
        sys.exit(1)

    print(f'Opening: {docx_path}')
    print('=' * 70)

    count = 0

    with zipfile.ZipFile(docx_path, 'r') as z:
        embeddings = sorted([
            f for f in z.namelist()
            if 'embeddings' in f.lower() and f.lower().endswith('.bin')
        ])

        if not embeddings:
            print('No embedded OLE objects found.')
            return

        print(f'Found {len(embeddings)} embedded OLE object(s).\n')

        for idx, emb_path in enumerate(embeddings, 1):
            data = z.read(emb_path)

            if data[:8] != b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
                continue  # not an OLE file

            tmp = os.path.join(tempfile.gettempdir(), f'_mt_{idx}.bin')
            with open(tmp, 'wb') as f:
                f.write(data)

            try:
                ole = olefile.OleFileIO(tmp)

                if not ole.exists('Equation Native'):
                    ole.close()
                    continue

                raw = ole.openstream('Equation Native').read()
                ole.close()

                if len(raw) < 28:
                    continue

                header_size = struct.unpack_from('<I', raw, 0)[0]
                mt_major    = raw[4]
                mt_minor    = raw[5]
                inline      = raw[6]
                mtef        = raw[header_size:]

                if len(mtef) < 2:
                    continue

                mtef_version = mtef[0]

                # Decode based on internal MTEF version
                if mtef_version >= 5:
                    decoded = decode_dsmt(mtef)
                else:
                    decoded = MTEFV3Decoder(mtef).decode()

                print(f'Formula #{idx}  [{emb_path}]')
                print(f'  MathType header  : {mt_major}.{mt_minor}  |  MTEF v{mtef_version}  |  {"inline" if inline else "display"}')
                print(f'  Decoded          : {decoded}')
                print()
                count += 1

            except Exception as e:
                print(f'Formula #{idx}: error — {e}')
                print()
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

    print('=' * 70)
    print(f'Done. {count} formula(s) decoded.')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python extract_mathtype.py <document.docx>')
        sys.exit(1)
    extract_mathtype_formulas(sys.argv[1])