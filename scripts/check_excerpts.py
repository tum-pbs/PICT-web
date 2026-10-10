"""Check that the code excerpts on the website are verbatim copies of the PICT source files.

Every fenced code block whose options give a file as title and a start line, e.g.
    ```python title="backward_facing_step_sample.py" linenums="135" hl_lines="2 3"
must match the lines of that file in the PICT repository, starting at that line. Blocks without both options are not checked.
Prints one line per checked block and exits with 1 if any block differs, so it can be re-run after the solver code changes.

usage: python check_excerpts.py <PICT repository> [<docs dir>]"""
import glob, os, re, sys

PICT = sys.argv[1]
DOCS = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
FENCE = re.compile(r"^(\s*)(`{3,})(\w*)\s*(.*)$")
OPTION = re.compile(r'(\w+)="([^"]*)"')

def code_blocks(path):
    # (line number in the page, options, code lines without the fence indentation)
    lines = open(path, encoding="utf-8").read().split("\n")
    i = 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if m is None:
            i += 1
            continue
        indent, fence, options = m.group(1), m.group(2), dict(OPTION.findall(m.group(4)))
        start = i
        code = []
        i += 1
        while i < len(lines) and lines[i].strip() != fence:
            code.append(lines[i][len(indent):] if lines[i].startswith(indent) else lines[i].lstrip())
            i += 1
        i += 1
        yield start + 1, options, code

failed = checked = 0
for page in sorted(glob.glob(os.path.join(DOCS, "**", "*.md"), recursive=True)):
    for page_line, options, code in code_blocks(page):
        if "title" not in options or "linenums" not in options:
            continue
        source = os.path.join(PICT, options["title"])
        first = int(options["linenums"])
        where = "%s:%d" % (os.path.relpath(page, DOCS), page_line)
        checked += 1
        if not os.path.isfile(source):
            print("MISSING %s: %s does not exist" % (where, options["title"]))
            failed += 1
            continue
        original = open(source, encoding="utf-8").read().split("\n")[first - 1:first - 1 + len(code)]
        diff = [k for k, (a, b) in enumerate(zip(code, original)) if a.rstrip() != b.rstrip()]
        if len(original) < len(code):
            diff.append(len(original))
        if diff:
            k = diff[0]
            print("DIFF    %s: %s line %d\n    page:   %r\n    source: %r" % (where, options["title"], first + k,
                code[k] if k < len(code) else None, original[k] if k < len(original) else None))
            failed += 1
        else:
            print("ok      %s: %s lines %d-%d" % (where, options["title"], first, first + len(code) - 1))
print("%d excerpts checked, %d differ" % (checked, failed))
sys.exit(1 if failed else 0)
