from statistical_debugger import TarantulaDebugger

def remove_html_markup(s: str) -> str:
    tag = False
    quote = False
    out = ""

    for c in s:
        if c == '<' and not quote:
            tag = True
        elif c == '>' and not quote:
            tag = False
        elif c == '"' or c == "'" and tag:
            quote = not quote
        elif not tag:
            out = out + c

    return out


debugger = TarantulaDebugger()
with debugger:
    assert remove_html_markup("abc") == "abc"
with debugger:
    assert remove_html_markup('<b>abc</b>') == "abc"
with debugger:
    assert remove_html_markup('"abc"') == '"abc"'

print(debugger.event_table_text(args=True))
debugger.pprint()
