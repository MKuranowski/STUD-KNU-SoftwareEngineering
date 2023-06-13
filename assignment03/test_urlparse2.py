from StatRepair import StatRepair
from urlparse import urlparse

def test_urlparse():
    debugger = StatRepair()

    with debugger.collect_pass():
        urlparse("http://aaaa.com")

    with debugger.collect_pass():
        urlparse("http://aaaa.com#aaa#bbb")

    with debugger.collect_pass():
        urlparse("https://[2001:db8:85a3:8d3:1319:8a2e:370:7348]:443/")

    with debugger.collect_fail():
        urlparse("http://aaaa.com#aaa#bbb;;;")


    line, dist = debugger.mostsimilarstmt(debugger.rank()[0])

    assert (line == "parts = ip_str.split(':')" or line == 'url = url.replace(b, "")') and dist == 20
