import my_compress


def test_compress1():
    mydata = "a"
    compressed = my_compress.dumps(mydata)
    assert mydata == my_compress.loads(compressed)

def test_compress2():
    mydata = dict()
    mydata[1] = "1"
    mydata["key1"] = "value1"
    mydata["anotherkey"] = [1,2,3,4,5]
    compressed = my_compress.dumps(mydata)
    assert mydata == my_compress.loads(compressed)

def test_compress3():
    with open("compress1.bin", "rb") as f:
        uncompressed = my_compress.load(f)

    assert uncompressed[112].endswith("e")

def test_compress4():
    with open("compress2.bin", "rb") as f:
        uncompressed = my_compress.load(f)

    assert "red" in uncompressed["colors"] and "white" in uncompressed["colors"] and "blue" in uncompressed["colors"]

def test_compress5():
    with open("compress3.bin", "rb") as f:
        uncompressed = my_compress.load(f)

    assert len(uncompressed["glossary"]["GlossDiv"]["GlossList"]["GlossEntry"]) == 7

def test_compress6():
    with open("compress4.bin", "rb") as f:
        uncompressed = my_compress.load(f)

    assert uncompressed["menu"]["popup"]["menuitem"][0]["value"] == "New"

def test_compress7():
    with open("compress5.bin", "rb") as f:
        uncompressed = my_compress.load(f)

    assert uncompressed[0]["type"] == uncompressed[2]["type"]

def test_compress8():
    with open("compress6.bin", "rb") as f:
        uncompressed = my_compress.load(f)

    assert uncompressed[0]["age"] < uncompressed[1]["age"]
