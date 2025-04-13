from Flask_app_predict import app
import pytest

@pytest.fixture
def client():
    return app.test_client()

def test_hellow(client):
    resp=client.get('/')
    assert resp.status_code==200
    assert resp.data.decode()=="<p> Hello World!</p>"

def test_predict(client):
    test_data={
        "lastflag": 10,
        "flag": "S0",
        "dstbytes": 0.0,
        "srcbytes": 2000.0,
        "diffsrvrate": 0.2,
        "countt": 0.0,
        "dsthostsrvcount": 0.0,
        "dsthostdiffsrvrate": 0.14,
        "suattempted": 0,
        "rootshell": 0,
        "numfailedlogins": 0,
        "land": 0,
        "numfilecreations": 0,
        "wrongfragment": 0,
        "urgent": 0,
        "srvdiffhostrate": 0.0,
        "protocoltype": "tcp",
        "isguestlogin": 0,
        "dsthostsrvdiffhostrate": 0.22,
        "dsthostcount": 100,
        "numaccessfiles": 0,
        "srvcount": 8,
        "numcompromised": 0,
        "dsthostsamesrcportrate": 0.12,
        "dsthostsrvrerrorrate": 0.14,
        "numroot": 0,
        "duration": 20,
        "hot": 0,
        "numshells": 0
    }  
    resp=client.post("/predict",json=test_data)
    assert resp.status_code==200
    assert resp.json=={
        "DOS probability(%)": 3.11,
        "NORMAL probability(%)": 12.13,
        "OTHER ATTACKS probability(%)": 44.66,
        "PROBE probability(%)": 40.1
    }