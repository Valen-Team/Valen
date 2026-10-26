"""Credential routing cannot alter the request or discard cached predictions."""
import json
import httpx
import pytest

from evaluation.det.jev import ENDPOINT
from evaluation.det.jev_keys import RotatingClient


def pool(tmp_path,respond):
    calls=[]
    def handler(request):
        auth=request.headers['Authorization']
        calls.append((auth,json.loads(request.content)))
        return httpx.Response(respond(auth),json={'ok':True})
    def factory(**kwargs):return httpx.Client(transport=httpx.MockTransport(handler),**kwargs)
    client=RotatingClient([('key2','secret-a'),('key3','secret-b'),('key4','secret-c')],
        tmp_path/'status.json',client_factory=factory)
    return client,calls


def test_round_robin_keeps_requests_identical_and_credentials_out_of_logs(tmp_path):
    client,calls=pool(tmp_path,lambda _:200)
    with client:
        for _ in range(5):assert client.post(ENDPOINT,json={'model':'same'}).status_code==200
    assert [a for a,_ in calls]==['Bearer secret-a','Bearer secret-b','Bearer secret-c','Bearer secret-a','Bearer secret-b']
    assert all(b=={'model':'same'} for _,b in calls)
    text=(tmp_path/'status.json').read_text()
    assert all(secret not in text for secret in ('secret-a','secret-b','secret-c'))


@pytest.mark.parametrize('status',[401,402,403,429])
def test_rejected_or_limited_key_fails_over_without_scoring_an_error(tmp_path,status):
    client,calls=pool(tmp_path,lambda a:status if a=='Bearer secret-a' else 200)
    with client:
        assert client.post(ENDPOINT,json={'sample':1}).status_code==200
        assert len(calls)==2 and calls[0][1]==calls[1][1]
        assert client.post(ENDPOINT,json={'sample':2}).status_code==200
        assert client.post(ENDPOINT,json={'sample':3}).status_code==200
    assert sum(a=='Bearer secret-a' for a,_ in calls)==1


def test_all_credentials_expired_stops_and_does_not_return_a_zero_prediction(tmp_path):
    client,calls=pool(tmp_path,lambda _:402)
    with client,pytest.raises(RuntimeError,match='All configured'):
        client.post(ENDPOINT,json={'sample':1})
    assert len(calls)==3


@pytest.mark.parametrize('status',[500,502,503,504,529,599])
def test_service_errors_retry_without_disabling_valid_credentials(tmp_path,status):
    client,calls=pool(tmp_path,lambda a:status if a=='Bearer secret-a' else 200)
    with client:
        assert client.post(ENDPOINT,json={'sample':1}).status_code==200
        assert len(calls)==2 and calls[0][1]==calls[1][1]
        assert client.post(ENDPOINT,json={'sample':2}).status_code==200
        assert client.post(ENDPOINT,json={'sample':3}).status_code==200
        assert sum(a=='Bearer secret-a' for a,_ in calls)==2
        assert not any(entry['disabled'] for entry in client.stats)


def test_all_service_errors_are_bounded_and_preserve_the_actual_http_status(tmp_path):
    client,calls=pool(tmp_path,lambda _:529)
    with client:
        assert client.post(ENDPOINT,json={'sample':1}).status_code==529
        assert len(calls)==3 and all(body=={'sample':1} for _,body in calls)
        assert not any(entry['disabled'] for entry in client.stats)


def test_credentials_are_never_sent_to_a_different_host(tmp_path):
    client,calls=pool(tmp_path,lambda _:200)
    with client,pytest.raises(ValueError,match='official'):
        client.post('https://example.com',json={})
    assert calls==[]
