"""Rotate Jev credentials while preserving the sealed evaluator and predictions."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import httpx

from evaluation.det import jev

ROUTER_SHA256 = jev.digest(Path(__file__))


class RotatingClient:
    """Bounded round robin for authentication, quota and transient service errors."""
    def __init__(self, credentials, status_path, client_factory=httpx.Client, **kwargs):
        if not credentials or len({value for _,value in credentials})!=len(credentials):
            raise ValueError('Provide distinct API credentials')
        self.status_path=Path(status_path)
        self.names=[name for name,_ in credentials]
        self.clients=[]
        for _,key in credentials:
            headers=dict(kwargs.get('headers',{}),Authorization='Bearer '+key)
            self.clients.append(client_factory(**dict(kwargs,headers=headers)))
        self.stats=[dict(variable=name,requests=0,successes=0,disabled=False,cooldown_until=0.)
                    for name in self.names]
        self.position=0

    def __enter__(self):
        for client in self.clients:client.__enter__()
        return self

    def __exit__(self,*args):
        for client in self.clients:client.__exit__(*args)
        self.save()

    def save(self):
        self.status_path.parent.mkdir(parents=True,exist_ok=True)
        jev.write_json(self.status_path,dict(policy='sequential_round_robin',credentials=self.stats,
            credential_values_saved=False,credential_router_sha256=ROUTER_SHA256))

    def post(self,url,**kwargs):
        if url!=jev.ENDPOINT:
            raise ValueError('Credentials are restricted to the official Jev endpoint')
        last=None
        for _ in range(len(self.clients)):
            index=self.position
            self.position=(index+1)%len(self.clients)
            entry=self.stats[index]
            if entry['disabled'] or entry['cooldown_until']>time.time():continue
            entry['requests']+=1
            try:
                response=self.clients[index].post(url,**kwargs)
            except httpx.TransportError:
                entry['last_status']='transport_error'
                self.save()
                raise
            entry['last_status']=response.status_code
            if response.status_code==200:entry['successes']+=1
            elif response.status_code in (401,402,403):entry['disabled']=True
            elif response.status_code==429:
                try:delay=float(response.headers.get('Retry-After','30'))
                except ValueError:delay=30
                entry['cooldown_until']=time.time()+min(300,max(1,delay))
            self.save()
            retryable=response.status_code in (401,402,403,429) or 500<=response.status_code<600
            if not retryable:return response
            last=response
        if all(entry['disabled'] for entry in self.stats):
            raise RuntimeError('All configured Jev credentials rejected authentication or quota')
        # The evaluator's bounded retry/backoff handles an all-key rate limit.
        return last if last is not None else httpx.Response(429)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--credential-env-file',type=Path,required=True)
    parser.add_argument('--key-envs',nargs='+',required=True)
    parser.add_argument('--credential-status',type=Path,required=True)
    args,remaining=parser.parse_known_args()
    credentials=[(name,jev.load_key(name,args.credential_env_file)) for name in args.key_envs]
    factory=httpx.Client
    def rotating(**kwargs):
        return RotatingClient(credentials,args.credential_status,client_factory=factory,**kwargs)
    # Keep jev.py byte-for-byte identical: credential routing does not change
    # requests, preprocessing, model version, labels or benchmark score rules.
    jev.httpx=SimpleNamespace(Client=rotating,Timeout=httpx.Timeout,TransportError=httpx.TransportError)
    sys.argv=[sys.argv[0],*remaining,'--env-file',str(args.credential_env_file),'--api-key-env',args.key_envs[0]]
    jev.main()


if __name__=='__main__':
    main()
