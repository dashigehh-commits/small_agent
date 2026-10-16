import asyncio
import pytest
from agent import Agent

class FakeLLM:
    def __init__(self, loop=False): self.calls=0; self.loop=loop
    async def complete(self, messages, tools):
        self.calls+=1
        if self.loop or self.calls==1:
            return {'role':'assistant','tool_calls':[{'id':'c','function':{'name':'calculator','arguments':'{"expression":"2+3*4"}'}}]}
        return {'role':'assistant','content':'结果是 14'}

def run(coro): return asyncio.run(coro)

def test_registry_and_schema():
    a=Agent(FakeLLM()); assert {'calculator','search','todo','weather'} <= set(a.registry.tools)
    assert a.registry.tools['calculator'].schema()['function']['parameters']['required']==['expression']

def test_loop_and_trace():
    r=run(Agent(FakeLLM()).run('计算','s1')); assert r['answer']=='结果是 14'; assert r['trace'][0]['result']==14

def test_session_isolation():
    a=Agent(FakeLLM()); run(a.run('窗口一','w1')); run(a.run('窗口二','w2'))
    assert a.sessions['w1'].messages[0]['content']=='窗口一'; assert a.sessions['w2'].messages[0]['content']=='窗口二'

def test_compaction():
    a=Agent(FakeLLM(),max_messages=4); s=a.session('x'); s.messages=[{'role':'user','content':str(i)} for i in range(10)]; a._compact(s)
    assert len(s.messages)==4 and 'summarized' in s.messages[1]['content']

def test_max_rounds():
    with pytest.raises(RuntimeError,match='maximum agent rounds'): run(Agent(FakeLLM(loop=True),max_rounds=2).run('循环','x'))

def test_unknown_tool():
    with pytest.raises(ValueError,match='unknown tool'): Agent(FakeLLM()).registry.call('missing',{})
