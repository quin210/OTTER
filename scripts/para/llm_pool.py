"""Global 3-slot semaphore for every LLM request made by the Para pipeline (servers are shared)."""
import threading, time
from para_lib import ROOT
_SEM = threading.BoundedSemaphore(3)
_llm = None
def chat(messages, temperature=0.0, max_tokens=1200, retries=3):
    global _llm
    if _llm is None:
        from otter.llm.interface import get_llm
        _llm = get_llm("main")
    for a in range(retries):
        try:
            with _SEM:
                return _llm.chat(messages, temperature=temperature, max_tokens=max_tokens)
        except Exception as e:  # noqa
            err = e; time.sleep(5 * (a + 1))
    raise err
