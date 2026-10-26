"""Подмена только для испытаний. Не входит ни в образ, ни в рабочие профили."""

import json

import httpx

ANSWER = "<table><tr><th>Код</th><th>Сумма</th></tr><tr><td>0012</td><td>−0,25</td></tr></table>"


class FakeCompute:
    def __init__(self, fail_first=False):
        self.operations = {}
        self.fail_first = fail_first

    async def __call__(self, request):
        path = request.url.path
        if path in {"/status", "/health"}:
            return httpx.Response(
                200, json={"state": "ready", "model_loaded": True, "runtime": "test-fixture"}
            )
        if path == "/slots":
            return httpx.Response(200, json=[{"is_processing": False}])
        if path.startswith("/operations/"):
            identifier = path.split("/")[2]
            if request.method == "PUT":
                self.operations.setdefault(identifier, json.loads(request.content))
                return httpx.Response(202, json={"id": identifier})
            if path.endswith("/cancel"):
                return httpx.Response(200, json={"cancel_requested": True})
            failed = self.fail_first and identifier == next(iter(self.operations))
            return httpx.Response(
                200,
                json={
                    "state": "completed",
                    "result": {
                        "text": "" if failed else f"![table](20,20,980,980)\n{ANSWER}",
                        "error": "Ошибка одной страницы" if failed else None,
                        "metadata": {"runtime": "test-fixture"},
                    },
                },
            )
        if path == "/v1/chat/completions":
            chunk = {"choices": [{"delta": {"content": ANSWER}, "finish_reason": "stop"}]}
            return httpx.Response(
                200,
                content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(404)
