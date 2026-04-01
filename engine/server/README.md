
# MCP Server 실행 가이드

WhyLab MCP 서버는 표준 입출력(stdio)을 통해 통신합니다.
외부 에이전트(Claude Desktop 등) 설정 파일에 다음을 추가하세요.

```json
{
  "mcpServers": {
    "whylab": {
      "command": "python",
      "args": ["-m", "engine.server.mcp_server"],
      "env": {
        "PYTHONPATH": "D:\\00.test\\PAPER\\WhyLab"
      }
    }
  }
}
```

## 서버 테스트 방법
터미널에서 직접 실행하여 JSON-RPC 통신이 되는지 불가능하지만,
에러 없이 실행되는지는 확인할 수 있습니다.

```bash
python -m engine.server.mcp_server
```
(실행 후 아무 반응이 없으면 정상 - 입력을 기다리는 중)
