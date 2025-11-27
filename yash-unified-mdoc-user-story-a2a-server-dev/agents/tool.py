import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from typing import cast
from langchain_mcp_adapters.sessions import Connection, StdioConnection

from dotenv import load_dotenv

load_dotenv()

# Coder tool
obligation_configs = {
    "obligation": {
        "command": "python",
        "args" : [
            "tools-data/user_story/mcp_server.py",
        ],
        "env": {
            "BACKEND_URL": os.getenv("BACKEND_URL"),
        },
        "transport": "stdio",
    }
}
mdoc_tools_client = MultiServerMCPClient(obligation_configs)
