"""Container health check: GET /livez on the local server. Standard library only (no curl in the image).

Sends the origin secret when one is set, in case the edge check covers /livez too.
"""

import os
import sys
import urllib.request

port = os.environ.get("KINDLING_PORT", "8000")
request = urllib.request.Request(f"http://127.0.0.1:{port}/livez")
secret = os.environ.get("KINDLING_ORIGIN_SECRET", "")
if secret:
    request.add_header(os.environ.get("KINDLING_ORIGIN_SECRET_HEADER", "X-Kindling-Origin-Secret"), secret)
try:
    with urllib.request.urlopen(request, timeout=4) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception as error:
    print(f"unhealthy: {error}", file=sys.stderr)
    sys.exit(1)
