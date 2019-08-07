# Functionally equivalent to:
# jq -r '.["pytorch"][-1].version' | sed 's/+.*$//'

import sys
import json
import re

print(re.sub(r'\+.*$', '', json.load(sys.stdin)["pytorch"][-1]["version"]))
