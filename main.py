"""
Copyright (c) VKU.NewEnergy.

This source code is licensed under the Apache-2.0 license found in the
LICENSE file in the root directory of this source tree.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=9000, reload=True)
