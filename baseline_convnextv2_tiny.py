"""Train or evaluate the convnextv2_tiny comparison encoder with shared fusion heads."""
from __future__ import annotations

import sys
from stage1_multitask import main


if __name__ == "__main__":
    if "--backbone" in sys.argv:
        raise SystemExit("This entry point fixes the comparison backbone.")
    sys.argv.extend(["--backbone", "convnextv2_tiny"])
    main()
