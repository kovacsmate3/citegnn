"""Backwards-compatible alias for the HeaRT split builder CLI.

The legacy random-positive/random-negative pipeline has been replaced
with the HeaRT protocol; this script now forwards to
``citegnn.data.smoke.main`` which produces a HeaRT split artifact.
"""

from __future__ import annotations

from citegnn.data.smoke import main


if __name__ == "__main__":
    main()
