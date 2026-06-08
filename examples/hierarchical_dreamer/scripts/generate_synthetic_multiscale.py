import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.hierarchical_dreamer.synthetic_multiscale.generate_dataset import main


if __name__ == "__main__":
    main()
