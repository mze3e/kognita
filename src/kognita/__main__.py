"""``python -m kognita`` runs the same command line as the ``kognita`` script."""

from kognita.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
