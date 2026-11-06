"""Allow ``python -m signstream.cli`` execution."""

from signstream.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
