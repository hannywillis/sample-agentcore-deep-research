"""Administration CLI.

    python -m agri_advisor.accounts create-admin --email you@kalro.org --password ...
    python -m agri_advisor.accounts create-org --name KALRO --slug kalro --domain kalro.org
    python -m agri_advisor.accounts org-admin --org kalro --email you@kalro.org
    python -m agri_advisor.accounts set-role --email x@kalro.org --role consortium
"""

from __future__ import annotations

import argparse
import getpass
import json

from agri_advisor.accounts.store import PLATFORM_ROLES, AccountStore


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="agri_advisor.accounts")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("create-admin")
    a.add_argument("--email", required=True)
    a.add_argument("--password")
    a.add_argument("--name")
    o = sub.add_parser("create-org")
    o.add_argument("--name", required=True)
    o.add_argument("--slug", required=True)
    o.add_argument("--domain")
    oa = sub.add_parser("org-admin")
    oa.add_argument("--org", required=True)
    oa.add_argument("--email", required=True)
    r = sub.add_parser("set-role")
    r.add_argument("--email", required=True)
    r.add_argument("--role", required=True, choices=PLATFORM_ROLES)
    args = ap.parse_args(argv)

    store = AccountStore()
    if args.cmd == "create-admin":
        pw = args.password or getpass.getpass("Password: ")
        out = store.register(args.email, pw, args.name, role="admin").to_dict()
    elif args.cmd == "create-org":
        out = store.create_org(args.name, args.slug, args.domain)
    elif args.cmd == "org-admin":
        out = store.set_org_admin(args.org, args.email)
    else:
        store.set_role(args.email, args.role)
        out = {"email": args.email, "role": args.role}
    print(json.dumps(out, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
