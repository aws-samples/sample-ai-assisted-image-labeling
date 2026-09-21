#!/usr/bin/env python3
"""Start and stop the SAM SageMaker endpoint on demand.

The CDK stack deploys everything except the endpoint itself, because the
endpoint is the only per-hour cost. Bring it up when you sit down to label and
take it down when you are done:

    python endpoint_ctl.py up          # ~7-10 min, then start labeling
    python endpoint_ctl.py status
    python endpoint_ctl.py down        # back to $0/hr

Or, once installed as a UV project, via the console entry points:

    uv run start-sam-endpoint          # -> endpoint_ctl:start
    uv run stop-sam-endpoint           # -> endpoint_ctl:stop

Names are discovered from the CloudFormation stack outputs, so a redeployed
stack (new image, new EndpointConfig) needs no flag changes here.

This module is a thin CLI over ``endpoint_manager.EndpointManager``: it owns the
argparse UI and the friendly AWS error handling, but all AWS lifecycle work
(stack-output discovery, create/delete/status, the managed-tag
guardrail) is delegated to that shared module so behavior is defined once and
matches the GUI.

Requires boto3 and credentials with the stack's operator policy attached
(see the OperatorPolicyArn stack output).
"""
from __future__ import annotations

import argparse
import sys

from endpoint_manager import (
    APPROX_HOURLY_USD,
    DEFAULT_ENDPOINT,
    DEFAULT_STACK,
    EndpointManager,
    GuardrailError,
    MANAGED_TAG,
)


def _session(args):
    """Build a boto3 Session from the --profile / --region flags."""
    try:
        import boto3
    except ImportError:
        sys.exit(
            "boto3 is required.\n"
            "    uv pip install boto3      (or: pip install boto3)"
        )
    return boto3.Session(profile_name=args.profile, region_name=args.region)


def _manager(args) -> EndpointManager:
    """Construct an EndpointManager wired to a session built from the flags."""
    return EndpointManager(
        _session(args),
        stack=args.stack,
        endpoint=args.endpoint,
        endpoint_config=getattr(args, "endpoint_config", None),
        no_stack_lookup=getattr(args, "no_stack_lookup", False),
    )


def cmd_up(args) -> int:
    """Create (or wait on) the endpoint via EndpointManager.create()."""
    manager = _manager(args)
    names = manager.resolve()
    endpoint = names.endpoint

    instance = names.instance_type or ""
    rate = APPROX_HOURLY_USD.get(instance)
    cost = f" (~${rate:.2f}/hr while running)" if rate else ""
    print(f"Bringing up endpoint {endpoint}{cost}...")
    if args.wait:
        print("Waiting for InService (first start is typically 7-10 min)...")

    status = manager.create(wait=args.wait, timeout=args.timeout)

    if not args.wait:
        print("Creation started. Check progress:  python endpoint_ctl.py status")
        return 0

    if status.status == "InService":
        print(f"{endpoint} is InService. Ready to label.")
    else:
        print(f"{endpoint} is {status.status}.")
    return 0


def cmd_down(args) -> int:
    """Delete the endpoint via EndpointManager.delete() (guarded by tag/force)."""
    manager = _manager(args)
    endpoint = manager.resolve().endpoint

    print(f"Deleting endpoint {endpoint}...")
    try:
        manager.delete(force=args.force, wait=args.wait)
    except GuardrailError as e:
        print(f"{e}", file=sys.stderr)
        return 1
    print("Deleted (or already absent). Hourly cost is now $0.")
    return 0


def cmd_status(args) -> int:
    """Report endpoint existence, state, config, and billability."""
    manager = _manager(args)
    names = manager.resolve()
    status = manager.status()

    print(f"endpoint         {status.name}")
    print(f"status           {status.status if status.exists else 'DOES NOT EXIST'}")
    if status.instance_type:
        print(f"instance         {status.instance_type}")
    if names.endpoint_config:
        print(f"endpoint config  {names.endpoint_config}")
    if status.exists and status.failure_reason:
        print(f"failure reason   {status.failure_reason}")

    # Req 13.3: always report whether a billable endpoint is running.
    print(f"billable         {'yes' if status.billable else 'no'}")

    if not status.exists:
        print("\nNot running (no charges). Start it with:  python endpoint_ctl.py up")
    elif status.billable:
        rate = APPROX_HOURLY_USD.get(status.instance_type or "")
        cost = f" (~${rate:.2f}/hr)" if rate else ""
        print(
            f"\nRunning and billable{cost}. "
            f"Stop it with:  python endpoint_ctl.py down"
        )
    return 0


def _common_parser() -> argparse.ArgumentParser:
    """The shared flags used by every subcommand and by start()/stop()."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--stack", default=DEFAULT_STACK,
                        help=f"CDK stack to read names from (default: {DEFAULT_STACK})")
    common.add_argument("--endpoint", help="Endpoint name (default: from stack output)")
    common.add_argument("--endpoint-config",
                        help="EndpointConfig name (default: from stack output)")
    common.add_argument("--region", help="AWS region (default: from environment)")
    common.add_argument("--profile", help="AWS profile (default: from environment)")
    common.add_argument("--no-stack-lookup", action="store_true",
                        help="Skip CloudFormation; use only the flags given")
    common.add_argument("--timeout", type=int, default=1800,
                        help="Seconds to wait for InService (default: 1800)")
    return common


def _run(func) -> int:
    """Invoke a cmd_* function, translating boto3 setup errors into friendly text.

    Centralizes the NoRegionError/NoCredentialsError/ProfileNotFound/ClientError
    handling so both the argparse ``main()`` path and the ``start()``/``stop()``
    entry points share identical error UX.
    """
    from botocore.exceptions import (
        ClientError,
        NoCredentialsError,
        NoRegionError,
        ProfileNotFound,
    )

    try:
        return func()
    except KeyboardInterrupt:
        print("\nInterrupted. The AWS-side operation may still be in progress.",
              file=sys.stderr)
        return 130
    except NoRegionError:
        print(
            "No AWS region configured. Set one:\n"
            "    export AWS_REGION=us-west-2      (or pass --region)",
            file=sys.stderr,
        )
        return 2
    except (NoCredentialsError, ProfileNotFound) as e:
        print(f"AWS credentials problem: {e}", file=sys.stderr)
        return 2
    except ClientError as e:
        # AccessDenied is the common first-run failure for a new user, so name
        # the fix instead of printing a traceback.
        code = e.response["Error"]["Code"]
        print(f"AWS call failed ({code}): {e.response['Error'].get('Message', e)}",
              file=sys.stderr)
        if code in ("AccessDenied", "AccessDeniedException", "UnauthorizedOperation"):
            print(
                "\nYour credentials lack the needed permissions. Attach the "
                "stack's operator policy\n(see the OperatorPolicyArn output of "
                "the CDK stack) to this user or role.",
                file=sys.stderr,
            )
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Start/stop the SAM SageMaker endpoint on demand.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Shared options live on the subcommands, not the top-level parser, so
    # that `endpoint_ctl.py up --endpoint foo` works. Putting them in both
    # places would let the subparser's default silently overwrite a value
    # given before the subcommand.
    common = _common_parser()

    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("up", parents=[common],
                        help="Create the endpoint (starts billing)")
    up.add_argument("--no-wait", dest="wait", action="store_false",
                    help="Return immediately instead of waiting for InService")
    up.set_defaults(func=cmd_up, wait=True)

    down = sub.add_parser("down", parents=[common],
                          help="Delete the endpoint (stops billing)")
    down.add_argument("--no-wait", dest="wait", action="store_false",
                      help="Return without waiting for deletion to finish")
    down.add_argument("--force", action="store_true",
                      help="Delete even if not tagged as created by this tool")
    down.set_defaults(func=cmd_down, wait=True, force=False)

    status = sub.add_parser("status", parents=[common],
                            help="Show endpoint status and cost state")
    status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    return _run(lambda: args.func(args))


# -- console entry points ------------------------------------------------
#
# `start-sam-endpoint` / `stop-sam-endpoint` in pyproject.toml map to these.
# They accept the same optional flags as the `up`/`down` subcommands (minus the
# subcommand word) so `uv run start-sam-endpoint --region us-west-2` works.
#
# A CLI-created endpoint is NOT marked app-owned: create() only applies
# MANAGED_TAG, and no extra ownership marker is written. The app-vs-CLI
# distinction (which decides whether the GUI tears an endpoint down on close)
# is tracked in-process by the GUI's session-owned set, per design; it is not
# persisted on the endpoint. So an endpoint started here survives GUI close.


def start(argv: list[str] | None = None) -> int:
    """Console entry point for `start-sam-endpoint`: create the endpoint."""
    parser = argparse.ArgumentParser(
        prog="start-sam-endpoint",
        parents=[_common_parser()],
        description="Create the SAM SageMaker endpoint (starts billing).",
    )
    parser.add_argument("--no-wait", dest="wait", action="store_false",
                        help="Return immediately instead of waiting for InService")
    parser.set_defaults(wait=True)
    args = parser.parse_args(argv)
    return _run(lambda: cmd_up(args))


def stop(argv: list[str] | None = None) -> int:
    """Console entry point for `stop-sam-endpoint`: delete the endpoint."""
    parser = argparse.ArgumentParser(
        prog="stop-sam-endpoint",
        parents=[_common_parser()],
        description="Delete the SAM SageMaker endpoint (stops billing).",
    )
    parser.add_argument("--no-wait", dest="wait", action="store_false",
                        help="Return without waiting for deletion to finish")
    parser.add_argument("--force", action="store_true",
                        help="Delete even if not tagged as created by this tool")
    parser.set_defaults(wait=True, force=False)
    args = parser.parse_args(argv)
    return _run(lambda: cmd_down(args))


if __name__ == "__main__":
    sys.exit(main())
