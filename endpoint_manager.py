#!/usr/bin/env python3
"""Shared SAM SageMaker endpoint lifecycle manager.

This module extracts the reusable AWS lifecycle logic that used to live inside
``endpoint_ctl.py`` (stack-output discovery, create/delete/status,
the managed-tag guardrail, and orphan detection) into a single importable
``EndpointManager`` class. Both the GUI (``labeler.py``) and the CLI
(``endpoint_ctl.py``) call this module so lifecycle behavior is defined once.

All AWS work goes through a caller-injected boto3 session
(``session.client("sagemaker")`` / ``("s3")`` / ``("cloudformation")``), which
keeps the class unit-testable with a mocked session.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

DEFAULT_STACK = "SamLabelerStack"
DEFAULT_ENDPOINT = "sam-labeler"

# Set on endpoints we create, and checked before deleting so that a name
# collision can never take down an endpoint this tool did not start.
MANAGED_TAG = {"Key": "ManagedBy", "Value": "sam-labeler"}

# States in which SageMaker is running (or provisioning) an instance, and so
# the endpoint is billable.
_BILLABLE_STATES = frozenset(
    {"InService", "Creating", "Updating", "SystemUpdating"}
)

# Rough on-demand rates for the sizes HARDWARE.md considered, only to make the
# cost of leaving an endpoint running visible. Verify current pricing.
APPROX_HOURLY_USD = {
    "ml.g4dn.xlarge": 0.74,
    "ml.g4dn.2xlarge": 0.94,
    "ml.g5.xlarge": 1.41,
    "ml.g6.xlarge": 0.98,
}


class EndpointOrigin(Enum):
    """Who created an endpoint, tracked in-process by the GUI session."""

    APP = "app"  # created by the GUI in this session
    CLI = "cli"  # created by start-sam-endpoint
    EXTERNAL = "external"  # pre-existing / unknown


@dataclass
class ResolvedNames:
    """Names/URIs resolved from CloudFormation stack outputs (with fallback)."""

    endpoint: str
    endpoint_config: str | None
    instance_type: str | None


@dataclass
class EndpointStatus:
    """A point-in-time view of an endpoint's existence, state, and billability."""

    name: str
    exists: bool
    status: str | None  # Creating | InService | Failed | Deleting | None
    billable: bool  # True when an instance is (or is becoming) live
    instance_type: str | None
    failure_reason: str | None
    managed: bool  # carries MANAGED_TAG


class GuardrailError(Exception):
    """Raised by delete() when asked to delete an untagged endpoint without force."""


class EndpointManager:
    """Create, delete, inspect, and reap SAM SageMaker endpoints.

    All AWS calls route through the injected boto3 ``session`` so the class can
    be exercised with a mocked session in unit tests.
    """

    def __init__(
        self,
        session,
        stack: str = DEFAULT_STACK,
        endpoint: str | None = None,
        endpoint_config: str | None = None,
        no_stack_lookup: bool = False,
    ) -> None:
        self._session = session
        self._stack = stack
        self._endpoint_override = endpoint
        self._endpoint_config_override = endpoint_config
        self._no_stack_lookup = no_stack_lookup
        # Cached resolution so repeated calls do not re-hit CloudFormation.
        self._resolved: ResolvedNames | None = None

    # -- discovery -------------------------------------------------------

    def _stack_outputs(self) -> dict[str, str]:
        """Read the CDK stack's outputs, or {} if the stack is not reachable."""
        cfn = self._session.client("cloudformation")
        try:
            stacks = cfn.describe_stacks(StackName=self._stack)["Stacks"]
        except Exception:
            # The stack may not be deployed, or we may lack cfn permissions;
            # fall back to explicit overrides / defaults.
            return {}
        if not stacks:
            return {}
        return {
            o["OutputKey"]: o["OutputValue"]
            for o in stacks[0].get("Outputs", [])
        }

    def resolve(self) -> ResolvedNames:
        """Resolve endpoint/config/model names from stack outputs with fallback.

        Explicit constructor overrides win over stack outputs, which win over
        built-in defaults. The endpoint name always resolves to a non-None
        value; the others may be None when neither an override nor a stack
        output supplies them.
        """
        if self._resolved is not None:
            return self._resolved

        outputs = {} if self._no_stack_lookup else self._stack_outputs()
        endpoint = (
            self._endpoint_override
            or outputs.get("EndpointName")
            or DEFAULT_ENDPOINT
        )
        endpoint_config = (
            self._endpoint_config_override or outputs.get("EndpointConfigName")
        )
        instance_type = outputs.get("InstanceType")

        self._resolved = ResolvedNames(
            endpoint=endpoint,
            endpoint_config=endpoint_config,
            instance_type=instance_type,
        )
        return self._resolved

    # -- describe / status ----------------------------------------------

    def _describe(self, sm, endpoint: str) -> dict | None:
        """Describe the endpoint, or None if it does not exist."""
        from botocore.exceptions import ClientError

        try:
            return sm.describe_endpoint(EndpointName=endpoint)
        except ClientError as e:
            code = e.response["Error"]["Code"]
            # SageMaker reports a missing endpoint as ValidationException.
            if code in ("ValidationException", "ResourceNotFound"):
                return None
            raise

    def _is_managed(self, sm, endpoint_arn: str) -> bool:
        """Return True if the endpoint carries the MANAGED_TAG."""
        try:
            tags = sm.list_tags(ResourceArn=endpoint_arn).get("Tags", [])
        except Exception:
            tags = []
        return MANAGED_TAG in tags

    def status(self) -> EndpointStatus:
        """Describe the endpoint and report existence, state, and billability."""
        names = self.resolve()
        endpoint = names.endpoint
        sm = self._session.client("sagemaker")

        desc = self._describe(sm, endpoint)
        if desc is None:
            return EndpointStatus(
                name=endpoint,
                exists=False,
                status=None,
                billable=False,
                instance_type=names.instance_type,
                failure_reason=None,
                managed=False,
            )

        state = desc.get("EndpointStatus")
        managed = self._is_managed(sm, desc.get("EndpointArn", ""))
        return EndpointStatus(
            name=endpoint,
            exists=True,
            status=state,
            billable=state in _BILLABLE_STATES,
            instance_type=names.instance_type,
            failure_reason=desc.get("FailureReason"),
            managed=managed,
        )

    # -- create ----------------------------------------------------------

    def create(self, wait: bool = True, timeout: int = 1800) -> EndpointStatus:
        """Create the endpoint (tagged MANAGED_TAG), optionally waiting.

        If the endpoint already exists, mirrors ``cmd_up``: InService is a
        no-op, Creating/Updating optionally waits, Failed raises RuntimeError.
        """
        names = self.resolve()
        endpoint = names.endpoint
        sm = self._session.client("sagemaker")

        existing = self._describe(sm, endpoint)
        if existing:
            state = existing.get("EndpointStatus")
            if state == "InService":
                return self.status()
            if state in ("Creating", "Updating", "SystemUpdating"):
                if wait:
                    self._wait_in_service(sm, endpoint, timeout)
                return self.status()
            if state == "Failed":
                reason = existing.get("FailureReason", "unknown")
                raise RuntimeError(
                    f"{endpoint} is in Failed state: {reason}. "
                    f"Delete it first, then retry."
                )
            raise RuntimeError(f"{endpoint} is {state}; refusing to act.")

        if not names.endpoint_config:
            raise RuntimeError(
                "Could not determine the EndpointConfig name. Deploy the CDK "
                f"stack (default name {self._stack!r}) or pass endpoint_config."
            )

        sm.create_endpoint(
            EndpointName=endpoint,
            EndpointConfigName=names.endpoint_config,
            Tags=[MANAGED_TAG],
        )

        if wait:
            self._wait_in_service(sm, endpoint, timeout)
        return self.status()

    def _wait_in_service(self, sm, endpoint: str, timeout: int) -> None:
        """Block until the endpoint reaches InService; raise on timeout/failure."""
        try:
            sm.get_waiter("endpoint_in_service").wait(
                EndpointName=endpoint,
                WaiterConfig={"Delay": 15, "MaxAttempts": max(1, timeout // 15)},
            )
        except Exception as e:
            desc = self._describe(sm, endpoint) or {}
            reason = desc.get("FailureReason", "")
            msg = f"Endpoint {endpoint} did not reach InService: {e}"
            if reason:
                msg += f" (FailureReason: {reason})"
            raise RuntimeError(msg) from e

    # -- delete ----------------------------------------------------------

    def delete(self, force: bool = False, wait: bool = True) -> None:
        """Delete the endpoint if managed (or forced); no-op if absent.

        Raises GuardrailError when the endpoint exists, is not tagged with
        MANAGED_TAG, and ``force`` is False.
        """
        names = self.resolve()
        endpoint = names.endpoint
        sm = self._session.client("sagemaker")

        desc = self._describe(sm, endpoint)
        if desc is None:
            return  # nothing to delete

        if not force and not self._is_managed(sm, desc.get("EndpointArn", "")):
            raise GuardrailError(
                f"{endpoint} is not tagged {MANAGED_TAG['Key']}="
                f"{MANAGED_TAG['Value']}, so it was not created by this tool. "
                f"Refusing to delete it. Pass force=True if you are sure."
            )

        sm.delete_endpoint(EndpointName=endpoint)

        if wait:
            try:
                sm.get_waiter("endpoint_deleted").wait(
                    EndpointName=endpoint,
                    WaiterConfig={"Delay": 10, "MaxAttempts": 60},
                )
            except Exception:
                # Delete may still be in progress; treat as best-effort.
                pass

    # -- orphan detection ------------------------------------------------

    def find_orphans(self, owned: set[str]) -> list[EndpointStatus]:
        """Return managed endpoints not created by this session.

        Lists all endpoints, and for each checks whether it carries the
        MANAGED_TAG. Returns those managed endpoints whose name is not in
        ``owned``.
        """
        sm = self._session.client("sagemaker")

        endpoints: list[dict] = []
        paginator_supported = hasattr(sm, "get_paginator")
        if paginator_supported:
            try:
                paginator = sm.get_paginator("list_endpoints")
                for page in paginator.paginate():
                    endpoints.extend(page.get("Endpoints", []))
            except Exception:
                paginator_supported = False
        if not paginator_supported:
            # Fall back to a single list_endpoints call (mock-friendly).
            resp = sm.list_endpoints()
            endpoints.extend(resp.get("Endpoints", []))

        orphans: list[EndpointStatus] = []
        for ep in endpoints:
            name = ep.get("EndpointName")
            if name is None or name in owned:
                continue
            arn = ep.get("EndpointArn", "")
            if not self._is_managed(sm, arn):
                continue
            state = ep.get("EndpointStatus")
            orphans.append(
                EndpointStatus(
                    name=name,
                    exists=True,
                    status=state,
                    billable=state in _BILLABLE_STATES,
                    instance_type=None,
                    failure_reason=ep.get("FailureReason"),
                    managed=True,
                )
            )
        return orphans
