#!/usr/bin/env python3
"""CDK entry point for the SAM labeling endpoint.

Deploy with defaults (ml.g4dn.xlarge, a new weights bucket):

    cdk deploy

Override anything via context:

    cdk deploy \
      -c instanceType=ml.g5.xlarge \
      -c endpointName=sam-labeler \
      -c dtype=fp16

Account and region come from the ambient AWS credentials (CDK_DEFAULT_*),
so `AWS_PROFILE` / `AWS_REGION` select where this lands.
"""
import os

import aws_cdk as cdk

from sam_labeler_stack import (
    DEFAULT_ENDPOINT_NAME,
    DEFAULT_INSTANCE_TYPE,
    SamLabelerStack,
)

app = cdk.App()


def ctx(key: str, default=None):
    value = app.node.try_get_context(key)
    return default if value is None else value


def as_bool(v, default=False):
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "on")


SamLabelerStack(
    app,
    ctx("stackName", "SamLabelerStack"),
    instance_type=ctx("instanceType", DEFAULT_INSTANCE_TYPE),
    endpoint_name=ctx("endpointName", DEFAULT_ENDPOINT_NAME),
    dtype=ctx("dtype"),
    build_local=as_bool(ctx("buildLocal"), default=False),
    network_isolation=as_bool(ctx("networkIsolation"), default=True),
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION"),
    ),
    description="SAM auto-labeling inference endpoint (on-demand) (SO0361)",
)

app.synth()
