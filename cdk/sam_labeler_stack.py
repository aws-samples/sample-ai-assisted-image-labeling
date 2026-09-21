"""CDK stack for the SAM labeling endpoint.

Deliberately creates everything needed to *start* a SageMaker endpoint, but
not the endpoint itself:

    ECR image        built from sam_server/Dockerfile, linux/amd64; the
                     (Apache-2.0, public) model weights are baked in at build
    IAM role         SageMaker execution role: pull image, write logs
    SageMaker Model  the container image; no S3 model data — weights ship
                     inside the image
    EndpointConfig   instance type and startup timeouts
    IAM policy       attach to other users so they can run the endpoint

The Endpoint is left out on purpose: an endpoint is the only resource here
that costs money per hour, so it is created and deleted on demand by
`endpoint_ctl.py`, which discovers the config name from this stack's outputs.
That keeps the steady-state cost of a deployed stack at $0 (ECR storage aside).
"""
from __future__ import annotations

from pathlib import Path

from aws_cdk import (
    ArnFormat,
    CfnOutput,
    CustomResource,
    Duration,
    RemovalPolicy,
    Stack,
    aws_codebuild as codebuild,
    aws_ecr as ecr,
    aws_ecr_assets as ecr_assets,
    aws_iam as iam,
    aws_lambda as _lambda,
    aws_logs as logs,
    aws_s3 as s3,
    aws_s3_assets as s3_assets,
    aws_sagemaker as sagemaker,
    custom_resources as cr,
)
from constructs import Construct

# Inline source for the custom-resource Lambda handlers. Kept as `from_inline`
# strings so `cdk synth` never needs Docker to bundle them (CI parity with the
# existing offline synth test suite).
#
# BuildId plumbing: the Provider framework passes whatever `on_event` returns
# under "Data" into `is_complete`'s event["Data"]. So on_event returns the
# CodeBuild build id in Data.BuildId, and is_complete reads it from there. We
# ALSO set PhysicalResourceId to the ImageTag (the content-hash idempotency
# key) rather than the build id: this makes an unchanged tag a plain no-op
# update (no physical-id change => no replacement) while a changed source
# (=> changed tag => changed PhysicalResourceId) forces the resource to update
# and rebuild. Using the build id as the PhysicalResourceId would instead make
# every deploy look like a replacement, defeating the no-op-on-unchanged-source
# requirement.
ON_EVENT_SRC = '''\
import boto3

codebuild = boto3.client("codebuild")


def on_event(event, context):
    request_type = event["RequestType"]
    props = event.get("ResourceProperties", {})
    if request_type == "Delete":
        # Nothing to undo: image cleanup happens via ECR empty_on_delete when
        # the repository is deleted. Never start a build on Delete.
        return {"PhysicalResourceId": event.get("PhysicalResourceId", "deleted")}

    resp = codebuild.start_build(projectName=props["ProjectName"])
    build_id = resp["build"]["id"]
    # PhysicalResourceId = ImageTag (idempotency key); BuildId returned in Data
    # so is_complete can poll the correct build.
    return {
        "PhysicalResourceId": props["ImageTag"],
        "Data": {"BuildId": build_id},
    }
'''

IS_COMPLETE_SRC = '''\
import boto3

codebuild = boto3.client("codebuild")

_TERMINAL_FAILURES = ("FAILED", "FAULT", "STOPPED", "TIMED_OUT")


def is_complete(event, context):
    if event["RequestType"] == "Delete":
        return {"IsComplete": True}

    build_id = event.get("Data", {}).get("BuildId")
    if not build_id:
        # No build id available (e.g. nothing was started): not complete yet.
        return {"IsComplete": False}

    builds = codebuild.batch_get_builds(ids=[build_id]).get("builds", [])
    if not builds:
        return {"IsComplete": False}

    status = builds[0]["buildStatus"]
    if status == "SUCCEEDED":
        return {"IsComplete": True}
    if status == "IN_PROGRESS":
        return {"IsComplete": False}
    if status in _TERMINAL_FAILURES:
        raise Exception(
            f"CodeBuild build {build_id} ended with status {status}"
        )
    # Unknown/other transient status: keep polling.
    return {"IsComplete": False}
'''

# HARDWARE.md benchmarked T4/fp16 at ~0.5 s per interactive prompt for about a
# third of the cost of the Ampere options, making g4dn.xlarge the default pick.
DEFAULT_INSTANCE_TYPE = "ml.g4dn.xlarge"
DEFAULT_ENDPOINT_NAME = "sam-labeler"

# Anchored to this file, not the cwd: CDK resolves asset directories relative
# to the process working directory, so a relative literal would break whenever
# the app is invoked from anywhere but cdk/.
SAM_SERVER_DIR = str((Path(__file__).resolve().parent.parent / "sam_server"))


class SamLabelerStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        instance_type: str = DEFAULT_INSTANCE_TYPE,
        endpoint_name: str = DEFAULT_ENDPOINT_NAME,
        dtype: str | None = None,
        build_local: bool = False,
        network_isolation: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # -- container image ----------------------------------------------
        # Two paths, selected by `build_local` (wired from the `buildLocal`
        # context flag in app.py):
        #   * default (build_local=False): a native x86_64 AWS CodeBuild project
        #     builds and pushes the image to a stack-owned ECR repo *during*
        #     `cdk deploy`, so Apple Silicon deployers avoid a slow emulated
        #     cross-build. A custom resource waits for the build to SUCCEED
        #     before the model is created.
        #   * build_local=True: the legacy local DockerImageAsset cross-build.
        # Both paths ultimately produce `container_image_uri` and a
        # `_grant_image_pull` callable that grants pull to the execution role.
        build_trigger = None
        if build_local:
            # SageMaker GPU instances are x86_64. On an Apple Silicon machine
            # this is an emulated cross-build and the first one is slow (tens
            # of minutes) — the default CodeBuild path avoids that.
            image = ecr_assets.DockerImageAsset(
                self,
                "SamImage",
                directory=SAM_SERVER_DIR,
                platform=ecr_assets.Platform.LINUX_AMD64,
            )
            container_image_uri = image.image_uri

            def _grant_image_pull(grantee):
                image.repository.grant_pull(grantee)
        else:
            # Package sam_server/ as an S3 asset. asset_hash is a content hash
            # of the zipped directory, available at synth time. The `exclude`
            # list is the CodeBuild-path equivalent of the .dockerignore
            # allowlist: it keeps the hash stable against edits to files the
            # image build does not use (the Dockerfile COPY governs final image
            # contents regardless).
            source_asset = s3_assets.Asset(
                self,
                "SamSource",
                path=SAM_SERVER_DIR,
                exclude=["HARDWARE.md", "benchmark.py", "server.py", "assets", "*.md"],
            )
            # Deterministic per source content; short and DNS/tag-safe.
            image_tag = "src-" + source_asset.asset_hash[:16]

            # Stack-owned image target. DESTROY + empty_on_delete: the image is
            # fully reproducible from source, so nothing is lost on teardown and
            # we avoid orphaning a repo (and its storage cost) on cdk destroy.
            # IMMUTABLE tags are safe because distinct sources => distinct tags.
            repo = ecr.Repository(
                self,
                "SamRepo",
                image_scan_on_push=True,
                image_tag_mutability=ecr.TagMutability.IMMUTABLE,
                empty_on_delete=True,
                removal_policy=RemovalPolicy.DESTROY,
                # CKV_AWS_136: encrypt the repository with KMS. No
                # encryption_key is supplied, so ECR uses the AWS-managed KMS
                # key -- satisfies the check with no key-management overhead.
                encryption=ecr.RepositoryEncryption.KMS,
                lifecycle_rules=[
                    ecr.LifecycleRule(
                        description="Keep only the most recent images",
                        max_image_count=5,
                    )
                ],
            )

            # Native x86_64, privileged (docker-in-docker), LARGE for the
            # CUDA/torch build. The inline buildspec logs into ECR, builds the
            # image with an explicit --platform linux/amd64, and pushes it.
            project = codebuild.Project(
                self,
                "SamImageBuild",
                source=codebuild.Source.s3(
                    bucket=source_asset.bucket,
                    path=source_asset.s3_object_key,
                ),
                environment=codebuild.BuildEnvironment(
                    build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                    compute_type=codebuild.ComputeType.LARGE,
                    privileged=True,
                ),
                timeout=Duration.minutes(60),
                environment_variables={
                    "REPO_URI": codebuild.BuildEnvironmentVariable(
                        value=repo.repository_uri
                    ),
                    "IMAGE_TAG": codebuild.BuildEnvironmentVariable(
                        value=image_tag
                    ),
                    "AWS_ACCOUNT_ID": codebuild.BuildEnvironmentVariable(
                        value=self.account
                    ),
                    "AWS_DEFAULT_REGION": codebuild.BuildEnvironmentVariable(
                        value=self.region
                    ),
                },
                build_spec=codebuild.BuildSpec.from_object(
                    {
                        "version": "0.2",
                        "env": {"variables": {"DOCKER_BUILDKIT": "1"}},
                        "phases": {
                            "pre_build": {
                                "commands": [
                                    "echo Logging in to Amazon ECR...",
                                    'aws ecr get-login-password --region "$AWS_DEFAULT_REGION" '
                                    "| docker login --username AWS --password-stdin "
                                    '"$AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com"',
                                    # Authenticated ECR Public pulls (the base
                                    # image) get far higher rate limits than
                                    # anonymous ones. The token service lives
                                    # in us-east-1 regardless of build region.
                                    "echo Logging in to Amazon ECR Public...",
                                    "aws ecr-public get-login-password --region us-east-1 "
                                    "| docker login --username AWS --password-stdin public.ecr.aws",
                                ],
                            },
                            "build": {
                                "commands": [
                                    "echo Building image $REPO_URI:$IMAGE_TAG ...",
                                    "docker build --platform linux/amd64 "
                                    '-t "$REPO_URI:$IMAGE_TAG" .',
                                ],
                            },
                            "post_build": {
                                "commands": [
                                    "echo Pushing image...",
                                    'docker push "$REPO_URI:$IMAGE_TAG"',
                                ],
                            },
                        },
                    }
                ),
            )

            # Least privilege: push/pull scoped to this repo, read the source
            # zip from the CDK assets bucket. grant_pull_push also adds the
            # unavoidable ecr:GetAuthorizationToken (resource "*") for the login.
            repo.grant_pull_push(project)
            source_asset.grant_read(project)
            # ECR Public auth for the base-image pull (see buildspec). Both
            # actions are account-scoped; neither supports resource ARNs.
            project.add_to_role_policy(
                iam.PolicyStatement(
                    actions=[
                        "ecr-public:GetAuthorizationToken",
                        "sts:GetServiceBearerToken",
                    ],
                    resources=["*"],
                )
            )

            # Custom resource: start the build on create/update and poll until
            # SUCCEEDED (see handler-source comments above for the BuildId /
            # PhysicalResourceId plumbing decision).
            on_event = _lambda.Function(
                self,
                "BuildTriggerOnEvent",
                runtime=_lambda.Runtime.PYTHON_3_12,
                handler="index.on_event",
                code=_lambda.Code.from_inline(ON_EVENT_SRC),
                timeout=Duration.minutes(1),
                log_retention=logs.RetentionDays.ONE_MONTH,
            )
            is_complete = _lambda.Function(
                self,
                "BuildTriggerIsComplete",
                runtime=_lambda.Runtime.PYTHON_3_12,
                handler="index.is_complete",
                code=_lambda.Code.from_inline(IS_COMPLETE_SRC),
                timeout=Duration.minutes(1),
                log_retention=logs.RetentionDays.ONE_MONTH,
            )
            # This aws-cdk-lib version has no Project.grant_start_build; grant
            # the equivalent least-privilege statement (StartBuild scoped to the
            # single project ARN) directly.
            on_event.add_to_role_policy(
                iam.PolicyStatement(
                    actions=["codebuild:StartBuild"],
                    resources=[project.project_arn],
                )
            )
            is_complete.add_to_role_policy(
                iam.PolicyStatement(
                    actions=["codebuild:BatchGetBuilds"],
                    resources=[project.project_arn],
                )
            )

            provider = cr.Provider(
                self,
                "BuildProvider",
                on_event_handler=on_event,
                is_complete_handler=is_complete,
                query_interval=Duration.seconds(30),
                total_timeout=Duration.minutes(90),
                log_retention=logs.RetentionDays.ONE_MONTH,
            )

            build_trigger = CustomResource(
                self,
                "SamImageBuildTrigger",
                service_token=provider.service_token,
                properties={
                    "ProjectName": project.project_name,
                    "ImageTag": image_tag,
                    "SourceObjectKey": source_asset.s3_object_key,
                },
            )
            build_trigger.node.add_dependency(project)
            build_trigger.node.add_dependency(repo)

            container_image_uri = f"{repo.repository_uri}:{image_tag}"

            def _grant_image_pull(grantee):
                repo.grant_pull(grantee)

        # -- execution role -----------------------------------------------
        role = iam.Role(
            self,
            "SamExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="Execution role for the SAM labeling endpoint",
        )
        _grant_image_pull(role)
        # CKV_AWS_111: do not grant write-class actions on Resource "*".
        # Scope the logs actions to the SageMaker endpoint log group, and split
        # cloudwatch:PutMetricData (which does not support resource-level
        # scoping) into its own statement gated by a cloudwatch:namespace
        # condition. The role still needs image pull (granted above,
        # unchanged).
        role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogStreams",
                ],
                resources=[
                    self.format_arn(
                        service="logs",
                        resource="log-group",
                        resource_name="/aws/sagemaker/Endpoints/*",
                        arn_format=ArnFormat.COLON_RESOURCE_NAME,
                    )
                ],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["cloudwatch:PutMetricData"],
                resources=["*"],
                conditions={
                    "StringEquals": {"cloudwatch:namespace": "/aws/sagemaker/"}
                },
            )
        )

        # -- model ---------------------------------------------------------
        # Weights are baked into the image at /opt/models (Dockerfile sets
        # SAM2_MODEL_DIR); only the optional dtype override is injected here.
        environment = {"SAM2_DTYPE": dtype} if dtype else None

        model = sagemaker.CfnModel(
            self,
            "SamModel",
            execution_role_arn=role.role_arn,
            # CKV_AWS_370: network isolation. The weights are baked into the
            # image and inference makes no outbound calls (HF_HUB_OFFLINE=1),
            # so blocking container network traffic is safe for this workload.
            # Toggle to False via the `networkIsolation` context flag (app.py)
            # if the container is ever found to need outbound network.
            enable_network_isolation=network_isolation,
            primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                image=container_image_uri,
                environment=environment,
            ),
        )
        model.node.add_dependency(role)
        # On the CodeBuild path the image must exist (build SUCCEEDED) before
        # the model references it.
        if build_trigger is not None:
            model.node.add_dependency(build_trigger)

        # -- endpoint config ----------------------------------------------
        # No explicit name: EndpointConfig is immutable, so CloudFormation
        # must replace it whenever the image or instance type changes. Letting
        # it generate the name keeps updates from colliding; endpoint_ctl.py
        # reads the current name from the stack output below.
        endpoint_config = sagemaker.CfnEndpointConfig(
            self,
            "SamEndpointConfig",
            production_variants=[
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    variant_name="AllTraffic",
                    model_name=model.attr_model_name,
                    initial_instance_count=1,
                    instance_type=instance_type,
                    initial_variant_weight=1.0,
                    # Generous: pulling a multi-GB image plus loading the
                    # checkpoint bounds how long the first /ping takes.
                    model_data_download_timeout_in_seconds=1800,
                    container_startup_health_check_timeout_in_seconds=1800,
                )
            ],
        )
        endpoint_config.node.add_dependency(model)

        # -- operator policy ----------------------------------------------
        # Attach to any user or role that should be able to run the tool
        # without permission to change the infrastructure.
        operator_policy = iam.ManagedPolicy(
            self,
            "SamOperatorPolicy",
            description=(
                "Start, stop, and invoke the SAM labeling endpoint "
                "(no infrastructure changes)"
            ),
            statements=[
                iam.PolicyStatement(
                    sid="RunAndInvokeEndpoint",
                    actions=[
                        "sagemaker:CreateEndpoint",
                        "sagemaker:DeleteEndpoint",
                        "sagemaker:DescribeEndpoint",
                        "sagemaker:InvokeEndpoint",
                        "sagemaker:AddTags",
                        "sagemaker:ListTags",
                    ],
                    resources=[
                        self.format_arn(
                            service="sagemaker",
                            resource="endpoint",
                            resource_name=endpoint_name,
                        )
                    ],
                ),
                iam.PolicyStatement(
                    sid="ReadEndpointConfig",
                    actions=["sagemaker:DescribeEndpointConfig"],
                    resources=[
                        self.format_arn(
                            service="sagemaker",
                            resource="endpoint-config",
                            resource_name="*",
                        )
                    ],
                ),
                iam.PolicyStatement(
                    sid="DiscoverStackOutputs",
                    actions=["cloudformation:DescribeStacks"],
                    resources=[self.stack_id],
                ),
                iam.PolicyStatement(
                    sid="ListEndpoints",
                    actions=["sagemaker:ListEndpoints"],
                    resources=["*"],
                ),
            ],
        )

        # -- outputs -------------------------------------------------------
        # endpoint_ctl.py reads EndpointConfigName and EndpointName from here.
        CfnOutput(
            self,
            "EndpointConfigName",
            value=endpoint_config.attr_endpoint_config_name,
            description="Pass to endpoint_ctl.py / CreateEndpoint",
        )
        CfnOutput(
            self,
            "EndpointName",
            value=endpoint_name,
            description="Endpoint name the labeler should connect to",
        )
        CfnOutput(self, "ModelName", value=model.attr_model_name)
        CfnOutput(self, "ImageUri", value=container_image_uri)
        CfnOutput(self, "InstanceType", value=instance_type)
        CfnOutput(
            self,
            "OperatorPolicyArn",
            value=operator_policy.managed_policy_arn,
            description="Attach to other users so they can run the endpoint",
        )
