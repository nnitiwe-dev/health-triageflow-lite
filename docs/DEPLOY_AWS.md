# Deploy the demo to AWS (Amazon ECS Express Mode, with CI/CD)

This runs TriageFlow Lite as a public HTTPS demo on **Amazon ECS Express Mode**, redeployed
automatically every time the tests pass on `main`.

**What you get:** a public `https://<service>.ecs.us-east-1.on.aws` URL, TLS included, an
Application Load Balancer and autoscaling provisioned for you, and the 7B model baked into the
image so there is no cold-start download.

> ### Why not App Runner?
> AWS **closed App Runner to new customer accounts on 2026-04-30**. Existing App Runner
> customers keep running, but a new account (like this one) gets
> `InvalidRequestException: Account … can't create new App Runner services` no matter what IAM
> permissions it has — it is an account-level restriction, not a policy gap. AWS now points new
> and migrating users to **ECS Express Mode**, which gives the same "hand it an image, get a URL"
> experience on top of Fargate + an ALB. This guide (and the CI/CD pipeline) use Express Mode.
> See the [App Runner availability change](https://docs.aws.amazon.com/apprunner/latest/dg/apprunner-availability-change.html)
> and [Express Mode docs](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/express-service-getting-started.html).

**Honest notes before you start**

- **Cost.** The 7B model needs **4 vCPU / 12 GB** of Fargate, plus a small always-on ALB.
  Running it continuously is roughly **US$5–12 per day**. **Scale the service to zero tasks when
  you are not demoing** (one command, below) and you pay only the ALB (a few cents an hour).
  Check [Fargate pricing](https://aws.amazon.com/fargate/pricing/) for current rates.
- **Data residency.** This public demo runs inference **on AWS**, on **synthetic data only**.
  The "patient data never leaves the facility" property in the README is for running the same
  image on a clinic's own hardware — it is not what the public cloud demo demonstrates. The
  UI's data chip says "cloud demo · synthetic data only" so nobody is misled.
- **Region & VPC.** This guide uses **`us-east-1`**, the region you chose. Express Mode puts the
  load balancer in your account's **default VPC** with public subnets; new AWS accounts have one
  automatically. If yours was deleted, run `aws ec2 create-default-vpc` first (or pass your own
  `--subnets`).

You do the one-time setup in phases A–C. After that, `git push` to `main` is the whole deploy
workflow: CI builds the image, pushes it to ECR, and the first-party
[`aws-actions/amazon-ecs-deploy-express-service`](https://github.com/aws-actions/amazon-ecs-deploy-express-service)
action **creates the service the first time and updates it in place on every later push**.

---

## Before you begin

> **Run the one-time setup (Phases A–B) as your normal console identity**, which already has
> admin rights — in AWS CloudShell that is automatic. The `triageflow-ci` user you created is
> deliberately minimal (push images + deploy only); it **cannot** create the ECR repository or
> the IAM roles. If you ran `aws configure` with the CI user's keys in CloudShell, revert to your
> console identity first:
>
> ```bash
> rm -f ~/.aws/credentials        # drop the CI user's static keys
> aws sts get-caller-identity     # should now show your own user/role, not triageflow-ci
> ```
>
> The CI user's keys belong in **GitHub secrets**, not in CloudShell.

Set a couple of shell variables used throughout:

```bash
export AWS_REGION=us-east-1
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REPO=health-triageflow-lite
export ECR_URI=$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO
```

---

## Phase A — Create the image registry

Express Mode deploys *from* an image, so a registry must exist. This is the only AWS resource
you create by hand for images; **CI builds and pushes the image itself** (a 5+ GB model image
won't build in CloudShell, and you shouldn't have to build it locally).

```bash
aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION
```

## Phase B — Create the two IAM roles Express Mode needs

Express Mode uses two roles. These are **standard, named roles** — create them once as your
console identity and both the CI action and AWS reuse them. (Full detail in the
[AWS getting-started guide](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/express-service-getting-started.html).)

| Role | What it does | Managed policy |
|------|--------------|----------------|
| `ecsTaskExecutionRole` | lets ECS pull the image from ECR and write CloudWatch logs | `AmazonECSTaskExecutionRolePolicy` |
| `ecsInfrastructureRoleForExpressServices` | lets ECS create/manage the ALB, target groups, autoscaling | `AmazonECSInfrastructureRoleforExpressGatewayServices` |

```bash
# Task execution role (trusted by ECS tasks)
aws iam create-role --role-name ecsTaskExecutionRole \
  --assume-role-policy-document '{
    "Version":"2012-10-17",
    "Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]
  }'
aws iam attach-role-policy --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

# Infrastructure role (trusted by the ECS service)
aws iam create-role --role-name ecsInfrastructureRoleForExpressServices \
  --assume-role-policy-document '{
    "Version":"2012-10-17",
    "Statement":[{"Sid":"AllowAccessInfrastructureForECSExpressServices","Effect":"Allow","Principal":{"Service":"ecs.amazonaws.com"},"Action":"sts:AssumeRole"}]
  }'
aws iam attach-role-policy --role-name ecsInfrastructureRoleForExpressServices \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSInfrastructureRoleforExpressGatewayServices
```

> IAM roles are eventually consistent — if the very first deploy fails with "Unable to assume
> the … role" seconds after you create these, wait a minute and re-run it.

That is all the hand-setup there is: **you do not create the service by hand.** CI's deploy
action creates it on the first push to `main` and updates it afterwards.

---

## Phase C — Wire up CI/CD

**1. Create a least-privilege IAM policy** for the deploy user (push to this one repo; manage the
Express service; pass only the two roles above). Save as `deploy-policy.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Sid": "EcrLogin", "Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*" },
    { "Sid": "EcrPush", "Effect": "Allow",
      "Action": ["ecr:BatchCheckLayerAvailability","ecr:GetDownloadUrlForLayer","ecr:BatchGetImage",
                 "ecr:InitiateLayerUpload","ecr:UploadLayerPart","ecr:CompleteLayerUpload","ecr:PutImage"],
      "Resource": "arn:aws:ecr:us-east-1:ACCOUNT_ID:repository/health-triageflow-lite" },
    { "Sid": "EcsExpressDeploy", "Effect": "Allow",
      "Action": ["ecs:CreateCluster","ecs:RegisterTaskDefinition",
                 "ecs:CreateExpressGatewayService","ecs:UpdateExpressGatewayService","ecs:DescribeExpressGatewayService",
                 "ecs:DescribeClusters","ecs:DescribeServices","ecs:ListServiceDeployments","ecs:DescribeServiceDeployments",
                 "ecs:TagResource","ecs:UntagResource"],
      "Resource": "*" },
    { "Sid": "PassEcsRoles", "Effect": "Allow", "Action": "iam:PassRole",
      "Resource": [
        "arn:aws:iam::ACCOUNT_ID:role/ecsTaskExecutionRole",
        "arn:aws:iam::ACCOUNT_ID:role/ecsInfrastructureRoleForExpressServices"
      ] }
  ]
}
```

> Scoping notes:
> - **ECR push** is pinned to the one repository.
> - **The ECS Express actions** need `Resource: "*"` — these API calls don't support resource-level
>   scoping, and the service name is fixed in the workflow, so CI can only touch the one service it
>   names. This matches AWS's own documented policy for the deploy action.
> - **`iam:PassRole`** is pinned to the **two roles above and nothing else** — CI cannot pass any
>   other role (this is the privilege-escalation surface, so it is the one that matters most).
>   Unlike the old App Runner policy, there is **no `iam:PassedToService` condition** — resource
>   scoping is sufficient, and the condition is what silently broke the earlier App Runner deploy.
>
> If you already have the `triageflow-ci-deploy` policy from the App Runner attempt, replace it in
> place (keeps the same ARN, no re-attach needed):
>
> ```bash
> aws iam create-policy-version \
>   --policy-arn arn:aws:iam::$ACCOUNT_ID:policy/triageflow-ci-deploy \
>   --policy-document file://deploy-policy.json --set-as-default
> ```
> (If that errors with *"cannot exceed 5 versions"*, delete a non-default version first:
> `aws iam list-policy-versions --policy-arn arn:aws:iam::$ACCOUNT_ID:policy/triageflow-ci-deploy`
> then `aws iam delete-policy-version … --version-id vN`.)

First time only — create the policy and the user, and attach them:

```bash
aws iam create-policy --policy-name triageflow-ci-deploy --policy-document file://deploy-policy.json
aws iam attach-user-policy --user-name triageflow-ci \
  --policy-arn arn:aws:iam::$ACCOUNT_ID:policy/triageflow-ci-deploy
aws iam create-access-key --user-name triageflow-ci   # note the AccessKeyId + SecretAccessKey
```

**2. Add these in GitHub → Settings → Secrets and variables → Actions:**

| Kind | Name | Value |
|------|------|-------|
| Secret | `AWS_ACCESS_KEY_ID` | the CI user's access key id |
| Secret | `AWS_SECRET_ACCESS_KEY` | the CI user's secret |
| Variable | `AWS_REGION` | `us-east-1` |
| Variable | `ECR_REPOSITORY` | `health-triageflow-lite` |
| Variable | `ECS_EXECUTION_ROLE_ARN` | `arn:aws:iam::ACCOUNT_ID:role/ecsTaskExecutionRole` |
| Variable | `ECS_INFRASTRUCTURE_ROLE_ARN` | `arn:aws:iam::ACCOUNT_ID:role/ecsInfrastructureRoleForExpressServices` |
| Variable | `DEPLOY_TO_AWS` | `true` |

The role ARNs are **not secrets** (they're just identifiers) — set them with the GitHub CLI if
you prefer:

```bash
gh variable set ECS_EXECUTION_ROLE_ARN      --body "arn:aws:iam::$ACCOUNT_ID:role/ecsTaskExecutionRole"
gh variable set ECS_INFRASTRUCTURE_ROLE_ARN --body "arn:aws:iam::$ACCOUNT_ID:role/ecsInfrastructureRoleForExpressServices"
```

> `APPRUNNER_SERVICE_ARN` is no longer used — you can delete that secret if it's still there.

The `deploy` job in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) stays **skipped**
until `DEPLOY_TO_AWS` is `true`, runs **only on `main`**, and **only after** every test and
runtime job passes. It also only runs when the push actually changed something that goes into the
image (`Dockerfile`, `.dockerignore`, `pyproject.toml`, `uv.lock`, `src/**`, `protocols/**`, or
the workflow itself) — a docs- or test-only commit skips the build and redeploy. When it does
build, a `:buildcache` image tag (mode=max) lets it reuse the ~5 GB model layer instead of
re-downloading it, so only changed app layers rebuild. The buildcache adds a few GB of ECR
storage (≈ US$0.10/GB-month).

**3. Deploy** by pushing to `main` (or re-running the latest `main` workflow). The job builds the
image, pushes it to ECR reusing the unchanged model layer, then the deploy action creates the
Express service on the first run (provisioning the ALB + URL — this first time takes several
minutes) and updates it in place every run after. When it finishes it posts a commit comment with
the live `https://…ecs.us-east-1.on.aws` URL.

Open the URL. The priority scoring, approval gate and audit trail work immediately; the 7B model
finishes loading a minute or two after the task starts, and the banner clears.

---

## Managing the running service

Find the service ARN and URL:

```bash
aws ecs list-express-gateway-services --region $AWS_REGION    # lists ARNs
SERVICE_ARN=<paste the one named health-triageflow-lite>
aws ecs describe-express-gateway-service --service-arn "$SERVICE_ARN" \
  --query '{status:status.statusCode,url:targetConfiguration.ingressPaths[0].endpoint}'
```

**Stop paying when you're not demoing** — scale to zero tasks (the ALB stays, so the URL is
preserved; a few cents an hour). Resume by scaling back to 1:

```bash
# pause (0 tasks)
aws ecs update-express-gateway-service --service-arn "$SERVICE_ARN" \
  --scaling-target '{"minTaskCount":0,"maxTaskCount":0}' --monitor-resources TEXT-ONLY
# resume (1 task)
aws ecs update-express-gateway-service --service-arn "$SERVICE_ARN" \
  --scaling-target '{"minTaskCount":1,"maxTaskCount":1}' --monitor-resources TEXT-ONLY
```

To delete it entirely (removes the ALB and URL):

```bash
aws ecs delete-express-gateway-service --service-arn "$SERVICE_ARN" --monitor-resources TEXT-ONLY
```

---

## Changing the model later

Edit the `ARG` defaults at the top of the [`Dockerfile`](../Dockerfile) (or pass `--build-arg`),
and set the matching size in the workflow's deploy step (`cpu` in CPU units, `memory` in MiB):

| Model | `MODEL_REPO` / `MODEL_FILE` / `MODEL_ID` | `cpu` / `memory` |
|-------|------------------------------------------|------------------|
| 7B (current) | `bartowski/Qwen2.5-7B-Instruct-GGUF` · `Qwen2.5-7B-Instruct-Q4_K_M.gguf` · `qwen2.5-7b-instruct-q4` | `4096` / `12288` (4 vCPU / 12 GB) |
| 3B | `bartowski/Qwen2.5-3B-Instruct-GGUF` · `Qwen2.5-3B-Instruct-Q4_K_M.gguf` · `qwen2.5-3b-instruct-q4` | `2048` / `4096` (2 vCPU / 4 GB) |
| 0.5B | `bartowski/Qwen2.5-0.5B-Instruct-GGUF` · `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` · `qwen2.5-0.5b-instruct-q4` | `1024` / `2048` (1 vCPU / 2 GB) |

For a deterministic-only demo (no model — scoring, the gate and the audit trail all still work,
only free-text sign suggestions are off), add `TRIAGEFLOW_NO_LLM=1` as a container environment
variable (via the action's `environment-variables` input) and drop to `1024`/`2048`.

---

## Troubleshooting

- **Deploy job was skipped** — `DEPLOY_TO_AWS` is not `true`, the push was not to `main`, or the
  commit changed only docs/tests (nothing that affects the image). To force a redeploy with no
  code change, bump a comment in the `Dockerfile` or re-run the last image-changing workflow.
- **`Account … can't create new App Runner services`** — you're on an old App Runner workflow;
  this guide/pipeline no longer use App Runner (see the top). Pull `main` so you have the
  Express Mode `deploy` job.
- **`not authorized to perform: iam:PassRole`** — the CI policy's `PassEcsRoles` statement is
  missing or names the wrong role ARNs. Re-apply `deploy-policy.json` (step C.1). It must list
  **both** `ecsTaskExecutionRole` and `ecsInfrastructureRoleForExpressServices`, with **no**
  `iam:PassedToService` condition.
- **"Unable to assume role" right after creating the roles** — IAM eventual consistency; wait a
  minute and re-run the deploy.
- **No default VPC** — `aws ec2 create-default-vpc`, or pass your own `--subnets` / `subnets`.
- **Service never goes healthy** — the ALB health check hits `/api/status` on port `8000`; both
  are set in the workflow. Check the task's CloudWatch logs (log group is created automatically).
  The web server answers in seconds; the model keeps loading in the background after that, so a
  slow first *suggestion* is normal, a failing *health check* is not.
- **`docker build` fails downloading the model** — a Hugging Face hiccup; re-run. The URL and
  filename are pinned in the Dockerfile.
