# Deploy the demo to AWS App Runner (with CI/CD)

This runs TriageFlow Lite as a public HTTPS demo on **AWS App Runner**, redeployed
automatically every time the tests pass on `main`.

**What you get:** a public `https://<id>.us-east-1.awsapprunner.com` URL, TLS included, the
7B model baked into the image so there is no cold-start download.

**Honest notes before you start**

- **Cost.** The 7B model needs App Runner's largest size (**4 vCPU / 12 GB**). Running it
  continuously is roughly **US$5–10 per day**. **Pause the service when you are not demoing**
  (one command, below) and it costs almost nothing. Check the
  [App Runner pricing page](https://aws.amazon.com/apprunner/pricing/) for current rates.
- **Data residency.** This public demo runs inference **on AWS**, on **synthetic data only**.
  The "patient data never leaves the facility" property in the README is for running the same
  image on a clinic's own hardware — it is not what the public cloud demo demonstrates. The
  UI's data chip says "cloud demo · synthetic data only" so nobody is misled.
- **Region.** App Runner is not available in Cape Town (`af-south-1`). This guide uses
  **`us-east-1`**, the region you chose.

You do the one-time setup in phases A–C. After that, `git push` to `main` is the whole
deploy workflow.

---

## Before you begin

> **Run the one-time setup (Phases A–B) as your normal console identity**, which already has
> admin rights — in AWS CloudShell that is automatic. The `triageflow-ci` user you created is
> deliberately minimal (push images + deploy only); it **cannot** create the repository or the
> service. If you ran `aws configure` with the CI user's keys in CloudShell, revert to your
> console identity first:
>
> ```bash
> rm -f ~/.aws/credentials        # drop the CI user's static keys
> aws sts get-caller-identity     # should now show your own user/role, not triageflow-ci
> ```
>
> The CI user's keys belong in **GitHub secrets**, not in CloudShell.


Install and sign in:

```bash
aws --version          # AWS CLI v2
docker --version       # only needed for the one-time first push in Phase A
aws configure          # use the IAM user's access key, region us-east-1
```

Set a couple of shell variables used throughout:

```bash
export AWS_REGION=us-east-1
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REPO=health-triageflow-lite
export ECR_URI=$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO
```

---

## Phase A — Create the image registry

App Runner deploys *from* an image, so a registry must exist. This is the only AWS resource
you create by hand; **CI builds and pushes the image itself** (a 5+ GB model image won't build
in CloudShell, and you shouldn't have to build it locally).

```bash
aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION
```

Now wire up GitHub so the first push builds the image **(do Phase C's secrets/variables now,
but leave `APPRUNNER_SERVICE_ARN` unset)**, then push to `main`. The `deploy` job builds the
image, pushes it to ECR as `:latest`, and — because no service exists yet — prints a note and
stops without deploying. You now have an image to create the service from.

> Prefer to build the very first image yourself instead? With Docker and ~15 GB free disk:
> `docker build --platform linux/amd64 -t $ECR_URI:latest . && docker push $ECR_URI:latest`
> (after `aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com`).
> To test the container cheaply first, build the tiny model:
> `docker build --platform linux/amd64 --build-arg MODEL_REPO=bartowski/Qwen2.5-0.5B-Instruct-GGUF --build-arg MODEL_FILE=Qwen2.5-0.5B-Instruct-Q4_K_M.gguf --build-arg MODEL_ID=qwen2.5-0.5b-instruct-q4 -t test .`
> then `docker run -p 8000:8000 test` and open <http://127.0.0.1:8000>.

## Phase B — Create the App Runner service

Easiest in the **console** (it offers to create the ECR-access role for you); the CLI
equivalent is at the end.

1. Open **App Runner → Create service**.
2. **Source:** Container registry → Amazon ECR. **Browse** to `health-triageflow-lite:latest`.
3. **Deployment settings:** choose **Manual** (CI triggers each deploy). For the ECR access
   role, let App Runner **create a new service role** (`AppRunnerECRAccessRole`).
4. **Configure service:**
   - **CPU:** 4 vCPU · **Memory:** 12 GB (required for the 7B model).
   - **Port:** `8000`.
   - **Health check:** Protocol **HTTP**, Path **`/api/status`**. Set the healthy threshold to
     a few retries and a longer interval — the web server is healthy within seconds, and the
     model keeps loading in the background after that.
   - **Environment variables** (optional — the image already sets good defaults): you can add
     `TRIAGEFLOW_DATA_REGION` to change the data-chip label, or `TRIAGEFLOW_NO_LLM=1` to run a
     model-free (and much cheaper, 1 vCPU/2 GB) deterministic-only demo.
   - **Auto scaling:** min 1 / max 1 is plenty for a demo and avoids paying for extra instances.
5. **Create & deploy.** When it is ready, note the **Default domain** (your public URL) and the
   **Service ARN** (`arn:aws:apprunner:us-east-1:…:service/health-triageflow-lite/…`).

Open the URL. The priority scoring, approval gate and audit trail work immediately; the model
finishes loading within a minute or two and the banner clears.

**Pause it when you are done** (stops the bill; resume in seconds for the next demo):

```bash
aws apprunner pause-service  --service-arn <SERVICE_ARN>
aws apprunner resume-service --service-arn <SERVICE_ARN>
```

---

## Phase C — Wire up CI/CD

**1. Create a least-privilege IAM policy** for the deploy user (push to this one repo, deploy
this one service). Save as `deploy-policy.json`, filling in your account id and service ARN:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Sid": "EcrLogin", "Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*" },
    { "Sid": "EcrPush", "Effect": "Allow",
      "Action": ["ecr:BatchCheckLayerAvailability","ecr:GetDownloadUrlForLayer","ecr:BatchGetImage",
                 "ecr:InitiateLayerUpload","ecr:UploadLayerPart","ecr:CompleteLayerUpload","ecr:PutImage"],
      "Resource": "arn:aws:ecr:us-east-1:ACCOUNT_ID:repository/health-triageflow-lite" },
    { "Sid": "AppRunnerDeploy", "Effect": "Allow",
      "Action": ["apprunner:StartDeployment","apprunner:DescribeService"],
      "Resource": "arn:aws:apprunner:us-east-1:ACCOUNT_ID:service/health-triageflow-lite/*" }
  ]
}
```

Attach it to the IAM user whose access key you will give GitHub (create a dedicated user such
as `triageflow-ci` — not your personal admin user):

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
| Secret | `APPRUNNER_SERVICE_ARN` | the service ARN from Phase B |
| Variable | `AWS_REGION` | `us-east-1` |
| Variable | `ECR_REPOSITORY` | `health-triageflow-lite` |
| Variable | `DEPLOY_TO_AWS` | `true` |

The `deploy` job in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) stays **skipped**
until `DEPLOY_TO_AWS` is `true`, so adding the Docker files changed nothing and cost nothing
until this step. The job also runs **only on `main`** and **only after** every test and
runtime job passes.

You will set these in two passes: **before Phase A's first push**, add everything *except*
`APPRUNNER_SERVICE_ARN` (so CI can build the first image); **after Phase B** creates the
service, add `APPRUNNER_SERVICE_ARN` too.

**3. Deploy** by pushing to `main` (or re-running the latest `main` workflow). With the ARN
set, the job builds the image, pushes it to ECR reusing the unchanged model layer, and calls
`apprunner start-deployment`. App Runner rolls out the new image with no downtime.

---

## Changing the model later

Edit the `ARG` defaults at the top of the [`Dockerfile`](../Dockerfile) (or pass
`--build-arg`), and in App Runner set the size to match:

| Model | `MODEL_REPO` / `MODEL_FILE` / `MODEL_ID` | App Runner size |
|-------|------------------------------------------|-----------------|
| 7B (current) | `bartowski/Qwen2.5-7B-Instruct-GGUF` · `Qwen2.5-7B-Instruct-Q4_K_M.gguf` · `qwen2.5-7b-instruct-q4` | 4 vCPU / 12 GB |
| 3B | `bartowski/Qwen2.5-3B-Instruct-GGUF` · `Qwen2.5-3B-Instruct-Q4_K_M.gguf` · `qwen2.5-3b-instruct-q4` | 2 vCPU / 4 GB |
| 0.5B | `bartowski/Qwen2.5-0.5B-Instruct-GGUF` · `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` · `qwen2.5-0.5b-instruct-q4` | 1 vCPU / 2 GB |

Set `TRIAGEFLOW_NO_LLM=1` as a service env var for a deterministic-only demo (no model, 1 vCPU
/ 2 GB, cheapest) — scoring, the gate and the audit trail all still work; only free-text sign
suggestions are off.

---

## Troubleshooting

- **Deploy job was skipped** — `DEPLOY_TO_AWS` is not `true`, or the push was not to `main`.
- **Service stuck "Operation in progress" / rollback** — open the App Runner **Application
  logs**. A health check failing usually means the port (8000) or health path (`/api/status`)
  is wrong, or the size is too small for the model (the 7B needs 12 GB).
- **First request slow** — the model loads in the background after the container starts;
  priorities work immediately, suggestions appear once it is warm.
- **`docker build` fails downloading the model** — a Hugging Face hiccup; re-run. The URL and
  filename are pinned in the Dockerfile.

### CLI alternative to Phase B

```bash
aws apprunner create-service \
  --service-name health-triageflow-lite \
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "'"$ECR_URI"':latest",
      "ImageRepositoryType": "ECR",
      "ImageConfiguration": { "Port": "8000" }
    },
    "AutoDeploymentsEnabled": false,
    "AuthenticationConfiguration": { "AccessRoleArn": "arn:aws:iam::'"$ACCOUNT_ID"':role/service-role/AppRunnerECRAccessRole" }
  }' \
  --instance-configuration '{ "Cpu": "4 vCPU", "Memory": "12 GB" }' \
  --health-check-configuration '{ "Protocol": "HTTP", "Path": "/api/status", "Interval": 20, "Timeout": 5, "HealthyThreshold": 1, "UnhealthyThreshold": 5 }' \
  --region $AWS_REGION
```

(Create `AppRunnerECRAccessRole` once via the console's "Create service" flow, or with an IAM
role that trusts `build.apprunner.amazonaws.com` and has `AWSAppRunnerServicePolicyForECRAccess`.)
