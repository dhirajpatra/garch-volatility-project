Complete MLOps pipeline for deploying GARCH Volatility Model to Azure with CI/CD, monitoring, and automated testing.

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Deployment Steps](#deployment-steps)
- [CI/CD Pipeline](#cicd-pipeline)
- [Monitoring & Alerts](#monitoring--alerts)
- [Troubleshooting](#troubleshooting)
- [Cost Estimation](#cost-estimation)

## 🎯 Overview

This MLOps pipeline provides:

✅ **Infrastructure as Code** - Automated Azure resource provisioning  
✅ **Containerization** - Docker-based deployment  
✅ **CI/CD** - GitHub Actions & Azure Pipelines  
✅ **Automated Testing** - Unit, integration, and load tests  
✅ **Monitoring** - Azure Monitor with custom dashboards  
✅ **Alerting** - Email and webhook notifications  
✅ **Logging** - Centralized log management  
✅ **Auto-scaling** - Based on CPU/memory metrics  

## 📦 Prerequisites

### 1. Azure Account Setup

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to Azure
az login

# Set subscription
az account set --subscription "YOUR_SUBSCRIPTION_ID"

# Get subscription ID
az account show --query id -o tsv
```

### 2. Azure Service Principal

Create service principal for automation:

```bash
az ad sp create-for-rbac --name "garch-mlops-sp" \
  --role contributor \
  --scopes /subscriptions/YOUR_SUBSCRIPTION_ID \
  --sdk-auth
```

Save the output JSON - you'll need it for GitHub secrets.

### 3. Install Required Tools

```bash
# Python packages
pip install azure-identity azure-mgmt-resource azure-mgmt-containerregistry \
            azure-mgmt-containerinstance azure-mgmt-monitor \
            azure-storage-blob pyyaml requests

# Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# kubectl (for Kubernetes deployment)
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

## 🚀 Quick Start

### Option 1: Automated Full Deployment

```bash
# Clone repository
git clone https://github.com/yourusername/garch-volatility-project.git
cd garch-volatility-project

# Set environment variables
export AZURE_SUBSCRIPTION_ID="your-subscription-id"

# Run full deployment
python mlops_deploy.py --action full
```

### Option 2: Step-by-Step Deployment

```bash
# Step 1: Infrastructure
python mlops_deploy.py --action deploy

# Step 2: Run tests
python mlops_deploy.py --action test

# Step 3: Start monitoring
python mlops_deploy.py --action monitor
```

## 🔧 Detailed Setup

### 1. Configuration File

Create `azure_config.yaml`:

```yaml
azure:
  subscription_id: "your-subscription-id"
  resource_group: "garch-model-rg"
  location: "eastus"
  acr_name: "garchmodelregistry"
  aci_name: "garch-api-container"
  storage_account: "garchmodelstorage"

deployment:
  image_name: "garch-prediction-api"
  image_tag: "latest"
  cpu: 2.0
  memory_gb: 4.0
  port: 8001

monitoring:
  log_analytics_workspace: "garch-logs"
  alert_email: "alerts@example.com"
  health_check_interval: 300  # seconds

testing:
  test_endpoints: ["/health", "/models/list", "/predict"]
  load_test_duration: 60
  load_test_users: 10
```

### 2. Environment Variables

Create `.env` file:

```bash
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_TENANT_ID=your-tenant-id
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret
```

### 3. GitHub Secrets Setup

Add these secrets to your GitHub repository:

| Secret Name | Description | How to Get |
|-------------|-------------|------------|
| `AZURE_CREDENTIALS` | Service Principal JSON | From `az ad sp create-for-rbac` |
| `AZURE_SUBSCRIPTION_ID` | Subscription ID | `az account show` |
| `ACR_USERNAME` | ACR admin username | From Azure Portal |
| `ACR_PASSWORD` | ACR admin password | From Azure Portal |

## 📝 Deployment Steps

### Step 1: Infrastructure Provisioning

The script creates:

```
Azure Resource Group
├── Container Registry (ACR)
├── Container Instances (ACI)
├── Storage Account
├── Log Analytics Workspace
└── Application Insights
```

**Manual Creation (Alternative):**

```bash
# Resource Group
az group create --name garch-model-rg --location eastus

# Container Registry
az acr create --resource-group garch-model-rg \
  --name garchmodelregistry --sku Basic \
  --admin-enabled true

# Storage Account
az storage account create \
  --name garchmodelstorage \
  --resource-group garch-model-rg \
  --location eastus \
  --sku Standard_LRS

# Log Analytics
az monitor log-analytics workspace create \
  --resource-group garch-model-rg \
  --workspace-name garch-logs
```

### Step 2: Docker Image Build & Push

```bash
# Build image locally
docker build -t garch-prediction-api:latest .

# Login to ACR
az acr login --name garchmodelregistry

# Tag image
docker tag garch-prediction-api:latest \
  garchmodelregistry.azurecr.io/garch-prediction-api:latest

# Push to ACR
docker push garchmodelregistry.azurecr.io/garch-prediction-api:latest
```

### Step 3: Container Deployment

```bash
# Deploy to ACI
az container create \
  --resource-group garch-model-rg \
  --name garch-api-container \
  --image garchmodelregistry.azurecr.io/garch-prediction-api:latest \
  --cpu 2 --memory 4 \
  --registry-login-server garchmodelregistry.azurecr.io \
  --registry-username YOUR_ACR_USERNAME \
  --registry-password YOUR_ACR_PASSWORD \
  --dns-name-label garch-api \
  --ports 8001 \
  --environment-variables 'ENV=production'

# Get public IP
az container show --resource-group garch-model-rg \
  --name garch-api-container \
  --query ipAddress.ip --output tsv
```

### Step 4: Automated Testing

The deployment script runs:

#### 4.1 Health Check Test
```python
GET http://{container_ip}:8001/health
Expected: {"status": "healthy"}
```

#### 4.2 Endpoint Tests
```python
# Test all endpoints
endpoints = ['/health', '/models/list', '/predict']
for endpoint in endpoints:
    response = requests.get(f"{base_url}{endpoint}")
    assert response.status_code == 200
```

#### 4.3 Load Test
```python
# Simulate 10 concurrent users for 60 seconds
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(make_request) for _ in range(100)]
    results = [f.result() for f in futures]

# Metrics: Success rate, avg response time, RPS
```

#### 4.4 Integration Test
```python
# Full prediction workflow
POST /predict/custom
{
  "returns": [0.01, -0.02, ...],
  "p": 1,
  "q": 1,
  "horizon": 5
}
# Verify predictions are returned
```

### Step 5: Monitoring Setup

#### 5.1 Azure Monitor Dashboard

```bash
# Create monitoring dashboard
az portal dashboard create \
  --resource-group garch-model-rg \
  --name "GARCH Model Dashboard" \
  --input-path monitoring_dashboard.json
```

**Metrics Tracked:**
- CPU Usage (%)
- Memory Usage (%)
- Network In/Out (MB/s)
- Request Count
- Response Time (ms)
- Error Rate (%)

#### 5.2 Alert Rules

```bash
# CPU Alert
az monitor metrics alert create \
  --name "High CPU Usage" \
  --resource-group garch-model-rg \
  --scopes /subscriptions/{sub-id}/resourceGroups/garch-model-rg/providers/Microsoft.ContainerInstance/containerGroups/garch-api-container \
  --condition "avg CpuUsage > 80" \
  --description "CPU usage above 80%" \
  --evaluation-frequency 5m \
  --window-size 15m \
  --severity 2

# Memory Alert
az monitor metrics alert create \
  --name "High Memory Usage" \
  --resource-group garch-model-rg \
  --condition "avg MemoryUsage > 90" \
  --severity 1

# Health Check Alert
az monitor metrics alert create \
  --name "API Unhealthy" \
  --condition "count failures > 3" \
  --severity 0
```

#### 5.3 Action Groups

```bash
# Email notifications
az monitor action-group create \
  --name "MLOps-Team-Notifications" \
  --resource-group garch-model-rg \
  --short-name "MLOps" \
  --email-receiver name=admin email=admin@example.com

# Webhook for Slack/Teams
az monitor action-group create \
  --webhook-receiver name=slack \
    service-uri=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

File: `.github/workflows/cicd.yml`

```yaml
name: GARCH Model CI/CD

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Tests
        run: pytest tests/ --cov

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Build and Push to ACR
        run: |
          az acr build --registry ${{ secrets.ACR_NAME }} \
            --image garch-api:${{ github.sha }} .

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Azure
        run: python mlops_deploy.py --action deploy

  integration-test:
    needs: deploy
    runs-on: ubuntu-latest
    steps:
      - name: Run Integration Tests
        run: pytest tests/integration/ -v
```

### Azure DevOps Pipeline

File: `azure-pipelines.yml`

```yaml
trigger:
  - main

stages:
  - stage: Build
    jobs:
      - job: BuildAndTest
        steps:
          - script: pytest tests/
          - task: Docker@2
            inputs:
              command: buildAndPush

  - stage: Deploy
    jobs:
      - deployment: Production
        environment: production
        strategy:
          runOnce:
            deploy:
              steps:
                - script: python mlops_deploy.py
```

## 📊 Monitoring & Alerts

### Real-time Monitoring Dashboard

Access at: https://portal.azure.com

**Key Metrics:**

1. **Performance Metrics**
   - Response Time: P50, P95, P99
   - Throughput: Requests/second
   - Error Rate: 4xx, 5xx errors

2. **Resource Metrics**
   - CPU Usage: Current, Average, Peak
   - Memory Usage: Current, Average, Peak
   - Network: Bytes In/Out

3. **Model Metrics**
   - Prediction Count
   - Average Prediction Time
   - Model Load Time

### Log Queries (KQL)

```kusto
// Find errors in last 24 hours
ContainerInstanceLog_CL
| where TimeGenerated > ago(24h)
| where Message contains "ERROR"
| project TimeGenerated, Message
| order by TimeGenerated desc

// API response times
ContainerInstanceLog_CL
| where Message contains "prediction"
| extend ResponseTime = extract("time=([0-9.]+)", 1, Message)
| summarize avg(todouble(ResponseTime)) by bin(TimeGenerated, 5m)

// Failed health checks
ContainerInstanceLog_CL
| where Message contains "health" and Message contains "failed"
| summarize count() by bin(TimeGenerated, 1h)
```

### Custom Monitoring Script

```python
# monitor.py
import requests
import time
from datetime import datetime

def monitor_api(base_url, interval=60):
    while True:
        try:
            # Health check
            response = requests.get(f"{base_url}/health")
            status = response.status_code
            
            # Log metrics
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'status_code': status,
                'response_time': response.elapsed.total_seconds(),
                'healthy': status == 200
            }
            
            print(f"[{metrics['timestamp']}] Status: {status}, "
                  f"Response Time: {metrics['response_time']:.3f}s")
            
            # Send to monitoring service
            # send_metrics(metrics)
            
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(interval)

if __name__ == "__main__":
    monitor_api("http://your-api-url:8001")
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Container Won't Start

```bash
# Check container logs
az container logs --resource-group garch-model-rg \
  --name garch-api-container

# Check events
az container show --resource-group garch-model-rg \
  --name garch-api-container \
  --query instanceView.events
```

#### 2. ACR Authentication Failed

```bash
# Get ACR credentials
az acr credential show --name garchmodelregistry

# Test login
docker login garchmodelregistry.azurecr.io \
  -u USERNAME -p PASSWORD
```

#### 3. High Memory Usage

```bash
# Restart container
az container restart --resource-group garch-model-rg \
  --name garch-api-container

# Scale up
az container create ... --memory 8  # Increase to 8GB
```

#### 4. API Not Responding

```bash
# Check if port is open
curl -v http://container-ip:8001/health

# Check network security
az network nsg rule list --resource-group garch-model-rg

# Verify DNS
nslookup garch-api.eastus.azurecontainer.io
```

### Debug Mode

Enable debug logging:

```python
# In prediction_api.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

```bash
# Redeploy with debug
az container create ... \
  --environment-variables LOG_LEVEL=DEBUG
```

## 💰 Cost Estimation

### Monthly Azure Costs (Estimated)

| Service | Configuration | Cost/Month |
|---------|--------------|------------|
| Container Instances | 2 vCPU, 4GB RAM | $73 |
| Container Registry | Basic | $5 |
| Storage Account | Standard LRS, 10GB | $0.20 |
| Log Analytics | 5GB ingestion | $12 |
| Monitor Alerts | 5 rules | Free |
| **Total** | | **~$90/month** |

### Cost Optimization Tips

1. **Use Spot Instances**: 70% cheaper for non-critical workloads
2. **Auto-shutdown**: Stop containers during off-hours
3. **Reserved Instances**: Save 30-40% with 1-year commitment
4. **Right-sizing**: Monitor and adjust CPU/memory
5. **Data Retention**: Reduce log retention from 30 to 7 days

```bash
# Auto-shutdown schedule
az container create ... \
  --restart-policy OnFailure \
  --environment-variables AUTO_SHUTDOWN_TIME=20:00
```

## 🔐 Security Best Practices

### 1. Secrets Management

```bash
# Use Azure Key Vault
az keyvault create --name garch-secrets-kv \
  --resource-group garch-model-rg

# Store secrets
az keyvault secret set --vault-name garch-secrets-kv \
  --name acr-password --value "YOUR_PASSWORD"

# Reference in container
az container create ... \
  --secrets key-vault-id=/subscriptions/.../garch-secrets-kv
```

### 2. Network Security

```bash
# Create Virtual Network
az network vnet create --name garch-vnet \
  --resource-group garch-model-rg

# Deploy container in VNet
az container create ... \
  --vnet garch-vnet --subnet default
```

### 3. RBAC

```bash
# Assign minimal permissions
az role assignment create \
  --assignee user@example.com \
  --role "Container Instance Contributor" \
  --resource-group garch-model-rg
```

## 📈 Scaling Strategies

### Horizontal Scaling (Multiple Containers)

```bash
# Deploy multiple instances
for i in {1..3}; do
  az container create \
    --name garch-api-$i \
    --image garchmodelregistry.azurecr.io/garch-api:latest \
    ...
done

# Add load balancer
az network lb create --name garch-lb \
  --resource-group garch-model-rg
```

### Kubernetes (AKS) Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: garch-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: garch-api
  template:
    spec:
      containers:
      - name: api
        image: garchmodelregistry.azurecr.io/garch-api:latest
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: garch-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: garch-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## 🎯 Production Checklist

Before going live:

- [ ] All tests passing (unit, integration, load)
- [ ] Monitoring dashboard configured
- [ ] Alerts set up with proper thresholds
- [ ] Secrets stored in Key Vault
- [ ] Backup and disaster recovery plan
- [ ] Documentation updated
- [ ] Team trained on deployment process
- [ ] Rollback procedure tested
- [ ] Security scan completed
- [ ] Performance benchmarks met
- [ ] Cost budget approved
- [ ] Compliance requirements met

## 📞 Support

For issues:
1. Check logs: `az container logs --name garch-api-container`
2. Review monitoring dashboard
3. Check GitHub Issues
4. Contact: mlops-team@example.com

---

**Last Updated**: December 2024  
**Version**: 1.0.0  
**Maintained by**: MLOps Team</parameter>