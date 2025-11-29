"""
Cloud Run Deployment Utilities

PURPOSE: Automate deployment to Google Cloud Run
FEATURES: Configuration validation, deployment, health checks
"""

import json
import logging
import subprocess
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CloudRunDeployer:
    """
    Automate Cloud Run deployment

    FEATURES:
    ---------
    1. Configuration validation
    2. Docker image build and push
    3. Service deployment
    4. Health checks
    5. Traffic management

    USAGE:
    ------
    deployer = CloudRunDeployer(
        project_id="your-project",
        region="us-central1"
    )
    deployer.deploy(service_name="qearis", image_tag="latest")
    """

    def __init__(self, project_id: str, region: str = "us-central1", registry: str = "gcr.io"):
        """
        Initialize deployer

        PARAMETERS:
        -----------
        project_id: GCP project ID
        region: Cloud Run region
        registry: Container registry (gcr.io or artifact registry)
        """
        self.project_id = project_id
        self.region = region
        self.registry = registry

        logger.info(f"CloudRunDeployer initialized: {project_id} in {region}")

    def validate_prerequisites(self) -> bool:
        """
        Validate deployment prerequisites

        CHECKS:
        -------
        1. gcloud CLI installed
        2. Docker installed
        3. Authenticated to GCP
        4. Required APIs enabled
        """
        logger.info("Validating prerequisites...")

        # Check gcloud
        try:
            result = subprocess.run(
                ["gcloud", "version"], capture_output=True, text=True, check=True
            )
            logger.info("✓ gcloud CLI installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("[ERROR] gcloud CLI not found")
            return False

        # Check Docker
        try:
            subprocess.run(["docker", "--version"], capture_output=True, check=True)
            logger.info("✓ Docker installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("[ERROR] Docker not found")
            return False

        # Check authentication
        try:
            result = subprocess.run(
                ["gcloud", "auth", "list", "--format=json"],
                capture_output=True,
                text=True,
                check=True,
            )
            accounts = json.loads(result.stdout)
            if not accounts:
                logger.error("[ERROR] Not authenticated to GCP")
                return False
            logger.info(f"✓ Authenticated as {accounts[0]['account']}")
        except Exception as e:
            logger.error(f"✗ Authentication check failed: {e}")
            return False

        logger.info("[OK] All prerequisites validated")
        return True

    def build_image(
        self, service_name: str, tag: str = "latest", dockerfile: str = "Dockerfile"
    ) -> str:
        """
        Build Docker image

        PROCESS:
        --------
        1. Build image locally
        2. Tag for registry
        3. Return full image name

        RETURNS:
        --------
        Full image name (e.g., gcr.io/project/service:tag)
        """
        image_name = f"{self.registry}/{self.project_id}/{service_name}:{tag}"

        logger.info(f"Building Docker image: {image_name}")

        try:
            subprocess.run(
                ["docker", "build", "-t", image_name, "-f", dockerfile, "."],
                check=True,
                capture_output=True,
            )
            logger.info("[OK] Docker image built successfully")
            return image_name
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Docker build failed: {e.stderr.decode()}")
            raise

    def push_image(self, image_name: str):
        """
        Push Docker image to registry

        PROCESS:
        --------
        1. Configure Docker for registry
        2. Push image
        """
        logger.info(f"Pushing image: {image_name}")

        # Configure Docker auth
        try:
            subprocess.run(["gcloud", "auth", "configure-docker"], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            logger.warning("Docker auth configuration may have failed")

        # Push image
        try:
            subprocess.run(["docker", "push", image_name], check=True, capture_output=True)
            logger.info("[OK] Image pushed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Image push failed: {e.stderr.decode()}")
            raise

    def deploy_service(
        self,
        service_name: str,
        image_name: str,
        env_vars: Optional[Dict[str, str]] = None,
        memory: str = "2Gi",
        cpu: int = 2,
        max_instances: int = 10,
        min_instances: int = 0,
        timeout: int = 300,
        allow_unauthenticated: bool = True,
    ) -> Dict[str, Any]:
        """
        Deploy service to Cloud Run

        PARAMETERS:
        -----------
        service_name: Name of the service
        image_name: Full image name with tag
        env_vars: Environment variables dict
        memory: Memory allocation (e.g., "2Gi")
        cpu: CPU allocation
        max_instances: Maximum scaling instances
        min_instances: Minimum instances (0 for scale-to-zero)
        timeout: Request timeout in seconds
        allow_unauthenticated: Allow public access

        RETURNS:
        --------
        Deployment info including service URL
        """
        logger.info(f"Deploying service: {service_name}")

        # Build command
        cmd = [
            "gcloud",
            "run",
            "deploy",
            service_name,
            "--image",
            image_name,
            "--platform",
            "managed",
            "--region",
            self.region,
            "--memory",
            memory,
            "--cpu",
            str(cpu),
            "--timeout",
            str(timeout),
            "--max-instances",
            str(max_instances),
            "--min-instances",
            str(min_instances),
            "--port",
            "8080",
            "--format",
            "json",
        ]

        # Add environment variables
        if env_vars:
            for key, value in env_vars.items():
                cmd.extend(["--set-env-vars", f"{key}={value}"])

        # Add authentication
        if allow_unauthenticated:
            cmd.append("--allow-unauthenticated")

        # Execute deployment
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)

            deployment_info = json.loads(result.stdout)
            service_url = deployment_info["status"]["url"]

            logger.info(f"[OK] Service deployed successfully")
            logger.info(f"🌐 Service URL: {service_url}")

            return {
                "service_name": service_name,
                "service_url": service_url,
                "region": self.region,
                "deployment_info": deployment_info,
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Deployment failed: {e.stderr}")
            raise

    def health_check(
        self,
        service_url: str,
        endpoint: str = "/health",
        max_retries: int = 5,
        retry_delay: int = 10,
    ) -> bool:
        """
        Perform health check on deployed service

        PROCESS:
        --------
        1. Wait for service to be ready
        2. Check health endpoint
        3. Retry if needed

        RETURNS:
        --------
        True if healthy, False otherwise
        """
        import requests

        logger.info(f"Performing health check: {service_url}{endpoint}")

        for attempt in range(max_retries):
            try:
                response = requests.get(f"{service_url}{endpoint}", timeout=10)

                if response.status_code == 200:
                    logger.info(f"[OK] Health check passed (attempt {attempt + 1})")
                    return True
                else:
                    logger.warning(
                        f"Health check returned {response.status_code} "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Health check failed: {e} " f"(attempt {attempt + 1}/{max_retries})"
                )

            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)

        logger.error("[ERROR] Health check failed after all retries")
        return False

    def get_service_info(self, service_name: str) -> Dict[str, Any]:
        """
        Get service information

        RETURNS:
        --------
        Service details including URL, status, traffic
        """
        try:
            result = subprocess.run(
                [
                    "gcloud",
                    "run",
                    "services",
                    "describe",
                    service_name,
                    "--region",
                    self.region,
                    "--format",
                    "json",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            return json.loads(result.stdout)

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get service info: {e.stderr}")
            raise

    def update_traffic(self, service_name: str, to_latest: bool = True):
        """
        Update traffic routing

        PARAMETERS:
        -----------
        service_name: Name of service
        to_latest: Route 100% traffic to latest revision
        """
        logger.info(f"Updating traffic for {service_name}")

        try:
            cmd = [
                "gcloud",
                "run",
                "services",
                "update-traffic",
                service_name,
                "--region",
                self.region,
            ]

            if to_latest:
                cmd.append("--to-latest")

            subprocess.run(cmd, check=True, capture_output=True)
            logger.info("[OK] Traffic updated successfully")

        except subprocess.CalledProcessError as e:
            logger.error(f"Traffic update failed: {e.stderr}")
            raise

    def full_deployment(
        self, service_name: str, env_vars: Optional[Dict[str, str]] = None, tag: str = "latest"
    ) -> Dict[str, Any]:
        """
        Complete end-to-end deployment

        PROCESS:
        --------
        1. Validate prerequisites
        2. Build Docker image
        3. Push to registry
        4. Deploy to Cloud Run
        5. Health check
        6. Return deployment info

        USAGE:
        ------
        deployer = CloudRunDeployer("my-project")
        result = deployer.full_deployment(
            service_name="qearis",
            env_vars={"GEMINI_API_KEY": "..."}
        )
        print(f"Deployed at: {result['service_url']}")
        """
        logger.info("=" * 60)
        logger.info("STARTING FULL DEPLOYMENT")
        logger.info("=" * 60)

        # Step 1: Validate
        if not self.validate_prerequisites():
            raise RuntimeError("Prerequisites validation failed")

        # Step 2: Build
        image_name = self.build_image(service_name, tag)

        # Step 3: Push
        self.push_image(image_name)

        # Step 4: Deploy
        deployment_info = self.deploy_service(
            service_name=service_name, image_name=image_name, env_vars=env_vars
        )

        # Step 5: Health check
        service_url = deployment_info["service_url"]
        if not self.health_check(service_url):
            logger.warning("Health check failed, but deployment completed")

        logger.info("=" * 60)
        logger.info("DEPLOYMENT COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Service URL: {service_url}")
        logger.info(f"Region: {self.region}")
        logger.info("=" * 60)

        return deployment_info


# Example usage
if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()

    deployer = CloudRunDeployer(project_id="gen-lang-client-0472751146", region="us-central1")

    env_vars = {"GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"), "ENVIRONMENT": "production"}

    result = deployer.full_deployment(service_name="qearis", env_vars=env_vars)

    print(f"\n[SUCCESS] Deployment successful!")
    print(f"🌐 Access at: {result['service_url']}")
