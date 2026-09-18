# Deployment runbook

The browser application uses HTTPS, so the API also needs HTTPS. Point a DNS
name such as `mnist-api.example.com` at the VPS before the final deployment.
The Docker service binds to localhost; the VPS Nginx install terminates TLS and
forwards requests.

## VPS API

Prerequisites:

- A completed Colab run with `final_fc.keras` and `model_metadata.json`
- SSH access through `ssh gpl_server`
- A DNS record for the API hostname
- The VPS Nginx install, already present on `gpl_server`

Clone the repository on the VPS and stage the model. `final_fc.keras` is
committed, at 2.4 MB, so there is nothing to copy over separately:

```bash
ssh gpl_server
git clone https://github.com/MwangiNelson/mnist-network-lab.git /opt/mnist-network-lab
cd /opt/mnist-network-lab
mkdir -p deploy/models
cp artifacts/final_fc.keras artifacts/model_metadata.json deploy/models/
```

To update later, `git pull` and repeat the `cp`, then rebuild.

On the VPS, copy `deploy/.env.example` to `deploy/.env` and replace the example
Vercel origin. Install the vhost, substituting the real API hostname:

```bash
sudo cp deploy/nginx.conf.example /etc/nginx/sites-available/mnist-api
sudo sed -i 's/mnist-api.example.com/<real hostname>/' /etc/nginx/sites-available/mnist-api
sudo ln -sf /etc/nginx/sites-available/mnist-api /etc/nginx/sites-enabled/mnist-api
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d <real hostname>
```

Then start the service:

```bash
cd /opt/mnist-network-lab/deploy
docker compose up -d --build
curl http://127.0.0.1:8010/health
```

Verify the public endpoint:

```bash
curl https://mnist-api.example.com/health
```

Rollback with the previous Git commit and model file, then rebuild the Compose
service. The API stores no user data.

## Vercel web application

Import the GitHub repository in Vercel and use `web` as the root directory. Add
this production environment variable:

```text
VITE_API_URL=https://mnist-api.example.com
```

Deploy, then update `CORS_ORIGINS` on the VPS with the final Vercel URL and
restart the API.
