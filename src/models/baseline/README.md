# 说明

baseline采用glm4-8b-chat模型，采取如下的提示词
<details>
<summary><b>提示词学习</b> (prompt learning/In Context Learning)</summary>

#### Installing Docker

**macOS:**
```bash
# Install Docker Desktop from https://www.docker.com/products/docker-desktop/
# Or using Homebrew:
brew install --cask docker
```

**Linux (Ubuntu/Debian):**
```bash
# Install Docker Engine
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# Log out and back in for group changes to take effect

# Install Docker Compose
sudo apt-get update
sudo apt-get install docker-compose-plugin
```

#### Verifying Docker Installation
```bash
docker --version
docker compose version
```
</details>
```bash

```
进行few-shot学习，进行预测。

