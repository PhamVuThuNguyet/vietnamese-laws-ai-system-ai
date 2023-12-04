# vietnamese-laws-ai-system-ai

# Getting Started

Getting started developing with this template is pretty simple using docker and docker-compose.

## Clone the repository

```
git clone https://github.com/VKU-NewEnergy/vietnamese-laws-ai-system-ai.git
```

## cd into project root

```
cd vietnamese-laws-ai-system-ai
```

## Launch app
#### 1. Prepare environment
- Create `.env` file in the root directory.
- Copy content from `.env.example` and change to your correct data.
#### 2. Start app
#### 2.1 Using `make` command
```bash
make all
```
#### 2.2 Run locally
```
pip install -r requirements.txt
python main.py
```
#### 2.3 Using docker

> :warning: Please make sure that you have Docker installed.
```
docker compose up
```

Afterwards, FastAPI automatically generates documentation based on the specification of the endpoints you have written. You can find the docs at http://localhost:9000/docs.
